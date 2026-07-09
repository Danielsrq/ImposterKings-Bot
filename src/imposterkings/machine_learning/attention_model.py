"""Explainable attention model: an action-in q-net over ``features.tokenize`` output.

Per-type input projections lift each token kind (card/board/phase/action) to a shared ``d_model``; a
learned CLS token is prepended; ``L`` pre-norm transformer layers (multi-head self-attention + per-token
FFN, both residual) mix the set; the CLS output feeds a Tanh-bounded value head = ``q(state, action)``.
The last layer's per-head ``S x S`` attention (row 0 = CLS -> every token) is returned as the "which
cards mattered for this move" importance map. Variable-length token sets are batched by padding + a key
mask; there is NO positional encoding (it is a set -- meaning lives in the ``zone`` feature).

Everything is parameterizable via :class:`AttnConfig` (d_model, n_layers, n_heads, ffn_hidden, dropout,
bounded). CPU-only, matching the repo's ML pipeline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .features import ACTION_DIM, BOARD_DIM, CARD_DIM, PHASE_DIM, InformationSet, Tokens, tokenize


@dataclass(frozen=True)
class AttnConfig:
    d_model: int = 64
    n_layers: int = 1              # L
    n_heads: int = 4              # d_model must be divisible by n_heads
    ffn_hidden: int = 128         # per-token FFN hidden width
    dropout: float = 0.0
    bounded: bool = True          # Tanh on the value head (q in [-1, 1], matching mlp.py)

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")


class TokenEmbed(nn.Module):
    """Per-type Linear(native -> d) + LayerNorm for each token kind (the W_card/W_board/... projections)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.card = nn.Sequential(nn.Linear(CARD_DIM, d_model), nn.LayerNorm(d_model))
        self.board = nn.Sequential(nn.Linear(BOARD_DIM, d_model), nn.LayerNorm(d_model))
        self.phase = nn.Sequential(nn.Linear(PHASE_DIM, d_model), nn.LayerNorm(d_model))
        self.action = nn.Sequential(nn.Linear(ACTION_DIM, d_model), nn.LayerNorm(d_model))

    def forward(self, cards, board, phase, action):
        return self.card(cards), self.board(board), self.phase(phase), self.action(action)


class MultiHeadSelfAttention(nn.Module):
    """Explicit W_q/W_k/W_v/W_o so the per-head [B, heads, S, S] attention is exposed for the readout."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, add_mask=None):
        b, s, d = x.shape

        def split(t):                                       # [B,S,d] -> [B,h,S,dh]
            return t.view(b, s, self.h, self.dh).transpose(1, 2)

        q, k, v = split(self.wq(x)), split(self.wk(x)), split(self.wv(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)   # [B,h,S,S]
        if add_mask is not None:
            scores = scores + add_mask                      # additive -inf on padded keys
        attn = torch.softmax(scores, dim=-1)
        out = (self.drop(attn) @ v).transpose(1, 2).reshape(b, s, d)
        return self.wo(out), attn


class EncoderLayer(nn.Module):
    """Pre-norm transformer block: x = x + MHSA(LN(x)); x = x + FFN(LN(x))."""

    def __init__(self, cfg: AttnConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_hidden), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.ffn_hidden, cfg.d_model))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, add_mask=None):
        a, attn = self.attn(self.ln1(x), add_mask)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x, attn


class AttentionModel(nn.Module):
    """[CLS, card tokens, board, phase, action] -> pre-norm attention -> CLS readout -> q, + attention."""

    def __init__(self, cfg: AttnConfig = AttnConfig()):
        super().__init__()
        self.cfg = cfg
        self.embed = TokenEmbed(cfg.d_model)
        self.cls = nn.Parameter(torch.zeros(cfg.d_model))          # learned readout token (no projection)
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.head = nn.Linear(cfg.d_model, 1)

    def forward(self, cards, board, phase, action, card_mask=None):
        """cards [B,N,44], board [B,14], phase [B,53], action [B,23], card_mask [B,N] bool (True=real).
        Returns (q [B], attn [B, heads, S, S]) with S = N + 4 ([CLS] + N cards + board + phase + action)."""
        b, n = cards.shape[0], cards.shape[1]
        hc, hb, hp, ha = self.embed(cards, board, phase, action)
        cls = self.cls.view(1, 1, -1).expand(b, 1, -1)
        seq = torch.cat([cls, hc, hb.unsqueeze(1), hp.unsqueeze(1), ha.unsqueeze(1)], dim=1)   # [B,S,d]
        s = seq.shape[1]

        add_mask = None
        if card_mask is not None:
            ones = torch.ones(b, 1, dtype=torch.bool, device=cards.device)
            valid = torch.cat([ones, card_mask, ones.expand(b, 3)], dim=1)          # [B,S]; CLS/ctx valid
            add_mask = torch.zeros(b, 1, 1, s, device=cards.device)
            add_mask = add_mask.masked_fill(~valid[:, None, None, :], float("-inf"))

        attn = None
        for layer in self.layers:
            seq, attn = layer(seq, add_mask)
        q = self.head(seq[:, 0, :]).squeeze(-1)                     # CLS output -> scalar
        if self.cfg.bounded:
            q = torch.tanh(q)
        return q, attn                                              # attn = last layer, [B, heads, S, S]

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --- batching + interpretability helpers --------------------------------------------------------

def collate(toks: List[Tokens]) -> dict:
    """Pad a list of Tokens into a batch: cards [B,Nmax,44] + card_mask [B,Nmax], board/phase/action
    [B,·], and per-sample labels. Single-sample use is just ``collate([tok])``."""
    b = len(toks)
    nmax = max(1, max(t.cards.shape[0] for t in toks))
    cards = torch.zeros(b, nmax, CARD_DIM)
    card_mask = torch.zeros(b, nmax, dtype=torch.bool)
    board = torch.zeros(b, BOARD_DIM)
    phase = torch.zeros(b, PHASE_DIM)
    action = torch.zeros(b, ACTION_DIM)
    labels: List[List[str]] = []
    for i, t in enumerate(toks):
        k = t.cards.shape[0]
        cards[i, :k] = torch.from_numpy(t.cards)
        card_mask[i, :k] = True
        board[i] = torch.from_numpy(t.board)
        phase[i] = torch.from_numpy(t.phase)
        action[i] = torch.from_numpy(t.action)
        labels.append(t.labels)
    return {"cards": cards, "board": board, "phase": phase, "action": action,
            "card_mask": card_mask, "labels": labels}


def cls_importance(attn: torch.Tensor, labels: List[List[str]]) -> List[List[Tuple[str, float]]]:
    """CLS->card importance (head-averaged row 0, the card region) per batch element, sorted desc.
    ``attn`` [B, heads, S, S]; ``labels`` the per-sample card-token labels. One weight per real card."""
    row0 = attn[:, :, 0, :].mean(dim=1)                             # [B, S] head-averaged CLS row
    out: List[List[Tuple[str, float]]] = []
    for i, labs in enumerate(labels):
        weights = row0[i, 1:1 + len(labs)].tolist()                # card region = positions 1 .. 1+N
        out.append(sorted(zip(labs, weights), key=lambda p: -p[1]))
    return out


@torch.no_grad()
def print_attention(view: InformationSet, model: AttentionModel, action=None) -> None:
    """Tokenize a position, run the model, and print q + the CLS->card importance + per-head row-0 tops."""
    model.eval()
    tok = tokenize(view, action)
    batch = collate([tok])
    q, attn = model(batch["cards"], batch["board"], batch["phase"], batch["action"], batch["card_mask"])
    print(f"q = {q.item():+.3f}   (S={attn.shape[-1]} tokens, {attn.shape[1]} heads)")
    print("CLS->card importance (head-avg):")
    for lb, w in cls_importance(attn, batch["labels"])[0]:
        print(f"  {lb:24s} {w:.3f}")
    seq_labels = ["CLS"] + tok.labels + ["board", "phase", "action"]
    for h in range(attn.shape[1]):
        top = sorted(zip(seq_labels, attn[0, h, 0, :].tolist()), key=lambda p: -p[1])[:5]
        print(f"  head{h} row0 top: " + ", ".join(f"{l}={w:.2f}" for l, w in top))
