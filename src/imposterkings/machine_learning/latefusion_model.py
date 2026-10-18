"""Late-fusion attention Q-net (Study 11): encode the state ONCE, let every legal move cross-attend to it.

    state tokens (features3, S=23) --[embed]--> E0 --[n_self pre-norm self-attention blocks]--> ctx
    candidates (pointer records)   --[kind + pointed-at E0 rows]--> c0 --[n_cross cross blocks over ctx]--> y
    y --> q = tanh(w_q . y)  (search value of the move)      y --> logit = w_p . y  (policy, softmax over moves)

The design is the Attentris pattern (board self-attention, candidates cross-attend). What it buys over the
early-fusion ``attention_model`` (where the move is an input token from layer 0):

- **One state pass per decision.** Candidates never re-encode the state, and they never attend to each other,
  so a move's score is f(state, move) only, and padding / other candidates cannot change it.
- **Shared values.** Every candidate in every cross block reads the SAME value vectors (projections of the
  shared ``ctx``); two moves differ only in their own embedding and in WHERE they look. With no final
  LayerNorm the pre-tanh q is exactly linear in the candidate stream, so

      q_pre(a) = b + w.c0(a) + sum_L [ w.wo_L.bias + sum_{h,j} A_L^h[a,j] * (u_L^h . v_L^h[j]) ] + sum_L w.FFN_L(a)

  with ``u_L = w_q @ Wo_L`` split by head (:func:`decompose`). The A.(u.v) term is the cross-attention DLA;
  its v is shared across candidates, so a "why a over b" contrast is (A[a,j] - A[b,j]) * (u.v_j).
- **Pointers keep the move's identity unmixed.** A candidate is built from layer-0 embeddings of the card /
  king / decision token it names (Attentris builds its candidates from raw cell embeddings), plus a learned
  per-head bias toward the keys it points at.

V(s) = max_a q (option (a) of the Study 11 design): no CLS token. ``evaluator_from_model`` uses it as the
leaf value and softmax(logit) as the PUCT prior.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..actions import Action
from ..infoset import InformationSet
from ..rules import NUM_PLAYERS
from . import features3 as F3
from .attention_model import AttnConfig, EncoderLayer

MODEL_TYPE = "latefusion"
_TYPE_CARD, _TYPE_DECISION = 0, 5               # type ids: card | hand mine/theirs | king mine/theirs | decision
_STATE_TYPES = [_TYPE_CARD] * 18 + [1, 2, 3, 4, _TYPE_DECISION]


@dataclass(frozen=True)
class LFConfig:
    d_model: int = 96
    n_self: int = 1               # self-attention blocks over the 23 state tokens
    n_cross: int = 1              # cross-attention blocks (candidates -> state)
    n_heads: int = 6
    ffn_hidden: int = 384
    dropout: float = 0.1
    feat: str = "v3"

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.feat != "v3":
            raise ValueError("the late-fusion model reads features3 (feat='v3') only")


class CrossLayer(nn.Module):
    """Pre-norm cross block: c = c + Wo.Attn(LN(c) -> LN_kv(ctx)); c = c + FFN(LN(c)). The keys the candidate
    POINTS at get a learned per-head score bonus (``ref_bias``). No candidate-candidate attention."""

    def __init__(self, cfg: LFConfig):
        super().__init__()
        d = cfg.d_model
        self.h, self.dh = cfg.n_heads, d // cfg.n_heads
        self.ln1, self.ln_kv, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(d, d) for _ in range(4))
        self.ref_bias = nn.Parameter(torch.zeros(cfg.n_heads))
        self.ffn = nn.Sequential(nn.Linear(d, cfg.ffn_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
                                 nn.Linear(cfg.ffn_hidden, d))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, c, ctx, ref, score_bias=None, rec: Optional[dict] = None):
        b, n, d = c.shape
        s = ctx.shape[1]
        h = self.ln1(c)
        kv = self.ln_kv(ctx)
        q = self.wq(h).view(b, n, self.h, self.dh).transpose(1, 2)                 # [B,H,N,dh]
        k = self.wk(kv).view(b, s, self.h, self.dh).transpose(1, 2)                # [B,H,S,dh]
        v = self.wv(kv).view(b, s, self.h, self.dh).transpose(1, 2)                # [B,H,S,dh]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh) + self.ref_bias.view(1, -1, 1, 1) * ref
        if score_bias is not None:                                                  # interventions (knockout)
            scores = scores + score_bias
        attn = torch.softmax(scores, dim=-1)                                        # [B,H,N,S]
        concat = (self.drop(attn) @ v).transpose(1, 2).reshape(b, n, d)
        a = self.wo(concat)
        c_mid = c + self.drop(a)
        f = self.ffn(self.ln2(c_mid))
        out = c_mid + self.drop(f)
        if rec is not None:
            rec.setdefault("cross_in", []).append(c)
            rec.setdefault("cross_v", []).append(v)
            rec.setdefault("cross_attn", []).append(attn)
            rec.setdefault("cross_concat", []).append(concat)
            rec.setdefault("cross_ffn", []).append(f)
        return out, attn


class LateFusionModel(nn.Module):
    is_latefusion = True

    def __init__(self, cfg: LFConfig = LFConfig()):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        def proj(n):
            return nn.Sequential(nn.Linear(n, d), nn.LayerNorm(d))

        self.card, self.hand = proj(F3.CARD_DIM), proj(F3.HAND_DIM)
        self.king, self.decision = proj(F3.KING_DIM), proj(F3.DECISION_DIM)
        self.type_emb = nn.Embedding(6, d)
        self.depth_emb = nn.Embedding(F3.DEPTH_CAP + 2, d)        # index depth + 1 (0 = off-stack)
        self.src_proj = nn.Linear(d, d, bias=False)               # decision token += P . E0[source card]
        enc_cfg = AttnConfig(d_model=d, n_layers=cfg.n_self, n_heads=cfg.n_heads, ffn_hidden=cfg.ffn_hidden,
                             dropout=cfg.dropout, feat="v2")
        self.layers = nn.ModuleList([EncoderLayer(enc_cfg) for _ in range(cfg.n_self)])
        # candidates
        self.kind_emb = nn.Embedding(F3.N_KINDS, d)
        self.num_emb = nn.Embedding(10, d)                         # index number + 1 (0 = none)
        self.p_a, self.p_b = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.p_name, self.p_num = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        nm = torch.from_numpy(F3.NAME_MASK)
        vm = torch.from_numpy(F3.VALUE_MASK)
        self.register_buffer("name_avg", nm / nm.sum(1, keepdim=True).clamp(min=1), persistent=False)
        self.register_buffer("value_avg", vm / vm.sum(1, keepdim=True).clamp(min=1), persistent=False)
        self.register_buffer("name_mask", nm, persistent=False)
        self.register_buffer("value_mask", vm, persistent=False)
        self.register_buffer("state_types", torch.tensor(_STATE_TYPES), persistent=False)
        self.cross = nn.ModuleList([CrossLayer(cfg) for _ in range(cfg.n_cross)])
        self.head_q = nn.Linear(d, 1)
        self.head_p = nn.Linear(d, 1)

    # --- pieces ------------------------------------------------------------------------------------------

    def embed_state(self, b) -> torch.Tensor:
        """E0 [B, 23, d]: per-type projections + type / stack-depth embeddings + the source pointer."""
        types = self.type_emb(self.state_types)                                   # [23, d]
        card = self.card(b["cards"]) + types[:18] + self.depth_emb(b["card_depth"].long() + 1)
        hand = self.hand(b["hands"]) + types[18:20]
        king = self.king(b["kings"]) + types[20:22]
        src = b["src_ptr"].long()
        src_e = card[torch.arange(card.shape[0]), src.clamp(min=0)] * (src >= 0).unsqueeze(-1)
        dec = (self.decision(b["decision"]) + types[22] + self.src_proj(src_e)).unsqueeze(1)
        return torch.cat([card, hand, king, dec], dim=1)

    def _gather(self, e0, ptr):
        idx = ptr.long()
        g = torch.gather(e0, 1, idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, e0.shape[-1]))
        return g * (idx >= 0).unsqueeze(-1)

    def embed_candidates(self, e0, b) -> Tuple[torch.Tensor, torch.Tensor]:
        """(c0 [B,N,d], ref [B,1,N,S]): each candidate's embedding and the keys it points at."""
        kind, pa, pb = b["cand_kind"].long(), b["cand_ptr_a"].long(), b["cand_ptr_b"].long()
        name, num = b["cand_name"].long(), b["cand_number"].long()
        cards0 = e0[:, :18]
        has_name, has_num = (name >= 0).unsqueeze(-1), (num >= 0).unsqueeze(-1)
        name_e = (self.name_avg[name.clamp(min=0)] @ cards0) * has_name           # [B,N,d]
        num_e = (self.value_avg[num.clamp(min=0)] @ cards0) * has_num
        c0 = (self.kind_emb(kind.clamp(min=0)) + self.p_a(self._gather(e0, pa)) + self.p_b(self._gather(e0, pb))
              + self.p_name(name_e) + self.p_num(num_e) + self.num_emb(num + 1))
        S = e0.shape[1]
        ref = F.one_hot(pa.clamp(min=0), S) * (pa >= 0).unsqueeze(-1) + F.one_hot(pb.clamp(min=0), S) * (pb >= 0).unsqueeze(-1)
        ref = ref.float()
        ref[..., :18] += self.name_mask[name.clamp(min=0)] * has_name + self.value_mask[num.clamp(min=0)] * has_num
        return c0, ref.clamp(max=1.0).unsqueeze(1)

    # --- forward -----------------------------------------------------------------------------------------

    def forward(self, b, rec: Optional[dict] = None, cross_bias: Optional[List[torch.Tensor]] = None) -> Dict:
        """``b``: a :func:`collate` batch. Returns q / q_pre / logit [B,N] (logit = -inf on padding) and the
        attention maps. ``rec`` (a dict) collects the internals :func:`decompose` needs; ``cross_bias`` is an
        optional per-cross-layer additive score bias [B|1, H|1, N|1, S] (knockouts)."""
        e0 = self.embed_state(b)
        x = e0
        attn_self = []
        for layer in self.layers:
            x, a = layer(x)
            attn_self.append(a)
        c, ref = self.embed_candidates(e0, b)
        if rec is not None:
            rec["e0"], rec["ctx"], rec["c0"] = e0, x, c
        attn_cross = []
        for li, layer in enumerate(self.cross):
            c, a = layer(c, x, ref, None if cross_bias is None else cross_bias[li], rec)
            attn_cross.append(a)
        pad = b["cand_kind"] < 0
        q_pre = self.head_q(c).squeeze(-1)
        q = torch.tanh(q_pre).masked_fill(pad, 0.0)
        logit = self.head_p(c).squeeze(-1).masked_fill(pad, float("-inf"))
        if rec is not None:
            rec["y"] = c
        return {"q": q, "q_pre": q_pre, "logit": logit, "attn_self": attn_self, "attn_cross": attn_cross}

    @torch.no_grad()
    def score(self, view: InformationSet, moves: Sequence[Action]) -> Tuple[np.ndarray, np.ndarray]:
        """(q, logit) for the decider's ``moves`` -- one state encode, one batch of candidates."""
        st, cd = F3.tokenize3(view, moves)
        out = self(collate([st], [cd]))
        n = len(moves)
        return out["q"][0, :n].numpy(), out["logit"][0, :n].numpy()

    def q_moves(self, view: InformationSet, moves: Sequence[Action]) -> np.ndarray:
        return self.score(view, moves)[0]


# --- batching ----------------------------------------------------------------------------------------------

_STATE_KEYS = ("cards", "card_depth", "hands", "kings", "decision", "src_ptr")
CAND_KEYS = ("cand_kind", "cand_ptr_a", "cand_ptr_b", "cand_name", "cand_number")


def collate(states: Sequence[F3.State3], cands: Sequence[F3.Cands3]) -> Dict[str, torch.Tensor]:
    """Live positions -> a padded batch (N = the largest candidate set; padding has kind -1)."""
    n = max(len(c.kind) for c in cands)
    out = {"cards": torch.from_numpy(np.stack([s.cards for s in states])),
           "card_depth": torch.from_numpy(np.stack([s.card_depth for s in states]).astype(np.int64)),
           "hands": torch.from_numpy(np.stack([s.hands for s in states])),
           "kings": torch.from_numpy(np.stack([s.kings for s in states])),
           "decision": torch.from_numpy(np.stack([s.decision for s in states])),
           "src_ptr": torch.tensor([s.src_ptr for s in states], dtype=torch.long)}
    for key, field in zip(CAND_KEYS, ("kind", "ptr_a", "ptr_b", "name", "number")):
        arr = np.full((len(cands), n), -1, np.int64)
        for i, c in enumerate(cands):
            v = getattr(c, field)
            arr[i, :len(v)] = v
        out[key] = torch.from_numpy(arr)
    return out


def batch_from_arrays(arrs: Dict[str, np.ndarray], idx: np.ndarray, trim: bool = True) -> Dict[str, torch.Tensor]:
    """Rows ``idx`` of a v3 (per-decision) dataset -> a batch; ``trim`` drops all-padding candidate columns.
    Labels (y, w, chosen) ride along when present."""
    kind = arrs["cand_kind"][idx]
    n = int((kind >= 0).sum(1).max()) if trim else kind.shape[1]
    out = {"cards": torch.from_numpy(arrs["cards"][idx]),
           "card_depth": torch.from_numpy(arrs["card_depth"][idx].astype(np.int64)),
           "hands": torch.from_numpy(arrs["hands"][idx]), "kings": torch.from_numpy(arrs["kings"][idx]),
           "decision": torch.from_numpy(arrs["decision"][idx]),
           "src_ptr": torch.from_numpy(arrs["src_ptr"][idx].astype(np.int64))}
    for k in CAND_KEYS:
        out[k] = torch.from_numpy(arrs[k][idx, :n].astype(np.int64))
    for k in ("y", "w"):
        if k in arrs:
            out[k] = torch.from_numpy(arrs[k][idx, :n])
    if "chosen" in arrs:
        out["chosen"] = torch.from_numpy(arrs["chosen"][idx].astype(np.int64))
    return out


def to_device(b: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in b.items()}


# --- exact decomposition (the cross-attention DLA) -----------------------------------------------------------

@dataclass
class Decomp:
    """Per-candidate split of the pre-tanh q. ``attn[B,Lc,H,N,S]`` is the cross-attention DLA (shared values),
    ``cand[B,N]`` the candidate's own embedding (w.c0), ``ffn[B,Lc,N]`` the cross FFNs, ``const[B,N]`` the
    biases. ``resid`` = q_pre - sum(components) (float32 closure)."""
    q_pre: torch.Tensor
    const: torch.Tensor
    cand: torch.Tensor
    attn: torch.Tensor
    ffn: torch.Tensor
    resid: torch.Tensor


@torch.no_grad()
def decompose(model: LateFusionModel, b, cross_bias=None) -> Decomp:
    rec: dict = {}
    out = model(b, rec=rec, cross_bias=cross_bias)
    w = model.head_q.weight[0]                                                      # [d]
    const = model.head_q.bias[0] + sum(w @ layer.wo.bias for layer in model.cross)
    cand = rec["c0"] @ w                                                            # [B,N]
    attn_terms, ffn_terms = [], []
    for li, layer in enumerate(model.cross):
        u = (w @ layer.wo.weight).view(layer.h, layer.dh)                           # [H,dh]
        uv = torch.einsum("bhsd,hd->bhs", rec["cross_v"][li], u)                     # [B,H,S]
        attn_terms.append(rec["cross_attn"][li] * uv.unsqueeze(2))                   # [B,H,N,S]
        ffn_terms.append(rec["cross_ffn"][li] @ w)                                   # [B,N]
    attn = torch.stack(attn_terms, 1)
    ffn = torch.stack(ffn_terms, 1)
    total = const + cand + attn.sum((1, 2, 4)) + ffn.sum(1)
    pad = b["cand_kind"] < 0
    resid = (out["q_pre"] - total).masked_fill(pad, 0.0)
    return Decomp(out["q_pre"], const.expand_as(cand), cand, attn, ffn, resid)


# --- checkpoint + evaluator ------------------------------------------------------------------------------------

def save(path: str, model: LateFusionModel, meta: Optional[dict] = None) -> None:
    torch.save({"model_type": MODEL_TYPE, "config": asdict(model.cfg), "state_dict": model.state_dict(),
                "meta": meta or {}}, path)


def load(path: str, device: str = "cpu") -> Tuple[LateFusionModel, dict]:
    blob = torch.load(path, map_location=device, weights_only=False)
    model = LateFusionModel(LFConfig(**blob["config"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob.get("meta", {})


def evaluator_from_model(model: LateFusionModel, prior: str = "policy"):
    """`state -> ([per-seat value], {move: prior})`: value = max_a q (mover-relative, negated for the other
    seat), prior = softmax(policy logit) (``prior="q"``: softmax(q), the early-fusion convention)."""
    if prior not in ("policy", "q"):
        raise ValueError(f"unknown prior {prior!r}")

    def evaluate(state):
        mover = state.to_play
        moves = state.legal_moves()
        q, logit = model.score(state.information_set(mover), moves)
        v = float(q.max())
        value = [0.0] * NUM_PLAYERS
        value[mover] = v
        value[1 - mover] = -v
        z = logit if prior == "policy" else q
        e = np.exp(z - z.max())
        p = e / e.sum()
        return value, {m: float(x) for m, x in zip(moves, p)}

    return evaluate


def build_evaluator(checkpoint: str, device: str = "cpu", prior: str = "policy"):
    model, _ = load(checkpoint, device)
    return evaluator_from_model(model, prior)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
