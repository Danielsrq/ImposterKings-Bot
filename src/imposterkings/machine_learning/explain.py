"""Headless interpretability adapter: ``explain(view, action, model) -> AttentionExplanation``.

One forward pass of the attention q-net on a single ``(InformationSet, candidate action)`` yields the
per-head CLS->token attention that "explains" scoring that move. Torch-only (NO pygame) so it is the single
shared entry point for both the live UI drawer and the post-game review, and is unit-testable with no display.

The sequence axis is ``S = N + 4``, laid out ``[CLS, N card tokens, board, phase, action]``; ``seq_labels``
mirror that order. At ``L=1`` only row 0 (CLS) causally feeds ``q``; at ``L>=2`` layer-1's card rows feed
layer-2's CLS (``per_layer`` carries every layer so the renderer can route accordingly).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from ..cards import CARD_NAMES, asset_path, card_ids_for_name
from .attention_model import AttentionModel, collate
from .features import CAND_COL, tokenize

_CTX_LABELS = ["board", "phase", "action"]                          # the 3 singleton tokens (no card art)
_NAME_TO_ASSET: Dict[str, str] = {n: asset_path(card_ids_for_name(n)[0]) for n in CARD_NAMES}


@dataclass(frozen=True)
class AttentionExplanation:
    q: float                                 # value-head output for THIS (view, action)
    seq_labels: List[str]                    # len S; [0]="CLS", [-3:]=["board","phase","action"]
    attn: np.ndarray                         # [heads, S, S] float32 -- last (CLS-readout) layer
    per_layer: Optional[List[np.ndarray]]    # per-layer [heads, S, S], or None when not requested
    candidate_seq_index: Optional[int]       # primary candidate token seq index, or None
    candidate_seq_indices: List[int]         # all candidate token seq indices (a play+guess can flag two)
    display_names: List[Optional[str]]       # len S; parsed CardName per seq idx ("*" stripped) or None
    name_to_asset: Dict[str, str]            # CardName -> assets/ filename (the 14 names)
    n_heads: int
    n_layers: int
    ckpt_id: str                             # checkpoint fingerprint, for cache-keying in review


def _parse_name(label: str) -> Optional[str]:
    """``"zone:CardName"`` (possibly with a trailing ``"*"`` for a synthetic candidate) -> CardName if it
    is a real card, else None (CLS / board / phase / action / ``opp_unknown:?``)."""
    if ":" not in label:
        return None
    name = label.split(":", 1)[1]
    if name.endswith("*"):
        name = name[:-1]
    return name if name in _NAME_TO_ASSET else None


@torch.no_grad()
def explain(view, action, model: AttentionModel, *, all_layers: bool = False,
            ckpt_id: str = "") -> AttentionExplanation:
    """Run one forward pass for ``(view, action)`` and package the per-head CLS->token attention.

    ``view`` an InformationSet, ``action`` the candidate move to explain (or None), ``model`` an
    already-loaded :class:`AttentionModel`. With ``all_layers`` the payload carries every layer's attention
    (for L>=2 card-row routing); otherwise only the last (readout) layer. Deterministic (eval, no dropout)."""
    model.eval()
    tok = tokenize(view, action)
    batch = collate([tok])
    args = (batch["cards"], batch["board"], batch["phase"], batch["action"], batch["card_mask"])
    if all_layers:
        q_t, attns_t = model.forward_layers(*args)
        per_layer: Optional[List[np.ndarray]] = [a[0].cpu().numpy().astype(np.float32) for a in attns_t]
        attn = per_layer[-1]
    else:
        q_t, attn_t = model(*args)
        per_layer = None
        attn = attn_t[0].cpu().numpy().astype(np.float32)

    seq_labels = ["CLS"] + list(tok.labels) + list(_CTX_LABELS)
    display_names = [_parse_name(lb) for lb in seq_labels]

    # Candidate token(s): the is_candidate_action column of the (padded) card block -- catches both the
    # matched token and the synthetic "*" claim token. Card-region index j -> seq index 1 + j.
    n_cards = len(tok.labels)
    cand_col = batch["cards"][0, :, CAND_COL].cpu().numpy()
    cand = [1 + j for j in np.flatnonzero(cand_col == 1.0).tolist() if j < n_cards]
    if not cand:                                                   # fallback: any "*"-suffixed label
        cand = [i for i, lb in enumerate(seq_labels) if lb.endswith("*")]
    # Primary: prefer the my_hand play token; else the first flagged token.
    primary = next((i for i in cand if seq_labels[i].startswith("my_hand:")), cand[0] if cand else None)

    return AttentionExplanation(
        q=float(q_t.item()), seq_labels=seq_labels, attn=attn, per_layer=per_layer,
        candidate_seq_index=primary, candidate_seq_indices=cand, display_names=display_names,
        name_to_asset=dict(_NAME_TO_ASSET), n_heads=int(attn.shape[0]), n_layers=model.cfg.n_layers,
        ckpt_id=ckpt_id)
