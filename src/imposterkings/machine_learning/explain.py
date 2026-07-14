"""Headless interpretability adapter: ``explain(view, action, model) -> AttentionExplanation``.

One forward pass of the attention q-net on a single ``(InformationSet, candidate action)`` yields the
per-head CLS->token attention that "explains" scoring that move. Torch-only (NO pygame) so it is the single
shared entry point for both the live UI drawer and the post-game review, and is unit-testable with no display.

Both featurizations are supported; the checkpoint's ``cfg.feat`` picks the path:

- **v1** -- ``S = N + 4``: ``[CLS, N card tokens, board, phase, action]``; the card tokens are only the
  VISIBLE cards, so ``N`` (and the axis order) changes every ply.
- **v2** -- ``S = 24`` FIXED: ``[CLS, 18 card instances, king_mine, king_theirs, board, phase, action]``.
  The token set IS the deck, so a heatmap row means the same card all game, unseen cards are attendable
  (``card_seen`` False), and each card carries a ``zone_posterior`` (where the model believes it is).

In both, ``seq_labels`` mirrors the sequence and the last three tokens are board/phase/action. At ``L=1``
only row 0 (CLS) causally feeds ``q``; at ``L>=2`` layer-1's card rows feed layer-2's CLS (``per_layer``
carries every layer so the renderer can route accordingly).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..cards import CARD_NAMES, asset_path, card_ids_for_name
from .features import CAND_COL, tokenize

_CTX_LABELS = ["board", "phase", "action"]                          # the 3 singleton tokens (no card art)
_KING_LABELS = ["king:mine", "king:theirs"]                         # v2 only, between cards and context
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
    row0_signed: Optional[np.ndarray] = None # [heads, S] signed per-head contribution of each token to the
    #                                          q-logit via the readout-layer CLS row; None unless requested
    attribution: Optional[np.ndarray] = None # [S] head-summed signed contribution (Δq-logit per token)
    feat: str = "v1"                         # featurization behind this payload ("v1" | "v2")
    # --- v2 only (None at v1): the belief block -- ghost the unseen cards, show WHERE they might be -----
    zone_posterior: Optional[np.ndarray] = None   # [18, 12] P(zone) per card instance; each row sums to 1
    zone_names: Optional[List[str]] = None        # len 12; the zone_posterior column names
    card_seen: Optional[List[bool]] = None        # len 18; True = located exactly (the posterior is a delta)
    card_seq_range: Optional[Tuple[int, int]] = None   # [lo, hi) seq indices holding the card tokens


def _parse_name(label: str) -> Optional[str]:
    """``"zone:CardName"`` (possibly with a trailing ``"*"`` for a synthetic candidate) -> CardName if it
    is a real card, else None (CLS / board / phase / action / ``opp_unknown:?``)."""
    if ":" not in label:
        return None
    name = label.split(":", 1)[1]
    if name.endswith("*"):
        name = name[:-1]
    return name if name in _NAME_TO_ASSET else None


def _parse_name2(label: str) -> Optional[str]:
    """v2 instance label (``"Soldier#1"`` / ``"Judge"``) -> CardName, else None (CLS / kings / context).
    v2 labels carry no zone prefix: location is the zone posterior, not part of the token's identity."""
    name = label.split("#", 1)[0]
    return name if name in _NAME_TO_ASSET else None


def explain(view, action, model, *, all_layers: bool = False,
            attribution: bool = False, ckpt_id: str = "") -> AttentionExplanation:
    """Run one forward pass for ``(view, action)`` and package the per-head CLS->token attention.

    ``view`` an InformationSet, ``action`` the candidate move to explain (or None), ``model`` an
    already-loaded attention net -- EITHER the torch :class:`AttentionModel` OR the numpy
    ``npz_infer.NumpyAttention``; both implement ``explain_forward``/``readout_u``, and everything crossing
    that seam is numpy, so this module needs no torch. (That is what lets the shipped game draw the
    attention drawer without bundling a 4.2 GB framework.) Its ``cfg.feat`` selects the v1/v2 featurization.

    With ``all_layers`` the payload carries every layer's attention (for L>=2 card-row routing); otherwise
    only the last (readout) layer. With ``attribution`` it also computes the signed value-weighted
    contribution of each token to the q-logit (``row0_signed`` / ``attribution``). Deterministic (eval)."""
    feat = getattr(model.cfg, "feat", "v1")
    if feat == "v2":
        from . import features2 as F2
        tok = F2.tokenize(view, action)
    else:
        tok = tokenize(view, action)

    q, attns, values, cards = model.explain_forward(tok, need_values=attribution)
    attn = attns[-1]                                                # [heads, S, S], the readout layer
    per_layer: Optional[List[np.ndarray]] = list(attns) if all_layers else None

    # Signed value-weighted attribution (readout layer): Δq_logit(j) = sum_h A^h[0][j] * (u^h . v_j^h),
    # u = head.weight @ W_o.weight (the readout direction). See attention_exploration / the plan for the math.
    row0_signed = attribution_vec = None
    if attribution:
        u = model.readout_u()                                       # [heads, dh]
        c = (u[:, None, :] * values[-1]).sum(-1)                    # [heads, S]  c[h,j] = u[h].v[h,j]
        rs = attn[:, 0, :] * c                                      # [heads, S]  signed contribution to logit
        row0_signed = rs.astype(np.float32)
        attribution_vec = rs.sum(0).astype(np.float32)              # [S] head-summed

    n_cards = len(tok.labels)
    zone_posterior = zone_names = card_seen = card_seq_range = None
    if feat == "v2":
        seq_labels = ["CLS"] + list(tok.labels) + list(_KING_LABELS) + list(_CTX_LABELS)
        display_names = [_parse_name2(lb) for lb in seq_labels]
        zone_posterior = tok.cards[:, F2._ZONE_OFF:].astype(np.float32).copy()   # [18, 12]
        zone_names = list(F2.ZONES)
        card_seen = [bool(p.max() >= 1.0 - 1e-6) for p in zone_posterior]        # delta == located
        card_seq_range = (1, 1 + n_cards)
        cand_ix = F2.CAND_COL
    else:
        seq_labels = ["CLS"] + list(tok.labels) + list(_CTX_LABELS)
        display_names = [_parse_name(lb) for lb in seq_labels]
        cand_ix = CAND_COL

    # Candidate token(s): the is_candidate_action column of the card block. Card-region index j -> seq
    # index 1 + j (cards are contiguous from 1 in both featurizations). v1 also catches the synthetic "*"
    # claim token; v2 has no synthetic token -- a guess flags EVERY instance of the named card instead.
    cand_col = cards[:, cand_ix]                                    # the batch's card block, already numpy
    cand = [1 + j for j in np.flatnonzero(cand_col == 1.0).tolist() if j < n_cards]
    if not cand and feat != "v2":                                   # fallback: any "*"-suffixed label
        cand = [i for i, lb in enumerate(seq_labels) if lb.endswith("*")]
    if feat == "v2":
        # Primary = the copy I actually own (its posterior is a delta on my_hand/my_ante); a play+guess or
        # a duplicate-name guess legitimately flags several instances.
        mine = (F2._ZONE_OFF + F2._Z["my_hand"], F2._ZONE_OFF + F2._Z["my_ante"])
        primary = next((i for i in cand if any(tok.cards[i - 1, c] >= 1.0 for c in mine)),
                       cand[0] if cand else None)
    else:                                                          # v1: prefer the my_hand play token
        primary = next((i for i in cand if seq_labels[i].startswith("my_hand:")),
                       cand[0] if cand else None)

    return AttentionExplanation(
        q=q, seq_labels=seq_labels, attn=attn, per_layer=per_layer,
        candidate_seq_index=primary, candidate_seq_indices=cand, display_names=display_names,
        name_to_asset=dict(_NAME_TO_ASSET), n_heads=int(attn.shape[0]), n_layers=model.cfg.n_layers,
        ckpt_id=ckpt_id, row0_signed=row0_signed, attribution=attribution_vec, feat=feat,
        zone_posterior=zone_posterior, zone_names=zone_names, card_seen=card_seen,
        card_seq_range=card_seq_range)
