"""Causal interventions on the attention q-net, and the EXACT residual decomposition of its logit.

The drawer's signed mode is direct logit attribution (DLA) restricted to the last layer's attention path.
This module is the toolkit for checking whether that explanation is faithful (Study 10,
`dla_faithfulness.py`):

* :func:`decompose` -- the full, exact split of the pre-tanh q-logit. There is no final LayerNorm
  (``q = tanh(head(seq[:, 0]))``), so the CLS residual is a plain sum and the logit is linear in it::

      logit = head.b + w.cls + sum_L w.wo_L.b                              (const)
            + sum_{L,h,j} A^{L,h}[0,j] * (u_L^h . v^{L,h}_j)               (attn, per layer/head/token)
            + sum_L w.FFN_L(LN2(x_L))[CLS]                                  (ffn, per layer)

  with ``u_L = head.w @ Wo_L``. The drawer shows only the last layer's ``attn`` slice. Layer 0's slice
  attributes to UNMIXED card embeddings (nothing has attended yet), so it has a clean per-card meaning.
* :func:`intervene` -- head ablation (mean/zero on the pre-``wo`` head columns), attention knockout (an
  additive -inf score bias on chosen query->key pairs, pre-softmax) and residual patching (replace slots
  entering a layer). All via forward pre-hooks, removed on exit: the model source is untouched, and
  ``explain`` / the numpy backend stay byte-identical.
* :func:`cf_assassin` -- tensor-level semantic counterfactuals on the Assassin's card token.
* :class:`AblatedModel` -- a picklable wrapper that duck-types for ``agent.NNPolicy`` (greedy play).

Everything is analysed in pre-tanh LOGIT space, read from a hook on ``model.head`` (not ``atanh(q)``,
which needs clipping at saturation). Argmax is invariant (tanh is monotone).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from . import features2 as F2

ASSASSIN_SLOT = F2.INSTANCE_NAMES.index("Assassin")             # card-token row (0-based)
ASSASSIN_SEQ = 1 + ASSASSIN_SLOT                                # sequence index (CLS is 0)
REACTION_COL = F2._MECH_OFF + 1                                 # mech:reaction
ZONE_SL = slice(F2._ZONE_OFF, F2._ZONE_OFF + F2._NZONE)
THEIR_HAND_COL = F2._ZONE_OFF + F2._Z["their_hand"]
DISCARD_COL = F2._ZONE_OFF + F2._Z["discard"]


def seq_labels() -> List[str]:
    """The v2 sequence axis: [CLS, 18 card instances, 2 kings, board, phase, action] (S = 24)."""
    return ["CLS"] + list(F2.INSTANCE_LABELS) + ["king:mine", "king:theirs", "board", "phase", "action"]


# --- intervention specs (frozen + picklable, so they cross joblib worker boundaries) ----------------

@dataclass(frozen=True)
class HeadAblation:
    """Replace head ``head`` of layer ``layer`` at the INPUT of ``wo`` (its dh columns of the concat).
    ``mode``: "mean" (per-position mean over the TRAIN split -- the primary, on-distribution choice) or
    "zero" (off-distribution; a contrast only). ``positions`` None = every sequence position."""
    layer: int
    head: int
    mode: str = "mean"
    positions: Optional[Tuple[int, ...]] = None


@dataclass(frozen=True)
class AttnKnockout:
    """Pre-softmax -inf on query->key pairs of layer ``layer``: the keys get zero attention and the rest
    renormalise. ``head`` None = all heads; ``queries`` None = every query row (full information removal
    of the key at that layer); the default ``(0,)`` blocks only the CLS readout row."""
    layer: int
    keys: Tuple[int, ...]
    head: Optional[int] = None
    queries: Optional[Tuple[int, ...]] = (0,)


@dataclass(frozen=True)
class ResidualPatch:
    """Overwrite ``seq[:, positions]`` entering ``layers[layer]`` with donor values (given at call time)."""
    layer: int
    positions: Tuple[int, ...]


@dataclass(frozen=True)
class InterventionSpec:
    ablations: Tuple[HeadAblation, ...] = ()
    knockouts: Tuple[AttnKnockout, ...] = ()

    @property
    def empty(self) -> bool:
        return not self.ablations and not self.knockouts

    def label(self) -> str:
        parts = [f"abl:L{a.layer}h{a.head}:{a.mode}" for a in self.ablations]
        parts += [f"ko:L{k.layer}h{'*' if k.head is None else k.head}:k{list(k.keys)}"
                  f"{'' if k.queries == (0,) else ':allq' if k.queries is None else ':q' + str(list(k.queries))}"
                  for k in self.knockouts]
        return "+".join(parts) or "none"


def _dh(model) -> int:
    return model.cfg.d_model // model.cfg.n_heads


def _knockout_bias(model, kos: Sequence[AttnKnockout], s: int, device) -> torch.Tensor:
    """[1, H, S, S] additive score bias for one layer's knockouts. Refuses to knock out a whole row."""
    h = model.cfg.n_heads
    bias = torch.zeros(1, h, s, s, device=device)
    for ko in kos:
        hs = range(h) if ko.head is None else [ko.head]
        qs = range(s) if ko.queries is None else ko.queries
        for hh in hs:
            for q in qs:
                bias[0, hh, q, list(ko.keys)] = float("-inf")
    if torch.isinf(bias).all(dim=-1).any():
        raise ValueError("knockout would remove every key of an attention row (softmax undefined)")
    return bias


@contextmanager
def intervene(model, spec: InterventionSpec = InterventionSpec(), *, means: Optional[torch.Tensor] = None,
              patch: Optional[Tuple[ResidualPatch, torch.Tensor]] = None) -> Iterator[None]:
    """Apply ``spec`` (and an optional residual ``patch``) for the duration of the block.

    ``means`` [L, S, d]: per-layer, per-position mean of the pre-``wo`` concat (:func:`head_means`),
    required by mean-mode ablations. ``patch`` = (ResidualPatch, donor [B, len(positions), d] or
    broadcastable). Every hook is removed in ``finally``, even on error."""
    handles = []
    dh = _dh(model)
    try:
        by_layer_abl: Dict[int, List[HeadAblation]] = {}
        for a in spec.ablations:
            if a.mode not in ("mean", "zero"):
                raise ValueError(f"unknown ablation mode {a.mode!r}")
            if a.mode == "mean" and means is None:
                raise ValueError("mean ablation needs `means` (see head_means)")
            by_layer_abl.setdefault(a.layer, []).append(a)
        for li, abls in by_layer_abl.items():
            def wo_pre(mod, args, abls=abls, li=li):
                out = args[0].clone()                                   # [B,S,d] concat, h-major
                for a in abls:
                    cols = slice(a.head * dh, (a.head + 1) * dh)
                    pos = slice(None) if a.positions is None else list(a.positions)
                    if a.mode == "zero":
                        out[:, pos, cols] = 0.0
                    else:
                        out[:, pos, cols] = means[li][pos, cols].to(out.dtype)
                return (out,) + tuple(args[1:])
            handles.append(model.layers[li].attn.wo.register_forward_pre_hook(wo_pre))

        by_layer_ko: Dict[int, List[AttnKnockout]] = {}
        for k in spec.knockouts:
            by_layer_ko.setdefault(k.layer, []).append(k)
        for li, kos in by_layer_ko.items():
            def attn_pre(mod, args, kwargs, kos=kos):
                x = args[0]
                bias = _knockout_bias(model, kos, x.shape[1], x.device)
                mask = args[1] if len(args) > 1 else kwargs.get("add_mask")
                new_mask = bias if mask is None else mask + bias
                if len(args) > 1:
                    return (x, new_mask) + tuple(args[2:]), kwargs
                return (x,), {**kwargs, "add_mask": new_mask}
            handles.append(model.layers[li].attn.register_forward_pre_hook(attn_pre, with_kwargs=True))

        if patch is not None:
            rp, donor = patch

            def layer_pre(mod, args):
                seq = args[0].clone()
                seq[:, list(rp.positions)] = donor.to(seq.dtype)
                return (seq,) + tuple(args[1:])
            handles.append(model.layers[rp.layer].register_forward_pre_hook(layer_pre))
        yield
    finally:
        for hd in handles:
            hd.remove()


@contextmanager
def record(model) -> Iterator[dict]:
    """Capture, per layer: ``resid_in`` [B,S,d], ``concat`` (pre-wo, post-ablation) [B,S,d], ``attn``
    probs [B,H,S,S] (post-knockout), ``ln1`` (attention input) [B,S,d], ``ffn_out`` [B,S,d]; and ``logit``
    [B] (pre-tanh head output). Read-only hooks; removed on exit."""
    rec: dict = {"resid_in": {}, "concat": {}, "attn": {}, "ln1": {}, "ffn_out": {}, "logit": None}
    handles = []
    try:
        for li, layer in enumerate(model.layers):
            handles.append(layer.register_forward_pre_hook(
                lambda m, a, li=li: rec["resid_in"].__setitem__(li, a[0].detach())))
            handles.append(layer.attn.register_forward_pre_hook(
                lambda m, a, li=li: rec["ln1"].__setitem__(li, a[0].detach())))
            # registered AFTER any intervene() hook on the same module -> sees the ablated concat
            handles.append(layer.attn.wo.register_forward_pre_hook(
                lambda m, a, li=li: rec["concat"].__setitem__(li, a[0].detach())))
            handles.append(layer.attn.register_forward_hook(
                lambda m, a, o, li=li: rec["attn"].__setitem__(li, o[1].detach())))
            handles.append(layer.ffn.register_forward_hook(
                lambda m, a, o, li=li: rec["ffn_out"].__setitem__(li, o.detach())))
        handles.append(model.head.register_forward_hook(
            lambda m, a, o: rec.__setitem__("logit", o.detach().squeeze(-1))))
        yield rec
    finally:
        for hd in handles:
            hd.remove()


# --- decomposition ------------------------------------------------------------------------------

@dataclass
class Decomp:
    """Exact split of the pre-tanh logit. ``attn[b,L,h,j]`` = A^{L,h}[0,j] * (u_L^h . v^{L,h}_j) with the
    UN-ablated values; ``attn_head[b,L,h]`` = the head's realised direct contribution (== attn.sum(-1)
    unless the head is ablated). ``resid`` = logit - (const + attn_head + ffn) and must be ~0."""
    logit: np.ndarray           # [B]
    q: np.ndarray               # [B]   model output (tanh(logit) when bounded)
    const: np.ndarray           # [B]
    attn: np.ndarray            # [B, L, H, S]
    attn_head: np.ndarray       # [B, L, H]
    ffn: np.ndarray             # [B, L]
    resid: np.ndarray           # [B]

    @property
    def components(self) -> Dict[str, np.ndarray]:
        """[B] per named component: const, attn_L{l}, ffn_L{l}."""
        out = {"const": self.const}
        for li in range(self.attn_head.shape[1]):
            out[f"attn_L{li}"] = self.attn_head[:, li].sum(-1)
            out[f"ffn_L{li}"] = self.ffn[:, li]
        return out


def batch_from_rows(rows, idx: np.ndarray, cards: Optional[np.ndarray] = None) -> dict:
    """v2 TokenRows -> model tensors for rows ``idx`` (optionally with substituted ``cards``)."""
    c = rows.cards[idx] if cards is None else cards
    return {"cards": torch.from_numpy(np.ascontiguousarray(c, dtype=np.float32)),
            "kings": torch.from_numpy(np.ascontiguousarray(rows.kings[idx], dtype=np.float32)),
            "board": torch.from_numpy(np.ascontiguousarray(rows.board[idx], dtype=np.float32)),
            "phase": torch.from_numpy(np.ascontiguousarray(rows.phase[idx], dtype=np.float32)),
            "action": torch.from_numpy(np.ascontiguousarray(rows.action[idx], dtype=np.float32))}


def _run(model, batch: dict):
    return model.forward_layers(batch["cards"], batch["board"], batch["phase"], batch["action"],
                                batch.get("card_mask"), kings=batch["kings"])


@torch.no_grad()
def decompose(model, batch: dict, spec: InterventionSpec = InterventionSpec(),
              means: Optional[torch.Tensor] = None,
              patch: Optional[Tuple[ResidualPatch, torch.Tensor]] = None) -> Decomp:
    """One forward (under ``spec``/``patch``) -> the exact decomposition of every row's logit."""
    model.eval()
    L, H, dh = model.cfg.n_layers, model.cfg.n_heads, _dh(model)
    with intervene(model, spec, means=means, patch=patch), record(model) as rec:
        q, _ = _run(model, batch)
    w = model.head.weight[0]                                                       # [d]
    const = float(model.head.bias[0]) + float(w @ model.cls)
    const = const + sum(float(w @ model.layers[li].attn.wo.bias) for li in range(L))
    B, S = q.shape[0], rec["attn"][0].shape[-1]
    attn = torch.zeros(B, L, H, S)
    attn_head = torch.zeros(B, L, H)
    ffn = torch.zeros(B, L)
    for li in range(L):
        mha = model.layers[li].attn
        u = (w @ mha.wo.weight).view(H, dh)                                        # [H, dh]
        v = mha.wv(rec["ln1"][li]).view(B, S, H, dh).transpose(1, 2)               # [B,H,S,dh] un-ablated
        c = (v * u[None, :, None, :]).sum(-1)                                      # [B,H,S]  u_h . v_hj
        attn[:, li] = rec["attn"][li][:, :, 0, :] * c                              # A[0,j] * c_j
        cat0 = rec["concat"][li][:, 0, :].view(B, H, dh)                           # realised (post-ablation)
        attn_head[:, li] = (cat0 * u[None]).sum(-1)
        ffn[:, li] = rec["ffn_out"][li][:, 0, :] @ w
    logit = rec["logit"]
    resid = logit - (const + attn_head.sum((1, 2)) + ffn.sum(1))
    return Decomp(logit=logit.numpy().astype(np.float64), q=q.numpy().astype(np.float64),
                  const=np.full(B, const), attn=attn.numpy().astype(np.float64),
                  attn_head=attn_head.numpy().astype(np.float64), ffn=ffn.numpy().astype(np.float64),
                  resid=resid.numpy().astype(np.float64))


@torch.no_grad()
def logits(model, batch: dict, spec: InterventionSpec = InterventionSpec(),
           means: Optional[torch.Tensor] = None,
           patch: Optional[Tuple[ResidualPatch, torch.Tensor]] = None) -> np.ndarray:
    """Pre-tanh logits [B] under an intervention (cheaper than :func:`decompose`)."""
    model.eval()
    got = {}
    hd = model.head.register_forward_hook(lambda m, a, o: got.__setitem__("z", o.detach().squeeze(-1)))
    try:
        with intervene(model, spec, means=means, patch=patch):
            _run(model, batch)
    finally:
        hd.remove()
    return got["z"].numpy().astype(np.float64)


@torch.no_grad()
def head_means(model, rows, idx: np.ndarray, batch: int = 2048) -> torch.Tensor:
    """[L, S, d] mean of the pre-``wo`` concat per layer and position over rows ``idx`` (use TRAIN rows:
    the ablation baseline must not be fitted on the evaluation split). v2 positions are fixed, so a
    per-position mean is the natural on-distribution replacement."""
    model.eval()
    tot, n = None, 0
    for s in range(0, len(idx), batch):
        b = batch_from_rows(rows, idx[s:s + batch])
        with record(model) as rec:
            _run(model, b)
        cur = torch.stack([rec["concat"][li].sum(0) for li in range(model.cfg.n_layers)])   # [L,S,d]
        tot = cur if tot is None else tot + cur
        n += b["cards"].shape[0]
    return tot / max(1, n)


# --- semantic counterfactuals on the Assassin token --------------------------------------------

def assassin_unseen_in_their_hand(cards: np.ndarray) -> np.ndarray:
    """[B] bool: the Assassin is unseen (posterior not a delta) and has mass on ``their_hand``."""
    post = cards[:, ASSASSIN_SLOT, ZONE_SL]
    return (post.max(-1) < 1.0 - 1e-6) & (cards[:, ASSASSIN_SLOT, THEIR_HAND_COL] > 1e-6)


def cf_assassin(cards: np.ndarray, kind: str) -> Tuple[np.ndarray, np.ndarray]:
    """Counterfactual copy of ``cards`` [B,18,46] and an ``applicable`` [B] mask (unseen + in-hand mass).

    * ``lacks`` -- their_hand -> 0, renormalise the posterior. Identical to tokenizing
      ``replace(view, opp_hand_lacks=... | {"Assassin"})`` (the lacks path touches only this row).
    * ``dead``  -- posterior := delta on discard (the card is known gone; only the zone block changes).
    * ``cold``  -- the reaction mechanic bit := 0 (the card can no longer react; location unchanged).
    Rows where it is not applicable are returned unchanged."""
    out = np.array(cards, dtype=np.float32, copy=True)
    ok = assassin_unseen_in_their_hand(cards)
    r = out[ok, ASSASSIN_SLOT]
    if kind == "lacks":
        post = r[:, ZONE_SL].copy()
        post[:, F2._Z["their_hand"]] = 0.0
        post /= post.sum(-1, keepdims=True)
        r[:, ZONE_SL] = post
    elif kind == "dead":
        r[:, ZONE_SL] = 0.0
        r[:, DISCARD_COL] = 1.0
    elif kind == "cold":
        r[:, REACTION_COL] = 0.0
    else:
        raise ValueError(f"unknown counterfactual {kind!r}")
    out[ok, ASSASSIN_SLOT] = r
    return out, ok


# --- greedy-play wrapper ------------------------------------------------------------------------

class AblatedModel(nn.Module):
    """Duck-types for ``agent.NNPolicy`` (``.cfg``, ``.eval()``, ``forward(...) -> (q, attn)``): every
    forward runs under ``spec``. Holds no live hooks, so it pickles."""

    def __init__(self, model: nn.Module, spec: InterventionSpec, means: Optional[torch.Tensor] = None):
        super().__init__()
        self.model, self.spec, self.cfg = model, spec, model.cfg
        self.register_buffer("means", means if means is not None else torch.zeros(0), persistent=False)

    def forward(self, *args, **kwargs):
        m = self.means if self.means.numel() else None
        with intervene(self.model, self.spec, means=m):
            return self.model(*args, **kwargs)
