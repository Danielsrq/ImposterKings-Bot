"""Pure-NumPy inference for the trained nets -- the shipped game runs WITHOUT torch.

torch is 4.2 GB installed; the attention net it serves is 108,737 parameters (435 KB). Bundling a deep
learning framework into a card game to multiply a matrix that fits in L2 cache is indefensible, so the
release reads weights from a plain ``.npz`` (written offline by ``export_npz.py``) and does the forward
pass here. numpy is already a hard dependency -- the engine deals cards with it -- so this costs nothing.

Both nets are supported and both mirror their torch twins EXACTLY (see ``tests/test_npz_parity.py``,
which pins q, the per-head attention AND the per-head values to 1e-5 against torch):

- **MLP**  -- Linear/ReLU stack, Tanh head. (``evaluator.build_evaluator`` already did this in numpy; it
  only needed torch to *unpickle* the checkpoint.)
- **Attention** -- per-type Linear+LayerNorm embed, learned CLS, L pre-norm blocks (multi-head
  self-attention + GELU FFN, both residual), Linear+Tanh readout off the CLS row. Dropout is identity at
  eval, so it is simply absent here.

The one numerical subtlety is GELU: torch's default is the EXACT erf form, and numpy ships no erf. See
:func:`_erf`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- primitives -----------------------------------------------------------------------------------

# GELU is the forward pass's hot spot -- it runs on [B, S, ffn_hidden] (~37k floats) twice per layer.
# torch's default nn.GELU is the EXACT erf form and numpy ships no erf, so we use Abramowitz & Stegun
# 7.1.26 (|err| <= 1.5e-7 on erf, measured 4.8e-7 on the composed GELU -- two orders inside the 1e-5
# parity gate). Written with in-place ops and a Horner fold because the naive expression allocated ~15
# temporaries of that size and cost 1.11 ms; this costs 0.28 ms for identical accuracy.
_A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)
_P = 0.3275911
_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _gelu(x: np.ndarray) -> np.ndarray:
    """GELU(x) = 0.5x(1 + erf(x/sqrt2)), matching torch's nn.GELU() default (approximate='none')."""
    u = np.abs(x) * _INV_SQRT2                       # the erf argument, folded to the positive branch
    t = np.reciprocal(1.0 + _P * u)
    poly = _A[4]
    for a in (_A[3], _A[2], _A[1], _A[0]):           # Horner: 4 fused steps, no temporaries per term
        poly = poly * t + a
    poly *= t
    np.multiply(u, u, out=u)                         # exp(-u^2) in place, reusing u's buffer
    np.negative(u, out=u)
    np.exp(u, out=u)
    e = 1.0 - poly * u                               # erf(|x|/sqrt2)
    np.copysign(e, x, out=e)                         # erf is odd -> restore the sign of x
    e += 1.0
    e *= x
    e *= 0.5
    return e


def _layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """torch.nn.LayerNorm over the last axis (population variance, as torch uses)."""
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)                       # biased -- matches torch
    return (x - mu) / np.sqrt(var + eps) * w + b


def _linear(x: np.ndarray, wt: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``wt`` is the PRE-TRANSPOSED, contiguous [in, out] weight (see NumpyAttention._prep). torch stores
    Linear as [out, in]; transposing on every call yields a non-contiguous view and a slower matmul."""
    return x @ wt + b


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


# --- config ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class NpConfig:
    """Mirrors AttnConfig's fields that inference actually reads (so `model.cfg.feat` etc. still work)."""
    d_model: int = 64
    n_layers: int = 1
    n_heads: int = 4
    ffn_hidden: int = 128
    bounded: bool = True
    feat: str = "v1"


class NumpyMLP:
    """The MLP q-net: Linear/ReLU stack with a Tanh head. ``q(view, action)`` batched over actions."""

    model_type = "mlp"

    def __init__(self, weights: Dict[str, np.ndarray], meta: Dict):
        n = len({k.split(".")[0] for k in weights if k.endswith(".weight")})
        self.layers = [(np.ascontiguousarray(weights[f"l{i}.weight"].T), weights[f"l{i}.bias"])
                       for i in range(n)]                # pre-transposed [in, out], like NumpyAttention
        self.meta = meta

    def q(self, x: np.ndarray) -> np.ndarray:
        for i, (w, b) in enumerate(self.layers):
            x = _linear(x, w, b)
            x = np.maximum(x, 0.0) if i < len(self.layers) - 1 else np.tanh(x)
        return x[:, 0]


class NumpyAttention:
    """The attention q-net. Duck-types the parts of :class:`AttentionModel` that the UI touches --
    ``cfg``, ``eval()``, and :meth:`explain_forward` -- so ``explain.py`` is backend-agnostic."""

    model_type = "attention"

    def __init__(self, weights: Dict[str, np.ndarray], meta: Dict, cfg: NpConfig):
        self.cfg = cfg
        self.meta = meta
        self.w = weights
        self._prep()

    def _prep(self) -> None:
        """One-time weight prep: pre-transpose every Linear to contiguous [in, out], and FUSE each layer's
        wq/wk/wv into a single [d, 3d] matrix. Three [B,S,d]x[d,d] matmuls become one [B,S,d]x[d,3d] --
        same arithmetic, a third of the numpy dispatch overhead, which is what actually costs at this size."""
        self.wt = {k[:-len(".weight")]: np.ascontiguousarray(v.T)
                   for k, v in self.w.items() if k.endswith(".weight") and v.ndim == 2}
        self.qkv = []
        for li in range(self.cfg.n_layers):
            p = f"layers.{li}.attn."
            self.qkv.append((
                np.ascontiguousarray(np.concatenate(
                    [self.wt[p + "wq"], self.wt[p + "wk"], self.wt[p + "wv"]], axis=1)),   # [d, 3d]
                np.concatenate([self.w[p + "wq.bias"], self.w[p + "wk.bias"], self.w[p + "wv.bias"]]),
            ))

    def eval(self):                                       # torch-API parity; nothing to switch off
        return self

    # -- embed ---------------------------------------------------------------------------------
    def _proj(self, kind: str, x: np.ndarray) -> np.ndarray:
        """Linear(native -> d) + LayerNorm, per token kind (embed.<kind>.0 / .1 in the state dict)."""
        h = _linear(x, self.wt[f"embed.{kind}.0"], self.w[f"embed.{kind}.0.bias"])
        return _layer_norm(h, self.w[f"embed.{kind}.1.weight"], self.w[f"embed.{kind}.1.bias"])

    def _attend(self, x: np.ndarray, li: int, add_mask: Optional[np.ndarray]):
        """One multi-head self-attention. Returns (out [B,S,d], attn [B,h,S,S], v [B,h,S,dh])."""
        b, s, d = x.shape
        h, dh = self.cfg.n_heads, d // self.cfg.n_heads
        p = f"layers.{li}.attn."

        w_qkv, b_qkv = self.qkv[li]
        qkv = (x @ w_qkv + b_qkv).reshape(b, s, 3, h, dh)            # one matmul for all three
        q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))   # each [B,h,S,dh]
        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh)           # [B,h,S,S]
        if add_mask is not None:
            scores = scores + add_mask
        attn = _softmax(scores, -1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(b, s, d)
        return _linear(out, self.wt[p + "wo"], self.w[p + "wo.bias"]), attn, v

    def _encode(self, cards, board, phase, action, kings=None, card_mask=None):
        """[CLS, cards, (kings), board, phase, action] -> per-layer attention + the final CLS q."""
        b = cards.shape[0]
        hc = self._proj("card", cards)
        hb, hp, ha = self._proj("board", board), self._proj("phase", phase), self._proj("action", action)
        cls = np.broadcast_to(self.w["cls"], (b, 1, self.cfg.d_model))
        parts = [cls, hc]
        if self.cfg.feat == "v2":
            parts.append(self._proj("king", kings))                  # [B,2,d]
        parts += [hb[:, None, :], hp[:, None, :], ha[:, None, :]]
        seq = np.concatenate(parts, axis=1).astype(np.float32)       # [B,S,d]

        add_mask = None
        if card_mask is not None:                                    # v1: pad -> -inf on invalid keys
            n_ctx = 5 if self.cfg.feat == "v2" else 3
            ones = np.ones((b, 1), dtype=bool)
            valid = np.concatenate([ones, card_mask.astype(bool),
                                    np.ones((b, n_ctx), dtype=bool)], axis=1)      # [B,S]
            add_mask = np.where(valid[:, None, None, :], 0.0, -np.inf).astype(np.float32)

        attns, values = [], []
        for li in range(self.cfg.n_layers):
            a, attn, v = self._attend(_layer_norm(seq, self.w[f"layers.{li}.ln1.weight"],
                                                  self.w[f"layers.{li}.ln1.bias"]), li, add_mask)
            seq = seq + a
            f = _layer_norm(seq, self.w[f"layers.{li}.ln2.weight"], self.w[f"layers.{li}.ln2.bias"])
            f = _linear(f, self.wt[f"layers.{li}.ffn.0"], self.w[f"layers.{li}.ffn.0.bias"])
            f = _linear(_gelu(f), self.wt[f"layers.{li}.ffn.3"], self.w[f"layers.{li}.ffn.3.bias"])
            seq = seq + f
            attns.append(attn)
            values.append(v)
        q = _linear(seq[:, 0, :], self.wt["head"], self.w["head.bias"])[:, 0]
        if self.cfg.bounded:
            q = np.tanh(q)
        return q, attns, values

    def readout_u(self) -> np.ndarray:
        """The readout direction u [heads, dh]: head.weight @ W_o of the LAST layer. A token's signed
        contribution to the q-logit is A[h,0,j] * (u[h] . v[h,j]) -- see explain()."""
        wo = self.w[f"layers.{self.cfg.n_layers - 1}.attn.wo.weight"]
        return (self.w["head.weight"][0] @ wo).reshape(self.cfg.n_heads, -1)

    def explain_forward(self, tok, need_values: bool = False):
        """The single entry point ``explain()`` uses (torch's AttentionModel implements it too).
        Returns ``(q: float, attns: [np [h,S,S]], values: [np [h,S,dh]] | None, cards: np [N, C])``."""
        b = batch_of(tok, self.cfg.feat)
        q, attns, values = self._encode(**b)
        return (float(q[0]), [a[0] for a in attns],
                [v[0] for v in values] if need_values else None, b["cards"][0])

    def q_batch(self, toks: List) -> np.ndarray:
        """q for a LIST of (view, action) token sets -- ONE batched forward (the MCTS leaf path, where
        every legal move is scored at once)."""
        return self._encode(**_collate([batch_of(t, self.cfg.feat) for t in toks]))[0]


def batch_of(tok, feat: str) -> Dict[str, np.ndarray]:
    """One tokenized (view, action) -> the [1, ...] numpy batch the encoder wants. The numpy twin of
    attention_model.collate/collate2 -- but torch-free, so the shipped app never imports torch."""
    out = {"cards": tok.cards[None].astype(np.float32),
           "board": tok.board[None].astype(np.float32),
           "phase": tok.phase[None].astype(np.float32),
           "action": tok.action[None].astype(np.float32)}
    if feat == "v2":
        out["kings"] = tok.kings[None].astype(np.float32)            # [1,2,4]; fixed S=24, no mask
    else:
        out["card_mask"] = np.ones((1, tok.cards.shape[0]), dtype=bool)
    return out


def _collate(batches: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Stack same-shaped single-row batches (v2's token count is fixed, so no padding is needed)."""
    return {k: np.concatenate([b[k] for b in batches], axis=0) for k in batches[0]}


# --- the MCTS leaf evaluator ----------------------------------------------------------------------

def evaluator_from(model):
    """``state -> ([per-seat value], {move: prior})`` for ``mcts.SearchConfig.evaluator`` -- the torch-free
    twin of ``evaluator.build_evaluator`` / ``attention_model.evaluator_from_model``. Value = max_a q
    (mover-relative, negated for the opponent: leaves are zero-sum), priors = softmax(q) over legal moves."""
    from ..rules import NUM_PLAYERS

    is_attn = isinstance(model, NumpyAttention)
    if is_attn and model.cfg.feat == "v2":
        from . import features2 as F
    elif is_attn:
        from . import features as F
    else:
        from . import features as F

    def evaluate(state):
        mover = state.to_play
        view = state.information_set(mover)
        moves = state.legal_moves()
        if is_attn:
            q = model.q_batch([F.tokenize(view, m) for m in moves])
        else:
            q = model.q(np.stack([F.encode(view, m) for m in moves]).astype(np.float32))
        v = float(q.max())
        value = [0.0] * NUM_PLAYERS
        value[mover] = v
        value[1 - mover] = -v                          # zero-sum leaf value, like state.result()
        e = np.exp(q - q.max())
        return value, dict(zip(moves, (e / e.sum()).tolist()))

    return evaluate


def build_evaluator(checkpoint: str):
    """Load an ``.npz`` and return its MCTS leaf evaluator. No torch, at any point."""
    return evaluator_from(load(checkpoint))


# --- loading --------------------------------------------------------------------------------------

def load(path: str):
    """Load a ``.npz`` written by ``export_npz.py`` -> NumpyAttention | NumpyMLP. No torch."""
    z = np.load(path, allow_pickle=False)
    kind = str(z["__model_type__"])
    weights = {k: z[k].astype(np.float32) for k in z.files if not k.startswith("__")}
    meta = {k[2:-2]: z[k].item() if z[k].shape == () else z[k]
            for k in z.files if k.startswith("__") and k != "__model_type__"}
    if kind == "attention":
        cfg = NpConfig(d_model=int(z["__d_model__"]), n_layers=int(z["__n_layers__"]),
                       n_heads=int(z["__n_heads__"]), ffn_hidden=int(z["__ffn_hidden__"]),
                       bounded=bool(z["__bounded__"]), feat=str(z["__feat__"]))
        return NumpyAttention(weights, meta, cfg)
    return NumpyMLP(weights, meta)
