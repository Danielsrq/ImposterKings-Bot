"""Activation / saturation audit of the attention q-net -- Karpathy's two init cautions, made checkable.

    python -m imposterkings.machine_learning.inspect_activations \
        --ckpt models/gen1_v3c/attn_d64_L2.pt --npz datasets/tensors/gen1_v3c_tokens.npz

Prints the numbers and writes PNGs (``--out``, default ``reports/activations``):

  1. ``hist_q.png``        -- plt.hist(h, 100) of the post-tanh output q  (does mass pile up at +-1?)
  2. ``saturation.png``    -- the imshow(|h| > 0.99) view, per Karpathy
  3. ``ffn_gelu.png``      -- the SAME two views for the FFN hidden layer, which is where our net actually
                              HAS a wide hidden activation (see the mapping note below)
  4. ``init_vs_trained.png`` -- q at init vs after training vs the label distribution

**Mapping to the makemore lecture.** Karpathy inspects ``h = tanh(x @ W1 + b1)``: a WIDE HIDDEN layer,
so ``h`` is ``[batch, n_hidden]`` and ``imshow(h.abs() > 0.99)`` is a picture -- a fully-white COLUMN is a
dead neuron (always saturated => never gets gradient). Our net is not shaped like that:

  * our ``tanh`` is on the **output**, so ``q`` is ``[batch, 1]`` -- one column. The imshow is degenerate,
    which is why we ALSO reshape the batch into a square grid: a white pixel = a saturated ROW (example),
    not a dead neuron. The histogram is the informative view here.
  * our wide hidden layer uses **GELU** (in the FFN), which cannot saturate at +-1; its failure mode is
    output ~ 0 (the "dead" regime). So for the FFN we image ``|a| < 0.01`` instead of ``|a| > 0.99``, and a
    white column there IS a dead neuron in exactly Karpathy's sense.

The 3 nonlinearities in the model: **GELU** (FFN), **softmax** (attention), **tanh** (value head).
"""
from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .attention_model import AttentionModel, AttnConfig
from .attention_model import load as load_attn
from .token_dataset import load as load_rows
from .train import game_split
from .train_tokens import TokenTorchDataset, collate_fn


def _forward_all(model, rows, idx, batch=512):
    """Run the model, collecting q, the pre-tanh logit, the FFN GELU activations and the attention."""
    acts: Dict[str, List[np.ndarray]] = {}

    def grab(name):
        def hook(_m, _i, out):
            t = out if isinstance(out, torch.Tensor) else out[0]
            acts.setdefault(name, []).append(t.detach().float().cpu().numpy())
        return hook

    hooks = []
    for li, layer in enumerate(model.layers):
        hooks.append(layer.ffn[0].register_forward_hook(grab(f"L{li}_pre")))    # INPUT to GELU
        hooks.append(layer.ffn[1].register_forward_hook(grab(f"L{li}_gelu")))   # OUTPUT of GELU

    dl = DataLoader(TokenTorchDataset(rows, idx), batch_size=batch, shuffle=False, collate_fn=collate_fn)
    qs, ent, mx = [], [], []
    model.eval()
    with torch.no_grad():
        for packed, _, _ in dl:
            q, A = model(packed["cards"], packed["board"], packed["phase"], packed["action"],
                         packed["card_mask"], kings=packed.get("kings"))
            qs.append(q.cpu().numpy())
            a = A.cpu().numpy()                                   # [B, heads, S, S]
            s = a.shape[-1]
            e = -(a * np.log(a + 1e-12)).sum(-1)                  # entropy per query row
            ent.append((e / math.log(s)).mean(axis=(0, 2)))       # as a FRACTION of uniform, per head
            mx.append(a.max(-1).mean(axis=(0, 2)))
    for h in hooks:
        h.remove()
    q = np.concatenate(qs)
    flat = {k: np.concatenate([v.reshape(-1, v.shape[-1]) for v in vs]) for k, vs in acts.items()}
    return q, flat, np.mean(ent, axis=0), np.mean(mx, axis=0)


def _square(v: np.ndarray) -> np.ndarray:
    """Reshape a 1-D per-row signal into a square image so it can be imshow'd (a pixel = one example)."""
    n = int(math.isqrt(len(v)))
    return v[: n * n].reshape(n, n)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Activation/saturation audit of the attention q-net.")
    p.add_argument("--ckpt", default=os.path.join("models", "gen1_v3c", "attn_d64_L2.pt"))
    p.add_argument("--npz", default=os.path.join("datasets", "tensors", "gen1_v3c_tokens.npz"))
    p.add_argument("--rows", type=int, default=20000, help="validation rows to audit")
    p.add_argument("--out", default=os.path.join("reports", "activations"))
    args = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(2)

    rows = load_rows(args.npz)
    tr, va = game_split(rows.game_id, 0.1, 0)
    vai = np.where(va)[0][: args.rows]
    y = rows.y[vai].astype(np.float64)
    y_all = rows.y.astype(np.float64)
    base = float(np.mean((y - rows.y[np.where(tr)[0]].mean()) ** 2))

    model, _ = load_attn(args.ckpt)
    cfg = model.cfg
    torch.manual_seed(0)
    fresh = AttentionModel(AttnConfig(**{**vars(cfg)}))            # same arch, UNTRAINED

    q, acts, ent, mx = _forward_all(model, rows, vai)
    q0, acts0, ent0, _ = _forward_all(fresh, rows, vai)
    z = np.arctanh(np.clip(q, -0.999999, 0.999999))                # pre-tanh logit (q = tanh(z))
    grad = 1.0 - q ** 2                                            # d tanh/dz -- the gradient multiplier

    # ---------------- numbers ----------------
    print("=" * 76)
    print(f"LABELS   n={len(y_all):,}  mean {y_all.mean():+.4f}  std {y_all.std():.4f}")
    for t in (0.9, 0.99):
        print(f"  |y| > {t:<4}: {float((np.abs(y_all) > t).mean()) * 100:5.1f}%  "
              f"(to emit this, tanh needs |z| > {np.arctanh(t):.2f}; gradient there = {1 - t ** 2:.3f})")
    print(f"  |y| == 1.0 EXACTLY: {float((np.abs(y_all) >= 1.0).mean()) * 100:.2f}%  "
          f"-> tanh can never reach it; gradient -> 0 on those rows")
    print()
    print("=" * 76)
    print("1) LOSS AT INIT   (Karpathy: init should NOT be confidently wrong)")
    print("=" * 76)
    print(f"  predict-the-mean baseline : {base:.4f}")
    print(f"  UNTRAINED model           : {float(np.mean((q0 - y) ** 2)):.4f}   "
          f"<- want ~= baseline")
    print(f"  trained model             : {float(np.mean((q - y) ** 2)):.4f}")
    print(f"  untrained q: mean {q0.mean():+.4f} std {q0.std():.4f}   (label mean {y_all.mean():+.4f})")
    print()
    print("=" * 76)
    print("2) TANH at the value head")
    print("=" * 76)
    print(f"  pre-tanh z : mean {z.mean():+.3f} std {z.std():.3f} range [{z.min():+.2f}, {z.max():+.2f}]")
    print(f"  tanh'=1-q^2: mean {grad.mean():.4f} median {np.median(grad):.4f}")
    print(f"  |q| > 0.99 : {float((np.abs(q) > 0.99).mean()) * 100:.2f}% of rows  "
          f"(SATURATED -- Karpathy's flat region)")
    print(f"  tanh' < 0.1: {float((grad < 0.1).mean()) * 100:.2f}% of rows (>=10x gradient shrink)")
    print()
    print("=" * 76)
    print("3) GELU in the FFN   (a white COLUMN in ffn_gelu.png = a dead neuron)")
    print("=" * 76)
    for k in sorted(a for a in acts if a.endswith("_gelu")):
        a, pre = acts[k], acts[k.replace("_gelu", "_pre")]
        dead_frac = (np.abs(a) < 0.01).mean(axis=0)                # per NEURON
        print(f"  {k:<8} out~0: {float((np.abs(a) < 0.01).mean()) * 100:4.1f}% of activations | "
              f"pre-GELU mean {pre.mean():+.3f} std {pre.std():.3f} | "
              f"neurons dead >90% of the time: {int((dead_frac > 0.9).sum())}/{a.shape[1]}")
    print()
    print("=" * 76)
    print("4) SOFTMAX in attention   (entropy as a FRACTION of uniform: 1.0 = flat, 0.0 = one-hot)")
    print("=" * 76)
    for h in range(len(ent)):
        tag = "COLLAPSED (no gradient to other keys)" if ent[h] < 0.25 else (
            "peaky/selective" if ent[h] < 0.6 else "diffuse (weakly selective)")
        print(f"  head {h}: entropy {ent[h] * 100:5.1f}% of uniform | mean max-weight {mx[h]:.3f}  -> {tag}")
    print(f"  (untrained, for reference: {np.round(ent0 * 100, 1)}% -- an untrained head is ~100% = uniform)")

    # ---------------- plots ----------------
    # (1) plt.hist(h, 100) of the post-tanh output -- THE view for "are we saturating?"
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(q, bins=100, color="#3b528b")
    ax[0].set_title(f"post-tanh q  (|q|>0.99: {float((np.abs(q) > 0.99).mean()) * 100:.2f}%)")
    ax[0].set_xlabel("q = tanh(z)"); ax[0].set_xlim(-1, 1)
    ax[1].hist(z, bins=100, color="#21918c")
    for s, c in ((2.65, "r"), (-2.65, "r")):
        ax[1].axvline(s, color=c, ls="--", lw=1)
    ax[1].set_title("pre-tanh logit z   (red = |z|=2.65, where tanh'<0.02)")
    ax[1].set_xlabel("z")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "hist_q.png"), dpi=110); plt.close(fig)

    # (2) imshow(|h| > 0.99) -- a pixel is one EXAMPLE (our tanh is an output, not a hidden layer)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].imshow(_square((np.abs(q) > 0.99).astype(float)), cmap="gray", interpolation="nearest")
    ax[0].set_title("|q| > 0.99  (white = saturated example)")
    ax[1].imshow(_square(grad), cmap="viridis", interpolation="nearest", vmin=0, vmax=1)
    ax[1].set_title("tanh' = 1-q^2  (dark = gradient dies)")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "saturation.png"), dpi=110); plt.close(fig)

    # (3) the FFN hidden layer -- this is the true analogue of Karpathy's [batch, n_hidden] picture
    key = sorted(a for a in acts if a.endswith("_gelu"))[0]
    A = acts[key][:600]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].imshow(np.abs(A) < 0.01, cmap="gray", interpolation="nearest", aspect="auto")
    ax[0].set_title(f"{key}: |a| < 0.01  (white = 'dead'; a white COLUMN = a dead neuron)")
    ax[0].set_xlabel("neuron"); ax[0].set_ylabel("example")
    ax[1].hist(acts[key].reshape(-1), bins=100, color="#5ec962")
    ax[1].set_title(f"{key} activations (GELU cannot saturate at +-1; its dead zone is 0)")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "ffn_gelu.png"), dpi=110); plt.close(fig)

    # (4) init vs trained vs labels
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(y, bins=100, alpha=0.5, label="labels (MCTS mean_q)", color="#440154", density=True)
    ax.hist(q0, bins=100, alpha=0.6, label=f"q at INIT (mean {q0.mean():+.2f})", color="#fde725", density=True)
    ax.hist(q, bins=100, alpha=0.6, label="q TRAINED", color="#21918c", density=True)
    ax.legend(); ax.set_xlim(-1, 1); ax.set_xlabel("q")
    ax.set_title("init should sit at the LABEL MEAN, not off to one side")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "init_vs_trained.png"), dpi=110); plt.close(fig)

    print(f"\nwrote 4 figures -> {args.out}/")


if __name__ == "__main__":
    main()
