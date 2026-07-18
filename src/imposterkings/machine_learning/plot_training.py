"""Plot a checkpoint's training history -- loss vs STEP and vs epoch.

    python -m imposterkings.machine_learning.plot_training --ckpt models/gen1_v3c/attn_d64_L2.pt

``train_tokens`` records the per-step minibatch loss + LR and the per-epoch train/val MSE, and persists
them into the checkpoint (``meta["history"]``), so every model carries its own curve. Two reference lines
make the plot readable at a glance:

* **predict-the-mean baseline** (~0.20) -- the loss of a model that ignores its input. Training must start
  HERE, not above it (the old init scored 0.2662: confidently wrong; see ``inspect_activations``).
* **label-noise floor** (~0.0059, measured by re-searching the same positions with different RNG seeds) --
  the target is a stochastic MCTS ``mean_q``, so NO model can score below this. The gap between the floor
  and the val curve is the part that is actually the model's fault.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

# Measured on the gen-1 corpus (see attention_exploration.md): 200 positions x 6 independent searches.
LABEL_NOISE_FLOOR = 0.0059
MEAN_BASELINE = 0.2020


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Plot a checkpoint's training curves (loss vs step/epoch).")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default=None, help="PNG path (default: reports/training_<ckpt name>.png)")
    p.add_argument("--smooth", type=int, default=50, help="moving-average window for the per-step loss")
    args = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    h = (blob.get("meta") or {}).get("history")
    if not h:
        raise SystemExit(f"{args.ckpt} carries no training history "
                         f"(it predates the history tracking -- retrain to get curves)")

    name = os.path.splitext(os.path.basename(args.ckpt))[0]
    out = args.out or os.path.join("reports", f"training_{name}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    step = np.asarray(h["step"], float)
    loss = np.asarray(h["train_loss"], float)
    lr = np.asarray(h["lr"], float)
    ep = np.asarray(h["epoch"], float)
    tr = np.asarray(h["epoch_train_mse"], float)
    va = np.asarray(h["epoch_val_mse"], float)

    # A belief checkpoint (Study 7) stores CE in the same 'mse' history keys; only the reference lines and
    # labels differ -- the two baselines are the net's own input (hand-crafted CE, the line to beat) and the
    # naive uniform CE. Value checkpoints keep the MSE floor/baseline constants.
    meta = blob.get("meta") or {}
    is_belief = meta.get("target") == "belief"
    bl = meta.get("baselines") or {}
    if is_belief:
        ref_lo, ref_hi = bl.get("handcrafted_ce"), bl.get("uniform_ce")
        lo_lab, hi_lab = "hand-crafted CE (beat this)", "uniform CE"
        yl = "cross-entropy (log)"
        loss_lab = "unseen-CE"
    else:
        ref_lo, ref_hi = LABEL_NOISE_FLOOR, MEAN_BASELINE
        lo_lab, hi_lab = "label-noise floor", "predict-the-mean"
        yl = "weighted MSE (log)"
        loss_lab = "mse"

    # With update-to-data present: a 2-row figure -- top = loss + per-epoch, bottom = the UD ratio split
    # BY LEVEL (embed | L0 | L1 | head), like the per-layer GELU plot. Older checkpoints (no ud) keep 1x2.
    ud = h.get("ud") or []
    if ud:
        fig = plt.figure(figsize=(15, 9.2))
        gs = fig.add_gridspec(2, 4)
        ax = [fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2:])]
        ud_axes = []
        for j in range(4):
            ud_axes.append(fig.add_subplot(gs[1, j], sharey=ud_axes[0] if ud_axes else None))
    else:
        fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))

    # --- (a) per-STEP training loss, with the LR schedule on a twin axis ---------------------------
    k = max(1, min(args.smooth, len(loss)))               # clamp: a window past the run's length would
    sm = np.convolve(loss, np.ones(k) / k, mode="valid")  # make 'valid' return |k|-|loss|+1 stray points
    ax[0].plot(step, loss, lw=0.4, alpha=0.30, color="#3b528b", label="minibatch loss")
    ax[0].plot(step[k - 1:], sm, lw=1.8, color="#21918c", label=f"moving avg ({k})")
    if ref_hi is not None:
        ax[0].axhline(ref_hi, ls=":", color="grey", lw=1)
        ax[0].text(step[-1], ref_hi, f" {hi_lab}", color="grey", va="bottom", ha="right", fontsize=8)
    if ref_lo is not None:
        ax[0].axhline(ref_lo, ls="--", color="crimson", lw=1)
        ax[0].text(step[-1], ref_lo, f" {lo_lab}", color="crimson", va="bottom", ha="right", fontsize=8)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("optimizer step"); ax[0].set_ylabel(yl)
    ax[0].set_title("training loss vs step")
    ax[0].legend(fontsize=8, loc="upper right")
    ax2 = ax[0].twinx()
    ax2.plot(step, lr, color="#fde725", lw=1.4, ls="-")
    ax2.set_ylabel("learning rate", color="#b8a800")
    ax2.tick_params(axis="y", labelcolor="#b8a800")

    # --- (b) per-EPOCH train vs val (the overfit gap + whether it SETTLED) -------------------------
    ax[1].plot(ep, tr, "o-", color="#5ec962", lw=1.8, ms=3, label=f"train_{loss_lab}")
    ax[1].plot(ep, va, "o-", color="#440154", lw=1.8, ms=3, label=f"val_{loss_lab}")
    if ref_lo is not None:
        ax[1].axhline(ref_lo, ls="--", color="crimson", lw=1, label=f"{lo_lab} ({ref_lo:.4f})")
    if ref_hi is not None:
        ax[1].axhline(ref_hi, ls=":", color="grey", lw=1, label=f"{hi_lab} ({ref_hi:.4f})")
    tail = va[-5:]
    ax[1].set_title(f"per-epoch  |  final val {va[-1]:.4f}  best {va.min():.4f}  "
                    f"last-5 std {tail.std():.4f}")
    ax[1].set_xlabel("epoch"); ax[1].set_yscale("log")
    ax[1].legend(fontsize=8)

    # --- (c) update-to-data ratio, SPLIT BY LEVEL (embed | L0 | L1 | head) -------------------------
    # One overlaid panel answers "is SOME tensor misbehaving"; per-level answers "is THIS level healthy" --
    # a frozen embedding or a dead readout is invisible in the overlay. Karpathy: want ~ -3.
    if ud:
        us = np.asarray(h["ud_step"], float)
        names = sorted(ud[0])

        def level(n):                               # which sub-panel a tensor belongs to
            if n.startswith("embed"):     return 0
            if n.startswith("layers.0"):  return 1
            if n.startswith("layers.1"):  return 2
            return 3                                # head / anything else

        def short(n):                               # a compact role label for the legend
            p = n.replace(".weight", "").split(".")
            if p[0] == "layers":  return ".".join(p[2:])     # attn.wq / ffn.0
            if p[0] == "embed":   return p[1]                # card / board / phase / action / king
            return p[0]                                      # head

        titles = ["embed (inputs)", "L0", "L1", "head (readout)"]
        cmap = plt.get_cmap("tab10")
        for lv, a in enumerate(ud_axes):
            grp = [n for n in names if level(n) == lv]
            for i, n in enumerate(grp):
                a.plot(us, [u[n] for u in ud], lw=1.1, alpha=0.85, color=cmap(i % 10), label=short(n))
            a.axhline(-3.0, ls="--", color="crimson", lw=1.0)
            a.axhspan(-4.0, -2.0, color="grey", alpha=0.12)      # the healthy band, in every sub-panel
            a.set_title(titles[lv], fontsize=10)
            a.set_xlabel("step")
            if grp:
                a.legend(fontsize=6, loc="upper right")
            if lv == 0:
                a.set_ylabel("log10( std(Δw) / std(w) )")
            else:
                a.tick_params(labelleft=False)                  # shared y -> label only the leftmost
        ud_axes[0].figure.text(0.5, 0.47, "update-to-data by level  —  band = healthy [-4,-2]; target -3; "
                               "the tail falls because cosine drives lr → 0 (read the plateau)",
                               ha="center", fontsize=9, color="#444")

    fig.tight_layout()
    fig.savefig(out, dpi=115)
    print(f"steps {len(step)}  epochs {len(ep)}  lr {lr.max():.2e} -> {lr[-1]:.2e}")
    print(f"final val {va[-1]:.4f} | best {va.min():.4f} | last-5 std {tail.std():.4f} "
          f"(a big std => the curve never settled)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
