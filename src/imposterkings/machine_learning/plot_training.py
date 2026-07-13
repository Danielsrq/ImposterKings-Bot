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

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))

    # --- (a) per-STEP training loss, with the LR schedule on a twin axis ---------------------------
    k = max(1, args.smooth)
    sm = np.convolve(loss, np.ones(k) / k, mode="valid")
    ax[0].plot(step, loss, lw=0.4, alpha=0.30, color="#3b528b", label="minibatch loss")
    ax[0].plot(step[k - 1:], sm, lw=1.8, color="#21918c", label=f"moving avg ({k})")
    ax[0].axhline(MEAN_BASELINE, ls=":", color="grey", lw=1)
    ax[0].axhline(LABEL_NOISE_FLOOR, ls="--", color="crimson", lw=1)
    ax[0].text(step[-1], LABEL_NOISE_FLOOR, " label-noise floor", color="crimson", va="bottom", ha="right",
               fontsize=8)
    ax[0].text(step[-1], MEAN_BASELINE, " predict-the-mean", color="grey", va="bottom", ha="right",
               fontsize=8)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("optimizer step"); ax[0].set_ylabel("weighted MSE (log)")
    ax[0].set_title("training loss vs step")
    ax[0].legend(fontsize=8, loc="upper right")
    ax2 = ax[0].twinx()
    ax2.plot(step, lr, color="#fde725", lw=1.4, ls="-")
    ax2.set_ylabel("learning rate", color="#b8a800")
    ax2.tick_params(axis="y", labelcolor="#b8a800")

    # --- (b) per-EPOCH train vs val (the overfit gap + whether it SETTLED) -------------------------
    ax[1].plot(ep, tr, "o-", color="#5ec962", lw=1.8, ms=3, label="train_mse")
    ax[1].plot(ep, va, "o-", color="#440154", lw=1.8, ms=3, label="val_mse")
    ax[1].axhline(LABEL_NOISE_FLOOR, ls="--", color="crimson", lw=1, label="label-noise floor (0.0059)")
    ax[1].axhline(MEAN_BASELINE, ls=":", color="grey", lw=1, label="predict-the-mean (0.2020)")
    tail = va[-5:]
    ax[1].set_title(f"per-epoch  |  final val {va[-1]:.4f}  best {va.min():.4f}  "
                    f"last-5 std {tail.std():.4f}")
    ax[1].set_xlabel("epoch"); ax[1].set_yscale("log")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=115)
    print(f"steps {len(step)}  epochs {len(ep)}  lr {lr.max():.2e} -> {lr[-1]:.2e}")
    print(f"final val {va[-1]:.4f} | best {va.min():.4f} | last-5 std {tail.std():.4f} "
          f"(a big std => the curve never settled)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
