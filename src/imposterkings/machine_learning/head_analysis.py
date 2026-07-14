"""What is each attention head actually FOR? Causal entropy + the tokens each head reads.

    python -m imposterkings.machine_learning.head_analysis \\
        --ckpt models/v3c_gen2_d96/attn_d96_L2.pt models/v3c_gen2/attn_d64_L2.pt \\
        --npz datasets/tensors/v3c_gen2_tokens_v2.npz

**Why not just average the attention map.** ``inspect_activations`` reports entropy averaged over every
query row, and that number is a lie by omission: at L=2 the final layer's CARD rows are computed and then
DISCARDED (only the CLS row is read out by the q-head), and they dominate the average. Averaged, every head
looks "diffuse" at 87-92% of uniform. Measured causally -- over only the rows whose output survives into the
answer -- the same heads are far sharper, down to 26%. The drawer already routes this way (phase 7); this is
the tool that measures it.

The causal rows, for an L=2 net:
  * **layer 0**: the 18 CARD rows. Their outputs are what layer 1's CLS then reads, so they carry gradient.
    (Layer 0's own CLS row is read too, but the card rows are where the position is actually encoded.)
  * **layer 1 (final)**: the CLS row ALONE. Every other row is computed and thrown away.

Entropy is reported as a fraction of uniform: 1.0 = attends to everything equally (an untrained head sits at
~99%), 0.0 = attends to exactly one token. Lower = more specialized.
"""
from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import cards
from .attention_model import load as load_attn
from .token_dataset import load as load_rows
from .train import game_split
from .train_tokens import TokenTorchDataset, collate_fn


def seq_labels(feat: str, n_cards: int = 18) -> List[str]:
    """The S=24 v2 sequence: [CLS, 18 card instances, 2 kings, board, phase, action]."""
    if feat == "v2":
        return (["CLS"] + [f"{cards.card_name(i)}#{i}" for i in range(n_cards)]
                + ["king_mine", "king_theirs", "board", "phase", "ACTION"])
    return ["CLS"] + [f"card{i}" for i in range(n_cards)] + ["board", "phase", "ACTION"]


def analyse(path: str, rows, idx: np.ndarray, batch: int = 512) -> Dict:
    """Per-head causal entropy + the mean attention mass each head puts on each token."""
    model, _ = load_attn(path)
    model.eval()
    cfg = model.cfg
    dl = DataLoader(TokenTorchDataset(rows, idx), batch_size=batch, shuffle=False, collate_fn=collate_fn)

    ent_sum: Optional[List[np.ndarray]] = None      # per layer, per head
    mass_sum: Optional[np.ndarray] = None           # layer-0 card rows -> [H, S]: what each head READS
    n_batches = 0
    with torch.no_grad():
        for packed, _, _ in dl:
            _, attns = model.forward_layers(packed["cards"], packed["board"], packed["phase"],
                                            packed["action"], packed["card_mask"], kings=packed.get("kings"))
            per_layer = []
            for li, a in enumerate(attns):
                A = a.numpy()                        # [B, H, S, S]
                S = A.shape[-1]
                last = li == len(attns) - 1
                # THE causal selection: the final layer contributes only through its CLS row; every earlier
                # layer contributes through the card rows that the next layer's CLS will read.
                sub = A[:, :, 0:1, :] if last else A[:, :, 1:, :]
                e = -(sub * np.log(sub + 1e-12)).sum(-1) / math.log(S)     # entropy / log(S) -> [B,H,rows]
                per_layer.append(e.mean(axis=(0, 2)))                      # -> [H]
                if li == 0:
                    m = A[:, :, 1:, :].mean(axis=(0, 2))                   # [H, S]
                    mass_sum = m if mass_sum is None else mass_sum + m
            ent_sum = per_layer if ent_sum is None else [x + y for x, y in zip(ent_sum, per_layer)]
            n_batches += 1

    return {"path": path, "cfg": cfg, "params": model.param_count(),
            "entropy": [e / n_batches for e in ent_sum],
            "mass": mass_sum / n_batches,
            "labels": seq_labels(cfg.feat)}


def report(r: Dict, top: int = 5) -> None:
    cfg = r["cfg"]
    name = os.path.basename(os.path.dirname(r["path"])) or os.path.basename(r["path"])
    print(f"\n{'=' * 92}")
    print(f"{name}   d{cfg.d_model}/h{cfg.n_heads}/L{cfg.n_layers}/ffn{cfg.ffn_hidden}   "
          f"head_dim {cfg.d_model // cfg.n_heads}   {r['params']:,} params")
    print("=" * 92)

    ent = r["entropy"]
    print("\nCAUSAL entropy, % of uniform (LOWER = more specialized; untrained ~99%)")
    for li, e in enumerate(ent):
        last = li == len(ent) - 1
        what = "READOUT row (CLS -> q; the ONLY row read out)" if last else \
               "card rows (feed the next layer's CLS -- these carry gradient)"
        print(f"  L{li} {what}")
        print("      " + "  ".join(f"h{h}:{v * 100:.0f}%" for h, v in enumerate(e)))
    sharp = ent[0]
    print(f"\n  layer-0 spread: {sharp.min() * 100:.0f}% (h{int(sharp.argmin())}, the sharpest) "
          f"-> {sharp.max() * 100:.0f}% (h{int(sharp.argmax())}, the broadest)   "
          f"range {(sharp.max() - sharp.min()) * 100:.0f}pp")
    print("  A WIDE range is the healthy sign: it means the heads DIVIDED the problem up. A narrow range at\n"
          "  a high value means every head is doing the same diffuse thing and the extra heads are wasted.")

    print(f"\nWHAT EACH HEAD READS (layer 0, mean attention mass over the card rows; top {top})")
    labels, mass = r["labels"], r["mass"]
    for h in np.argsort(sharp):                      # sharpest first -- the interesting ones
        order = np.argsort(-mass[h])[:top]
        bits = "  ".join(f"{labels[i]}:{mass[h][i] * 100:.0f}%" for i in order)
        print(f"  h{h} ({sharp[h] * 100:.0f}%)  {bits}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Causal per-head attention entropy + what each head attends to.")
    p.add_argument("--ckpt", nargs="+", required=True, help="one or more attention checkpoints to compare")
    p.add_argument("--npz", default=os.path.join("datasets", "tensors", "v3c_gen2_tokens_v2.npz"))
    p.add_argument("--rows", type=int, default=6000, help="validation rows to audit")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0, help="MUST match the training seed or the split differs")
    p.add_argument("--top", type=int, default=5)
    args = p.parse_args(argv)

    rows = load_rows(args.npz)
    _, va = game_split(rows.game_id, args.val_frac, args.seed)     # audit on HELD-OUT games only
    idx = np.where(va)[0][:args.rows]
    print(f"auditing {len(idx):,} held-out rows from {args.npz}")

    results = [analyse(c, rows, idx) for c in args.ckpt]
    for r in results:
        report(r, args.top)

    if len(results) > 1:
        print(f"\n{'=' * 92}\nSIDE BY SIDE (layer-0 causal entropy, sorted sharpest-first)\n{'=' * 92}")
        print(f"  {'net':>22} {'heads':>5} {'sharpest':>9} {'broadest':>9} {'range':>7}   per-head")
        for r in results:
            e = np.sort(r["entropy"][0])
            cfg = r["cfg"]
            tag = f"d{cfg.d_model}/h{cfg.n_heads}"
            print(f"  {tag:>22} {cfg.n_heads:>5} {e[0] * 100:>8.0f}% {e[-1] * 100:>8.0f}% "
                  f"{(e[-1] - e[0]) * 100:>6.0f}pp   " + " ".join(f"{v * 100:.0f}" for v in e))


if __name__ == "__main__":
    main()
