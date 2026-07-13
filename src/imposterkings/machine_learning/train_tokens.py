"""Train the attention model over the token corpus (state,action -> q). PyTorch.

    python -m imposterkings.machine_learning.train_tokens --npz datasets/tensors/k20l3_tokens.npz

Same target/loss/split as the MLP trainer (``train.py``): MCTS ``mean_q``, visit-share-weighted MSE,
split by ``game_id`` -- but over the variable-length token sets (``features.tokenize``), collated per
batch. Reports regression + ranking metrics and the **total training wall-time**.

GPU: ``--device auto`` (default) trains on CUDA when available (~3x+ per epoch; the DataLoader/collate is
the remaining CPU-side bottleneck). Checkpoints are always saved as CPU tensors, so the evaluator/UI/
explain consumers never see the training device. The venv pins the CUDA build (RTX 3060 Ti, driver-
compatible): ``pip install torch==2.12.1+cu126 --index-url https://download.pytorch.org/whl/cu126``.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .attention_model import AttentionModel, AttnConfig, collate
from .attention_model import save as save_attention
from .features import Tokens
from .token_dataset import TokenRows, load as load_tokens
from .train import game_split, ranking_metrics


class TokenTorchDataset(Dataset):
    """Yields one token row: (cards[Nᵢ,44], board[14], phase[53], action[23], y, w)."""

    def __init__(self, rows: TokenRows, idx: np.ndarray):
        self.rows, self.idx = rows, idx

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, k):
        i = int(self.idx[k])
        return (*self.rows.tokens(i), float(self.rows.y[i]), float(self.rows.w[i]))


def collate_fn(batch):
    """Batch rows into model tensors + stack (y, w). Dispatches by tuple arity: v1 (padded, reuses
    attention_model.collate) vs v2 (fixed shapes, collate2 -- cards[18,46] + kings[2,4], no mask)."""
    if len(batch[0]) == 7:                             # v2: (cards, kings, board, phase, action, y, w)
        from .attention_model import collate2
        from .features2 import Tokens2
        packed = collate2([Tokens2(c, k, b, p, a, []) for (c, k, b, p, a, _, _) in batch])
    else:                                              # v1: (cards, board, phase, action, y, w)
        packed = collate([Tokens(c, b, p, a, []) for (c, b, p, a, _, _) in batch])
    y = torch.tensor([r[-2] for r in batch], dtype=torch.float32)
    w = torch.tensor([r[-1] for r in batch], dtype=torch.float32)
    return packed, y, w


def resolve_device(name: str = "auto") -> "torch.device":
    """"auto" -> cuda when available, else cpu. Training-only: checkpoints are saved as CPU tensors, so
    every downstream consumer (evaluator, UI, explain) is unaffected by the training device."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def _to_device(packed, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in packed.items()}


def _forward(model, packed):
    q, _ = model(packed["cards"], packed["board"], packed["phase"], packed["action"],
                 packed["card_mask"], kings=packed.get("kings"))
    return q


@torch.no_grad()
def _predict(model, rows, idx, batch, device) -> np.ndarray:
    model.eval()
    dl = DataLoader(TokenTorchDataset(rows, idx), batch_size=batch, shuffle=False, collate_fn=collate_fn)
    preds = [_forward(model, _to_device(packed, device)).cpu().numpy() for packed, _, _ in dl]
    return np.concatenate(preds) if preds else np.zeros(0, np.float32)


def lr_lambda(step: int, total: int, warmup: int) -> float:
    """Linear warmup -> cosine decay to ~0, as a multiplier on the base LR (LambdaLR, stepped PER STEP).

    Warmup: Adam's second-moment estimate is built from almost no data in the first steps, so its adaptive
    step sizes are unreliable and can be huge -- ramping in avoids blowing up early.
    Cosine: the LR must reach ~0 by the end or the optimizer never STOPS MOVING. With our old constant
    1e-3 the val curve bounced +-0.008 forever (it was still taking full-size steps at the last epoch)."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)                 # 0 -> 1 over the remaining steps
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


def run(rows: TokenRows, cfg: AttnConfig, hp: Dict) -> Dict:
    torch.manual_seed(hp["seed"])
    device = resolve_device(hp.get("device", "auto"))
    tr, va = game_split(rows.game_id, hp["val_frac"], hp["seed"])
    tri, vai = np.where(tr)[0], np.where(va)[0]

    model = AttentionModel(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    pin = device.type == "cuda"
    dl = DataLoader(TokenTorchDataset(rows, tri), batch_size=hp["batch"], shuffle=True,
                    collate_fn=collate_fn, pin_memory=pin)

    sched_kind = hp.get("schedule", "cosine")
    total_steps = hp["epochs"] * max(1, len(dl))
    warmup = int(hp.get("warmup_frac", 0.05) * total_steps)
    sched = (torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total_steps, warmup))
             if sched_kind == "cosine" else None)
    # Cosine and early stopping do not mix: stopping mid-schedule parks the model at a HIGH lr -- the worst
    # of both. With cosine we run the full budget and keep the FINAL model (lr -> 0 means it has settled).
    # That also kills the old max-of-noise bug: reporting min(val_mse) over a jagged curve cherry-picked a
    # lucky epoch (it inflated v2.2's headline from ~-10% to -18%).
    early_stop = sched_kind != "cosine"

    ysub = tri[:20000]                                     # fixed subsample -> a cheap per-epoch train_mse
    yva = rows.y[vai].astype(np.float32)
    ytr_sub = rows.y[ysub].astype(np.float32)
    hist: Dict[str, list] = {"step": [], "lr": [], "train_loss": [],
                             "epoch": [], "epoch_train_mse": [], "epoch_val_mse": [], "epoch_secs": []}
    t0 = time.perf_counter()
    best_val, best_state, bad, epochs_run, epoch_secs, step = float("inf"), None, 0, 0, [], 0
    for ep in range(hp["epochs"]):
        te = time.perf_counter()
        model.train()
        for packed, yb, wb in dl:
            packed, yb, wb = _to_device(packed, device), yb.to(device), wb.to(device)
            opt.zero_grad()
            pred = _forward(model, packed)
            loss = (wb * (pred - yb) ** 2).sum() / wb.sum().clamp_min(1e-8)
            loss.backward(); opt.step()
            if sched is not None:
                sched.step()
            hist["step"].append(step)                      # loss-vs-STEP, persisted into the checkpoint
            hist["lr"].append(float(opt.param_groups[0]["lr"]))
            hist["train_loss"].append(float(loss.detach()))
            step += 1
        vmse = float(np.mean((_predict(model, rows, vai, hp["batch"], device) - yva) ** 2))
        tmse = float(np.mean((_predict(model, rows, ysub, hp["batch"], device) - ytr_sub) ** 2))
        epochs_run, esec = ep + 1, time.perf_counter() - te
        epoch_secs.append(esec)
        hist["epoch"].append(ep + 1)
        hist["epoch_train_mse"].append(tmse)
        hist["epoch_val_mse"].append(vmse)
        hist["epoch_secs"].append(esec)
        print(f"  epoch {ep + 1:>2}: train_mse {tmse:.4f}  val_mse {vmse:.4f}  "
              f"lr {opt.param_groups[0]['lr']:.2e}  ({esec:.1f}s)", flush=True)
        if vmse < best_val - 1e-6:
            best_val, best_state, bad = vmse, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if early_stop and bad >= hp["patience"]:
                break
    final_val = hist["epoch_val_mse"][-1] if hist["epoch_val_mse"] else float("nan")
    if early_stop and best_state is not None:
        model.load_state_dict(best_state)                  # constant-LR path: keep the old behaviour
    seconds = time.perf_counter() - t0

    pv, pt = _predict(model, rows, vai, hp["batch"], device), _predict(model, rows, tri, hp["batch"], device)
    model.cpu()                                   # checkpoints + downstream consumers stay CPU-native
    yt = rows.y[tri].astype(np.float32)
    rank = ranking_metrics(pv, yva, rows.decision_id[vai], rows.is_chosen[vai])
    n = len(tri) + len(vai)
    tail = hist["epoch_val_mse"][-3:] or [float("nan")]
    return {"model": model, "params": model.param_count(),
            "train_mse": float(np.mean((pt - yt) ** 2)),
            # the SHIPPED model's val_mse: the final (settled) one under cosine, the best one under constant
            "val_mse": float(np.mean((pv - yva) ** 2)),
            "final_val_mse": final_val, "best_val_mse": best_val, "tail3_val_mse": float(np.mean(tail)),
            "val_mae": float(np.mean(np.abs(pv - yva))),
            "baseline_mse": float(np.mean((yva - yt.mean()) ** 2)),
            "schedule": sched_kind, "total_steps": step, "warmup_steps": warmup, "history": hist,
            "epochs": epochs_run, "seconds": seconds, "epoch_secs": epoch_secs, "rows": n,
            "rows_per_sec": (n * epochs_run / seconds) if seconds > 0 else 0.0, **rank}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train the attention q-model over the token corpus.")
    p.add_argument("--npz", default=os.path.join("datasets", "tensors", "k20l3_tokens.npz"))
    p.add_argument("--epochs", type=int, default=40,
                   help="FIXED budget under --schedule cosine (early stopping is disabled: stopping "
                        "mid-cosine would park the model at a high LR)")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3, help="PEAK lr (cosine decays it to ~0)")
    p.add_argument("--schedule", default="cosine", choices=["cosine", "constant"],
                   help="cosine = linear warmup then cosine decay to ~0 (the model can finally SETTLE); "
                        "constant reproduces the old runs (whose val curve bounced +-0.008 forever)")
    p.add_argument("--warmup-frac", type=float, default=0.05, help="fraction of steps spent warming up")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=5, help="early stopping; --schedule constant only")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--ffn-hidden", type=int, default=128)
    p.add_argument("--out-dir", default="models")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="training device (auto = cuda when available); checkpoints are always CPU-saved")
    args = p.parse_args(argv)

    rows = load_tokens(args.npz)
    cfg = AttnConfig(d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
                     ffn_hidden=args.ffn_hidden, dropout=args.dropout,
                     feat=rows.feat)                    # featurization follows the dataset (v1/v2)
    hp = {k: getattr(args, k) for k in ("epochs", "batch", "lr", "patience", "val_frac", "seed", "device",
                                        "schedule", "warmup_frac")}
    print(f"loaded {len(rows)} rows; cfg={cfg}; target=q; device={resolve_device(args.device)}", flush=True)

    r = run(rows, cfg, hp)
    print(f"\nparams {r['params']}  epochs {r['epochs']}  steps {r['total_steps']} "
          f"(warmup {r['warmup_steps']}, schedule {r['schedule']})  TIME {r['seconds']:.1f}s "
          f"({r['rows_per_sec']:.0f} rows/s, {np.mean(r['epoch_secs']):.1f}s/epoch)")
    print(f"train_mse {r['train_mse']:.4f}  val_mse {r['val_mse']:.4f}  baseline {r['baseline_mse']:.4f}  "
          f"gap {r['val_mse'] - r['train_mse']:+.4f}")
    print(f"  final {r['final_val_mse']:.4f} | best {r['best_val_mse']:.4f} | last-3 mean "
          f"{r['tail3_val_mse']:.4f}   (a big final-vs-best gap => the curve never settled)")
    print(f"top1_bestq {r['top1_bestq'] * 100:.1f}%  top1_chosen {r['top1_chosen'] * 100:.1f}%  "
          f"recall2 {r['recall2'] * 100:.1f}%  spearman {r['spearman']:.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"attn_d{cfg.d_model}_L{cfg.n_layers}.pt")
    keys = ("train_mse", "val_mse", "final_val_mse", "best_val_mse", "tail3_val_mse", "val_mae",
            "baseline_mse", "top1_bestq", "top1_chosen", "recall2", "spearman")
    save_attention(out, r["model"], meta={"target": "q", "npz": args.npz, "seconds": r["seconds"],
                   "config": vars(args), "metrics": {k: r[k] for k in keys},
                   "history": r["history"]})       # loss-vs-step travels WITH the checkpoint
    print(f"saved {out}   (history: {len(r['history']['step'])} steps, "
          f"{len(r['history']['epoch'])} epochs -- plot with plot_training.py)")


if __name__ == "__main__":
    main()
