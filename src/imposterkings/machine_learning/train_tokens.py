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
        c, b, p, a = self.rows.tokens(i)
        return c, b, p, a, float(self.rows.y[i]), float(self.rows.w[i])


def collate_fn(batch):
    """Pad a list of rows into model tensors (reuses attention_model.collate) + stack (y, w)."""
    packed = collate([Tokens(c, b, p, a, []) for (c, b, p, a, _, _) in batch])
    y = torch.tensor([r[4] for r in batch], dtype=torch.float32)
    w = torch.tensor([r[5] for r in batch], dtype=torch.float32)
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
    q, _ = model(packed["cards"], packed["board"], packed["phase"], packed["action"], packed["card_mask"])
    return q


@torch.no_grad()
def _predict(model, rows, idx, batch, device) -> np.ndarray:
    model.eval()
    dl = DataLoader(TokenTorchDataset(rows, idx), batch_size=batch, shuffle=False, collate_fn=collate_fn)
    preds = [_forward(model, _to_device(packed, device)).cpu().numpy() for packed, _, _ in dl]
    return np.concatenate(preds) if preds else np.zeros(0, np.float32)


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

    yva = rows.y[vai].astype(np.float32)
    t0 = time.perf_counter()
    best_val, best_state, bad, epochs_run, epoch_secs = float("inf"), None, 0, 0, []
    for ep in range(hp["epochs"]):
        te = time.perf_counter()
        model.train()
        for packed, yb, wb in dl:
            packed, yb, wb = _to_device(packed, device), yb.to(device), wb.to(device)
            opt.zero_grad()
            pred = _forward(model, packed)
            loss = (wb * (pred - yb) ** 2).sum() / wb.sum().clamp_min(1e-8)
            loss.backward(); opt.step()
        vmse = float(np.mean((_predict(model, rows, vai, hp["batch"], device) - yva) ** 2))
        epochs_run, esec = ep + 1, time.perf_counter() - te
        epoch_secs.append(esec)
        print(f"  epoch {ep + 1:>2}: val_mse {vmse:.4f}  ({esec:.1f}s)", flush=True)
        if vmse < best_val - 1e-6:
            best_val, best_state, bad = vmse, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    seconds = time.perf_counter() - t0

    pv, pt = _predict(model, rows, vai, hp["batch"], device), _predict(model, rows, tri, hp["batch"], device)
    model.cpu()                                   # checkpoints + downstream consumers stay CPU-native
    yt = rows.y[tri].astype(np.float32)
    rank = ranking_metrics(pv, yva, rows.decision_id[vai], rows.is_chosen[vai])
    n = len(tri) + len(vai)
    return {"model": model, "params": model.param_count(),
            "train_mse": float(np.mean((pt - yt) ** 2)), "val_mse": best_val,
            "val_mae": float(np.mean(np.abs(pv - yva))),
            "baseline_mse": float(np.mean((yva - yt.mean()) ** 2)),
            "epochs": epochs_run, "seconds": seconds, "epoch_secs": epoch_secs, "rows": n,
            "rows_per_sec": (n * epochs_run / seconds) if seconds > 0 else 0.0, **rank}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train the attention q-model over the token corpus.")
    p.add_argument("--npz", default=os.path.join("datasets", "tensors", "k20l3_tokens.npz"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=5)
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
                     ffn_hidden=args.ffn_hidden, dropout=args.dropout)
    hp = {k: getattr(args, k) for k in ("epochs", "batch", "lr", "patience", "val_frac", "seed", "device")}
    print(f"loaded {len(rows)} rows; cfg={cfg}; target=q; device={resolve_device(args.device)}", flush=True)

    r = run(rows, cfg, hp)
    print(f"\nparams {r['params']}  epochs {r['epochs']}  TIME {r['seconds']:.1f}s "
          f"({r['rows_per_sec']:.0f} rows/s, {np.mean(r['epoch_secs']):.1f}s/epoch)")
    print(f"train_mse {r['train_mse']:.4f}  val_mse {r['val_mse']:.4f}  baseline {r['baseline_mse']:.4f}  "
          f"gap {r['val_mse'] - r['train_mse']:+.4f}")
    print(f"top1_bestq {r['top1_bestq'] * 100:.1f}%  top1_chosen {r['top1_chosen'] * 100:.1f}%  "
          f"recall2 {r['recall2'] * 100:.1f}%  spearman {r['spearman']:.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"attn_d{cfg.d_model}_L{cfg.n_layers}.pt")
    keys = ("train_mse", "val_mse", "val_mae", "top1_bestq", "top1_chosen", "recall2", "spearman")
    save_attention(out, r["model"], meta={"target": "q", "npz": args.npz, "seconds": r["seconds"],
                   "config": vars(args), "metrics": {k: r[k] for k in keys}})
    print(f"saved {out}")


if __name__ == "__main__":
    main()
