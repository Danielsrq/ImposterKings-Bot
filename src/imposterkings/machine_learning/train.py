"""Train MLP q-evaluators over an architecture sweep; report regression + ranking metrics (PyTorch).

    python -m imposterkings.machine_learning.train --npz datasets/tensors/k20l3.npz --sweep "16;32;64"
    python -m imposterkings.machine_learning.train --npz ... --sweep "16,16;32,32;64,64"   # multi-layer

Target is the MCTS action-value ``q`` (NOT ``z`` -- z is a state value identical across a decision's
candidates and can't rank actions). Loss = MSE weighted by ``visit_share`` (reliable q's dominate).
Split is by ``game_id`` (rows within a game are correlated). Metrics: val MSE/MAE vs a constant-mean
baseline, plus per-decision top-1 agreement (does argmax predicted value pick the played / the best-q move).
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .checkpoint import save as save_checkpoint
from .mlp import MLP


def load_npz(path: str) -> Dict[str, np.ndarray]:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def parse_archs(sweep: str) -> List[List[int]]:
    """'16;32;64' -> [[16],[32],[64]]; '16,16;32' -> [[16,16],[32]]; '' item -> [] (linear)."""
    out: List[List[int]] = []
    for spec in sweep.split(";"):
        spec = spec.strip()
        out.append([int(x) for x in spec.split(",") if x.strip()] if spec else [])
    return out


def game_split(game_id: np.ndarray, val_frac: float, seed: int):
    games = np.unique(game_id)
    np.random.default_rng(seed).shuffle(games)
    n_val = max(1, int(len(games) * val_frac))
    val_games = set(games[:n_val].tolist())
    val = np.fromiter((g in val_games for g in game_id), dtype=bool, count=len(game_id))
    return ~val, val


def ranking_metrics(pred, y, did, chosen) -> Dict[str, float]:
    """Per decision (grouped by decision_id): does argmax(pred) equal the best-q / the played candidate."""
    groups: Dict[int, List[int]] = defaultdict(list)
    for i, d in enumerate(did):
        groups[int(d)].append(i)
    n = top_q = top_ch = rec2 = sp_n = 0
    sp_sum = 0.0
    for idxs in groups.values():
        idxs = np.asarray(idxs)
        p, yy, ch = pred[idxs], y[idxs], chosen[idxs]
        amp = int(np.argmax(p))
        dbest = int(np.argmax(yy))
        top_q += int(amp == dbest)
        top_ch += int(ch[amp] == 1)
        rec2 += int(dbest in np.argsort(-p)[:2])     # true-best move is in the model's top-2 (trivial if n<=2)
        if len(idxs) > 1:
            rp, ry = np.argsort(np.argsort(p)), np.argsort(np.argsort(yy))
            if rp.std() > 0 and ry.std() > 0:
                sp_sum += float(np.corrcoef(rp, ry)[0, 1]); sp_n += 1
        n += 1
    return {"top1_bestq": top_q / n, "top1_chosen": top_ch / n, "recall2": rec2 / n,
            "spearman": sp_sum / sp_n if sp_n else float("nan"), "n_decisions": n}


def train_one(data: Dict[str, np.ndarray], tr, va, hidden: Sequence[int], hp: Dict) -> Dict:
    torch.manual_seed(hp["seed"])
    X = torch.from_numpy(data["X"]); y = torch.from_numpy(data["y"]); w = torch.from_numpy(data["w"])
    tri, vai = torch.from_numpy(np.where(tr)[0]), torch.from_numpy(np.where(va)[0])
    Xtr, ytr, wtr = X[tri], y[tri], w[tri]
    Xva, yva = X[vai], y[vai]

    model = MLP(X.shape[1], hidden, dropout=hp["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    dl = DataLoader(TensorDataset(Xtr, ytr, wtr), batch_size=hp["batch"], shuffle=True)

    t0 = time.perf_counter()
    best_val, best_state, bad, epochs_run = float("inf"), None, 0, 0
    for ep in range(hp["epochs"]):
        model.train()
        for xb, yb, wb in dl:
            opt.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = (wb * (pred - yb) ** 2).sum() / wb.sum().clamp_min(1e-8)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vmse = torch.mean((model(Xva).squeeze(-1) - yva) ** 2).item()
        epochs_run = ep + 1
        if vmse < best_val - 1e-6:
            best_val, best_state, bad = vmse, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(best_state)
    seconds = time.perf_counter() - t0

    with torch.no_grad():
        pv = model(Xva).squeeze(-1).numpy()
        pt = model(Xtr).squeeze(-1).numpy()
    yv, yt = yva.numpy(), ytr.numpy()
    train_mse = float(np.mean((pt - yt) ** 2))                      # low train + high val => overfit
    baseline = float(np.mean((yv - yt.mean()) ** 2))               # constant-mean predictor
    rank = ranking_metrics(pv, yv, data["decision_id"][vai.numpy()], data["is_chosen"][vai.numpy()])
    return {"arch": hidden, "params": model.param_count(), "train_mse": train_mse, "val_mse": best_val,
            "val_mae": float(np.mean(np.abs(pv - yv))), "baseline_mse": baseline,
            "epochs": epochs_run, "seconds": seconds, "model": model, **rank}


def _arch_str(a: Sequence[int]) -> str:
    return "-".join(map(str, a)) if a else "linear"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="MLP q-evaluator architecture sweep.")
    p.add_argument("--npz", default=os.path.join("datasets", "tensors", "k20l3.npz"))
    p.add_argument("--sweep", default="16;32;64", help="';'-separated archs, ','-separated layers")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="models")
    args = p.parse_args(argv)

    data = load_npz(args.npz)
    tr, va = game_split(data["game_id"], args.val_frac, args.seed)
    hp = {k: getattr(args, k) for k in ("epochs", "batch", "lr", "dropout", "patience", "seed")}
    print(f"loaded {data['X'].shape[0]} rows (D={data['X'].shape[1]}), "
          f"train {int(tr.sum())} / val {int(va.sum())} rows; target=q\n")

    os.makedirs(args.out_dir, exist_ok=True)
    hdr = f"  {'arch':>10} {'params':>7} {'ep':>3} {'secs':>5} {'trn_mse':>8} {'val_mse':>8} " \
          f"{'baseMSE':>8} {'gap':>6} {'top1_q':>7} {'rec@2':>6} {'spear':>6}"
    print(hdr)
    rows = []
    for arch in parse_archs(args.sweep):
        r = train_one(data, tr, va, arch, hp)
        print(f"  {_arch_str(arch):>10} {r['params']:>7} {r['epochs']:>3} {r['seconds']:>5.1f} "
              f"{r['train_mse']:>8.4f} {r['val_mse']:>8.4f} {r['baseline_mse']:>8.4f} "
              f"{r['val_mse']-r['train_mse']:>6.4f} {r['top1_bestq']*100:>6.1f}% "
              f"{r['recall2']*100:>5.1f}% {r['spearman']:>6.3f}")
        _METRICS = ("train_mse", "val_mse", "val_mae", "top1_bestq", "top1_chosen", "recall2", "spearman")
        save_checkpoint(os.path.join(args.out_dir, f"mlp_{_arch_str(arch)}.pt"), r.pop("model"),
                        meta={"target": "q", "npz": args.npz, "metrics": {k: r[k] for k in _METRICS}})
        rows.append({"arch": _arch_str(arch), **{k: r[k] for k in
                     ("params", "epochs", "seconds", "train_mse", "val_mse", "baseline_mse", "val_mae",
                      "top1_bestq", "top1_chosen", "recall2", "spearman")}})

    csv_path = os.path.join(args.out_dir, "sweep_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)
    print(f"\nsaved {len(rows)} models + {csv_path}")


if __name__ == "__main__":
    main()
