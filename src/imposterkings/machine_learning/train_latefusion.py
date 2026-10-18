"""Train the late-fusion Q-net (Study 11) on a v3 per-decision dataset. Resumable at epoch granularity.

    python -m imposterkings.machine_learning.train_latefusion --npz datasets/tensors/v3c_gen25_tokens_v3.npz \
        --arm LF-A --n-self 1 --n-cross 1 --out-dir models/study11

One sample is one DECISION (the state once + all its candidates). The loss is the d96 recipe's visit-weighted
MSE on the candidates' MCTS ``mean_q`` -- so val MSE is comparable with ``train_tokens`` -- plus
``beta`` x the listwise policy cross-entropy against the search's visit distribution ``w`` (which sums to 1
per decision). Everything else follows the d96 recipe: Adam, peak lr 1e-3, linear warmup then cosine to 0
over a fixed epoch budget, dropout 0.1, ``game_split(game_id, 0.1, seed 0)`` -- the same val games as d96.

The whole dataset lives on the training device (it is ~0.4 GB as float32), so there is no CPU collate
bottleneck. Each epoch atomically writes ``<out>/<arm>.last.pt`` (model, optimizer, scheduler, RNG states,
history); ``--resume`` continues from it. The final model is ``<out>/<arm>.pt`` (CPU tensors, model_type
"latefusion").
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict

import numpy as np
import torch

from .latefusion_model import (CAND_KEYS, LateFusionModel, LFConfig, n_params, save as save_lf)
from .token_dataset import load_v3
from .train import game_split, per_corpus_metrics, ranking_metrics
from .train_tokens import lr_lambda, resolve_device

_X_KEYS = ("cards", "card_depth", "hands", "kings", "decision", "src_ptr") + CAND_KEYS


class DeviceData:
    """The v3 arrays as device tensors + per-decision candidate counts (for per-batch trimming)."""

    def __init__(self, arrs: Dict[str, np.ndarray], device):
        self.t = {}
        for k in _X_KEYS:
            a = arrs[k]
            self.t[k] = torch.from_numpy(a.astype(np.int64) if a.dtype == np.int8 else a).to(device)
        self.t["y"] = torch.from_numpy(arrs["y"]).to(device)
        self.t["w"] = torch.from_numpy(arrs["w"]).to(device)
        self.n = torch.from_numpy((arrs["cand_kind"] >= 0).sum(1)).to(device)

    def batch(self, idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        k = int(self.n[idx].max())
        out = {}
        for key, v in self.t.items():
            out[key] = v[idx, :k] if key in CAND_KEYS or key in ("y", "w") else v[idx]
        return out


def losses(out, b, beta: float):
    valid = b["cand_kind"] >= 0
    w = b["w"] * valid
    mse = (w * (out["q"] - b["y"]) ** 2).sum() / w.sum().clamp_min(1e-8)
    logp = torch.log_softmax(out["logit"], dim=-1).masked_fill(~valid, 0.0)
    ce = -(w * logp).sum(1).mean()
    return mse + beta * ce, mse, ce


@torch.no_grad()
def predict(model, data: DeviceData, idx: np.ndarray, bs: int = 1024):
    """(q [D,15], logit [D,15]) as numpy, padding = 0 / -inf."""
    model.eval()
    K = data.t["cand_kind"].shape[1]
    q = np.zeros((len(idx), K), np.float32)
    lg = np.full((len(idx), K), -np.inf, np.float32)
    dev = data.n.device
    for s in range(0, len(idx), bs):
        ii = torch.as_tensor(idx[s:s + bs], device=dev)
        out = model(data.batch(ii))
        k = out["q"].shape[1]
        q[s:s + bs, :k] = out["q"].float().cpu().numpy()
        lg[s:s + bs, :k] = out["logit"].float().cpu().numpy()
    return q, lg


def flat_metrics(arrs, idx, q, logit) -> Dict:
    """d96-comparable metrics: rows = real candidates; val_mse is the UNWEIGHTED per-row mean (as
    ``train_tokens``), ranking metrics per decision; plus policy metrics."""
    valid = arrs["cand_kind"][idx] >= 0
    did = np.repeat(arrs["decision_id"][idx], valid.sum(1))
    chosen = np.zeros_like(valid, np.int8)
    chosen[np.arange(len(idx)), arrs["chosen"][idx]] = 1
    pred, y = q[valid], arrs["y"][idx][valid]
    gid = np.repeat(arrs["game_id"][idx], valid.sum(1))
    rank = ranking_metrics(pred, y, did, chosen[valid])
    w = arrs["w"][idx]
    pol_top = float(np.mean(logit.argmax(1) == w.argmax(1)))
    pol_chosen = float(np.mean(logit.argmax(1) == arrs["chosen"][idx]))
    lse = np.log(np.exp(logit - logit.max(1, keepdims=True)).sum(1)) + logit.max(1)
    ce = float(np.mean(-(w * np.where(valid, logit - lse[:, None], 0.0)).sum(1)))
    return {"val_mse": float(np.mean((pred - y) ** 2)), "val_mae": float(np.mean(np.abs(pred - y))),
            **rank, "policy_top1_visits": pol_top, "policy_top1_chosen": pol_chosen, "policy_ce": ce,
            "by_corpus": per_corpus_metrics(pred, y, gid, did, chosen[valid])}


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def run(arrs: Dict[str, np.ndarray], cfg: LFConfig, hp: Dict, last_path: str = None, resume: bool = False,
        max_minutes: float = None) -> Dict:
    device = resolve_device(hp.get("device", "auto"))
    torch.manual_seed(hp["seed"])
    tr, va = game_split(arrs["game_id"], hp["val_frac"], hp["seed"])
    tri, vai = np.where(tr)[0], np.where(va)[0]
    if hp.get("train_frac", 1.0) < 1.0:                    # learning curve: a random subset of TRAIN games
        g = np.unique(arrs["game_id"][tri])
        keep = np.random.default_rng(hp["seed"]).choice(g, int(round(len(g) * hp["train_frac"])), replace=False)
        tri = tri[np.isin(arrs["game_id"][tri], keep)]
    data = DeviceData(arrs, device)
    model = LateFusionModel(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    steps_per_epoch = int(np.ceil(len(tri) / hp["batch"]))
    total = hp["epochs"] * steps_per_epoch
    warm = int(hp["warmup_frac"] * total)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total, warm))
    hist: Dict[str, list] = {"epoch": [], "train_loss": [], "train_mse": [], "train_ce": [], "val_mse": [],
                             "val_top1_bestq": [], "epoch_secs": []}
    start_ep, seconds = 0, 0.0
    if resume and last_path and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        if ck["config"] != cfg.__dict__ or ck["hp"] != hp:
            raise ValueError(f"{last_path} was written with a different config/hp -- refuse to resume")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        torch.set_rng_state(ck["rng_cpu"])
        if ck.get("rng_cuda") is not None and device.type == "cuda":
            torch.cuda.set_rng_state(ck["rng_cuda"])
        hist, start_ep, seconds = ck["hist"], ck["epoch"], ck["seconds"]
        print(f"  resumed from {last_path} at epoch {start_ep}", flush=True)
    t_wall = time.perf_counter()
    for ep in range(start_ep, hp["epochs"]):
        te = time.perf_counter()
        model.train()
        perm = np.random.default_rng([hp["seed"], ep]).permutation(tri)     # resume-stable shuffling
        tl = tm = tc = 0.0
        for s in range(0, len(perm), hp["batch"]):
            b = data.batch(torch.as_tensor(perm[s:s + hp["batch"]], device=device))
            out = model(b)
            loss, mse, ce = losses(out, b, hp["beta"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl += float(loss.detach()); tm += float(mse.detach()); tc += float(ce.detach())
        nb = steps_per_epoch
        qv, lv = predict(model, data, vai)
        m = flat_metrics(arrs, vai, qv, lv)
        esec = time.perf_counter() - te
        seconds += esec
        for k, v in (("epoch", ep + 1), ("train_loss", tl / nb), ("train_mse", tm / nb), ("train_ce", tc / nb),
                     ("val_mse", m["val_mse"]), ("val_top1_bestq", m["top1_bestq"]), ("epoch_secs", esec)):
            hist[k].append(v)
        print(f"  epoch {ep + 1:>2}: loss {tl / nb:.4f} (mse {tm / nb:.4f}, ce {tc / nb:.3f})  val_mse "
              f"{m['val_mse']:.4f}  top1 {m['top1_bestq'] * 100:.1f}%  pol {m['policy_top1_visits'] * 100:.1f}%"
              f"  lr {opt.param_groups[0]['lr']:.2e} ({esec:.1f}s)", flush=True)
        if last_path:
            _atomic_save({"config": cfg.__dict__, "hp": hp, "model": model.state_dict(), "opt": opt.state_dict(),
                          "sched": sched.state_dict(), "rng_cpu": torch.get_rng_state(),
                          "rng_cuda": torch.cuda.get_rng_state() if device.type == "cuda" else None,
                          "hist": hist, "epoch": ep + 1, "seconds": seconds}, last_path)
        if max_minutes and (time.perf_counter() - t_wall) / 60 > max_minutes and ep + 1 < hp["epochs"]:
            print(f"  --max-minutes reached after epoch {ep + 1}; rerun with --resume to continue", flush=True)
            return {"model": None, "done": False, "epoch": ep + 1}
    qv, lv = predict(model, data, vai)
    m = flat_metrics(arrs, vai, qv, lv)
    sub = tri[np.random.default_rng(0).choice(len(tri), min(20000, len(tri)), replace=False)]
    qt, lt = predict(model, data, sub)                       # a RANDOM train subsample (not the first rows)
    mt = flat_metrics(arrs, sub, qt, lt)
    model.cpu()
    return {"model": model, "done": True, "params": n_params(model), "seconds": seconds, "history": hist,
            "train_mse": mt["val_mse"], **m}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train the late-fusion Q-net (Study 11).")
    p.add_argument("--npz", default=os.path.join("datasets", "tensors", "v3c_gen25_tokens_v3.npz"))
    p.add_argument("--arm", required=True, help="arm label; output <out-dir>/<arm>.pt")
    p.add_argument("--n-self", type=int, default=1)
    p.add_argument("--n-cross", type=int, default=1)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--ffn-hidden", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=1.0, help="weight of the listwise policy CE (0 = Q only)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=128, help="DECISIONS per batch (~5.7 candidates each)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--out-dir", default=os.path.join("models", "study11"))
    p.add_argument("--resume", action="store_true", help="continue from <out-dir>/<arm>.last.pt")
    p.add_argument("--max-minutes", type=float, default=None, help="stop cleanly after this many minutes")
    p.add_argument("--limit", type=int, default=None, help="first N decisions only (smoke)")
    p.add_argument("--train-frac", type=float, default=1.0,
                   help="train on a random fraction of the TRAIN games (learning curve); val is unchanged")
    args = p.parse_args(argv)

    arrs = load_v3(args.npz)
    if args.limit:
        arrs = {k: v[:args.limit] for k, v in arrs.items()}
    cfg = LFConfig(d_model=args.d_model, n_self=args.n_self, n_cross=args.n_cross, n_heads=args.n_heads,
                   ffn_hidden=args.ffn_hidden, dropout=args.dropout)
    hp = {k: getattr(args, k) for k in ("epochs", "batch", "lr", "warmup_frac", "val_frac", "seed", "device",
                                        "beta")}
    if args.train_frac < 1.0:                              # only when used: keeps older .last.pt resumable
        hp["train_frac"] = args.train_frac
    os.makedirs(args.out_dir, exist_ok=True)
    last = os.path.join(args.out_dir, f"{args.arm}.last.pt")
    print(f"{args.arm}: {len(arrs['game_id'])} decisions; cfg={cfg}; hp={hp}; "
          f"device={resolve_device(args.device)}", flush=True)
    r = run(arrs, cfg, hp, last_path=last, resume=args.resume, max_minutes=args.max_minutes)
    if not r["done"]:
        return
    print(f"\nparams {r['params']}  TIME {r['seconds']:.1f}s")
    print(f"train_mse {r['train_mse']:.4f}  val_mse {r['val_mse']:.4f}  val_mae {r['val_mae']:.4f}")
    print(f"top1_bestq {r['top1_bestq'] * 100:.2f}%  top1_chosen {r['top1_chosen'] * 100:.2f}%  "
          f"recall2 {r['recall2'] * 100:.2f}%  spearman {r['spearman']:.3f}  | policy top1(visits) "
          f"{r['policy_top1_visits'] * 100:.2f}%  policy top1(chosen) {r['policy_top1_chosen'] * 100:.2f}%  "
          f"ce {r['policy_ce']:.3f}")
    out = os.path.join(args.out_dir, f"{args.arm}.pt")
    keys = ("train_mse", "val_mse", "val_mae", "top1_bestq", "top1_chosen", "recall2", "spearman",
            "policy_top1_visits", "policy_top1_chosen", "policy_ce", "params", "seconds")
    save_lf(out, r["model"], meta={"arm": args.arm, "npz": args.npz, "config": vars(args),
                                   "metrics": {k: r[k] for k in keys}, "by_corpus": r["by_corpus"],
                                   "history": r["history"]})
    print(f"saved {out}")


if __name__ == "__main__":
    main()
