"""Train the belief net over the belief corpus (state -> per-card zone distribution). PyTorch. Study 7.

    python -m imposterkings.machine_learning.train_belief --npz datasets/tensors/belief_5k_v2.npz \
        --d-model 96 --n-heads 6 --ffn-hidden 384 --n-layers 2 --out-dir models/belief_5k

Same attention backbone as ``train_tokens`` but the value head (CLS -> q, MSE) is swapped for a per-card
BELIEF head (each of the 18 card tokens -> softmax over the 12 ZONES) and MSE for **cross-entropy** against
the card's TRUE zone (``features2.true_zones``, read off the omniscient replay). The net is fed the
hand-crafted zone posterior (the socket) and learns a residual, so the identity map IS the hand-crafted
baseline -- any drop in *unseen*-card CE below it is tell-reading beyond card-counting.

Loss is over ALL 18 cards; we report **seen-CE** (should collapse to ~0 -- the sanity check) and
**unseen-CE** (the real objective) separately, plus the uniform and hand-crafted baselines, Brier, and
top-1 zone accuracy. Split by ``game_id`` (15% val by default). GPU via ``--device auto``.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .attention_model import AttentionModel, AttnConfig, collate2
from .attention_model import save as save_attention
from .features2 import Tokens2, ZONES, _HIDDEN_ZONES, _NZONE, _ZONE_OFF, _Z
from .token_dataset import TokenRows, load as load_tokens
from .train import corpus_of, CORPUS_NAMES, game_split
from .train_tokens import lr_lambda, resolve_device, update_to_data

_HID_IX = np.array([_Z[z] for z in _HIDDEN_ZONES], dtype=np.int64)   # the 4 zones an unseen card can be in


class BeliefDataset(Dataset):
    """Yields one belief row: (cards[18,46], kings[2,4], board[4], phase[15], action[51], zone[18], seen[18])."""

    def __init__(self, rows: TokenRows, idx: np.ndarray):
        self.rows, self.idx = rows, idx

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, k):
        i = int(self.idx[k])
        c, kg, b, p, a = self.rows.tokens(i)
        return c, kg, b, p, a, self.rows.zone_label[i], self.rows.seen_mask[i]


def collate_belief(batch):
    """Batch belief rows into model tensors + stack (zone_label [B,18] long, seen_mask [B,18] bool)."""
    packed = collate2([Tokens2(c, k, b, p, a, []) for (c, k, b, p, a, _, _) in batch])
    zl = torch.tensor(np.stack([r[5] for r in batch]), dtype=torch.long)
    sm = torch.tensor(np.stack([r[6] for r in batch]), dtype=torch.bool)
    return packed, zl, sm


def _to_device(packed, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in packed.items()}


def _logits(model, packed):
    out, _ = model.forward_belief(packed["cards"], packed["board"], packed["phase"], packed["action"],
                                  packed["card_mask"], kings=packed.get("kings"))
    return out                                             # [B, 18, Z]


def _ce_per_card(logits, zl):
    """Cross-entropy per card: -log p(true zone). [B,18,Z], [B,18] -> [B,18]."""
    logp = F.log_softmax(logits, dim=-1)
    return -logp.gather(-1, zl.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def _predict_logits(model, rows, idx, batch, device) -> np.ndarray:
    model.eval()
    dl = DataLoader(BeliefDataset(rows, idx), batch_size=batch, shuffle=False, collate_fn=collate_belief)
    out = [_logits(model, _to_device(packed, device)).cpu().numpy() for packed, _, _ in dl]
    return np.concatenate(out) if out else np.zeros((0, 18, _NZONE), np.float32)


def _softmax_np(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def belief_metrics(model, rows: TokenRows, idx: np.ndarray, batch: int, device) -> Dict:
    """Val metrics: seen/unseen CE, uniform + hand-crafted baselines, Brier, top-1 zone accuracy, and a
    per-corpus unseen-CE breakdown. All 'unseen' numbers are averaged over the belief target (the ~9
    cards the observer cannot see)."""
    logits = _predict_logits(model, rows, idx, batch, device)          # [N,18,Z]
    zl = rows.zone_label[idx].astype(np.int64)                         # [N,18]
    seen = rows.seen_mask[idx].astype(bool)                            # [N,18]
    unseen = ~seen
    socket = rows.cards[idx][:, :, _ZONE_OFF:_ZONE_OFF + _NZONE].astype(np.float64)   # the hand-crafted prior

    probs = _softmax_np(logits.astype(np.float64))                    # [N,18,Z]
    p_true = np.take_along_axis(probs, zl[..., None], axis=-1)[..., 0]
    ce = -np.log(np.clip(p_true, 1e-9, 1.0))                          # [N,18]
    onehot = np.eye(_NZONE)[zl]                                        # [N,18,Z]
    brier = ((probs - onehot) ** 2).sum(-1)                           # [N,18]
    top1 = probs.argmax(-1) == zl                                      # [N,18]

    # baselines (unseen only)
    hc_true = np.take_along_axis(socket, zl[..., None], axis=-1)[..., 0]
    hc_ce = -np.log(np.clip(hc_true, 1e-9, 1.0))                      # hand-crafted posterior = the input
    support = socket > 1e-12
    k = support.sum(-1).clip(min=1)                                   # feasible-zone count per card
    uni_true = np.take_along_axis(support, zl[..., None], axis=-1)[..., 0]
    uni_p = np.where(uni_true, 1.0 / k, 1e-9)
    uni_ce = -np.log(uni_p)                                           # uniform over the feasible zones

    def m(a, mask):
        mask = mask & np.isfinite(a)
        return float(a[mask].mean()) if mask.any() else float("nan")

    out = {
        "val_ce": m(ce, np.ones_like(seen)), "seen_ce": m(ce, seen), "unseen_ce": m(ce, unseen),
        "uniform_ce": m(uni_ce, unseen), "handcrafted_ce": m(hc_ce, unseen),
        "unseen_brier": m(brier, unseen), "unseen_top1": m(top1.astype(float), unseen),
        "n_unseen": int(unseen.sum()), "n_seen": int(seen.sum()),
    }
    # per-corpus unseen-CE (decade == corpus id; see train.corpus_of)
    cid = corpus_of(rows.game_id[idx])
    by = {}
    for c in np.unique(cid):
        rowmask = np.zeros_like(seen); rowmask[cid == c] = True
        by[int(c)] = {"name": CORPUS_NAMES.get(int(c), f"seed{int(c)}0M"),
                      "rows": int((cid == c).sum()), "unseen_ce": m(ce, unseen & rowmask)}
    out["by_corpus"] = by
    return out


def run(rows: TokenRows, cfg: AttnConfig, hp: Dict) -> Dict:
    torch.manual_seed(hp["seed"])
    device = resolve_device(hp.get("device", "auto"))
    tr, va = game_split(rows.game_id, hp["val_frac"], hp["seed"])
    tri, vai = np.where(tr)[0], np.where(va)[0]

    model = AttentionModel(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    pin = device.type == "cuda"
    dl = DataLoader(BeliefDataset(rows, tri), batch_size=hp["batch"], shuffle=True,
                    collate_fn=collate_belief, pin_memory=pin)

    total_steps = hp["epochs"] * max(1, len(dl))
    warmup = int(hp.get("warmup_frac", 0.05) * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total_steps, warmup))

    hist: Dict[str, list] = {"step": [], "lr": [], "train_loss": [], "epoch": [],
                             "epoch_train_mse": [], "epoch_val_mse": [], "epoch_secs": [],
                             "ud_step": [], "ud": []}      # 'mse' keys reused for CE so plot_training works
    ud_every = max(1, len(dl) // 4)
    t0 = time.perf_counter()
    epoch_secs, step = [], 0
    for ep in range(hp["epochs"]):
        te = time.perf_counter()
        model.train()
        for packed, zl, sm in dl:
            packed, zl = _to_device(packed, device), zl.to(device)
            opt.zero_grad()
            ce = _ce_per_card(_logits(model, packed), zl)              # [B,18]
            loss = ce.mean()                                           # ALL 18 cards (loss_mask = ones)
            loss.backward()
            # snapshot only weights that actually MOVE: the unused value head gets no grad, and the action
            # embedding gets an all-ZERO grad (belief is state-only -> the action token is a constant zero).
            # Both would log10(0) -> nan and pollute the UD panel, so require a non-zero gradient.
            snap = ({n: p.detach().clone() for n, p in model.named_parameters()
                     if p.ndim >= 2 and p.grad is not None and bool(p.grad.any())}
                    if step % ud_every == 0 else None)
            opt.step()
            if snap is not None:
                hist["ud_step"].append(step)
                hist["ud"].append(update_to_data(model, snap))
            sched.step()
            hist["step"].append(step)
            hist["lr"].append(float(opt.param_groups[0]["lr"]))
            hist["train_loss"].append(float(loss.detach()))
            step += 1
        vm = belief_metrics(model, rows, vai, hp["batch"], device)
        tm = belief_metrics(model, rows, tri[:20000], hp["batch"], device)
        esec = time.perf_counter() - te
        epoch_secs.append(esec)
        hist["epoch"].append(ep + 1)
        hist["epoch_train_mse"].append(tm["unseen_ce"])
        hist["epoch_val_mse"].append(vm["unseen_ce"])
        hist["epoch_secs"].append(esec)
        print(f"  epoch {ep + 1:>2}: train_unseenCE {tm['unseen_ce']:.4f}  val_unseenCE {vm['unseen_ce']:.4f} "
              f" seenCE {vm['seen_ce']:.4f}  (hc {vm['handcrafted_ce']:.4f} / uni {vm['uniform_ce']:.4f})  "
              f"lr {opt.param_groups[0]['lr']:.2e}  ({esec:.1f}s)", flush=True)
    seconds = time.perf_counter() - t0

    final = belief_metrics(model, rows, vai, hp["batch"], device)
    model.cpu()
    n = len(tri) + len(vai)
    return {"model": model, "params": model.param_count(), "metrics": final,
            "history": hist, "seconds": seconds, "epoch_secs": epoch_secs,
            "epochs": hp["epochs"], "rows": n, "n_val": len(vai), "n_train": len(tri)}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train the belief net (per-card zone softmax) over a belief corpus.")
    p.add_argument("--npz", default=os.path.join("datasets", "tensors", "belief_5k_v2.npz"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3, help="PEAK lr (cosine decays it to ~0)")
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--ffn-hidden", type=int, default=384)
    p.add_argument("--out-dir", default=os.path.join("models", "belief_5k"))
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = p.parse_args(argv)

    rows = load_tokens(args.npz)
    if rows.target != "belief":
        raise SystemExit(f"{args.npz} is a '{rows.target}' dataset, not a belief dataset "
                         f"(build with `token_dataset --target belief`).")
    cfg = AttnConfig(d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
                     ffn_hidden=args.ffn_hidden, dropout=args.dropout, feat=rows.feat, head="belief")
    hp = {k: getattr(args, k) for k in ("epochs", "batch", "lr", "val_frac", "seed", "device", "warmup_frac")}
    print(f"loaded {len(rows)} belief rows; cfg={cfg}; device={resolve_device(args.device)}", flush=True)

    r = run(rows, cfg, hp)
    m = r["metrics"]
    print(f"\nparams {r['params']}  epochs {r['epochs']}  TIME {r['seconds']:.1f}s "
          f"({np.mean(r['epoch_secs']):.1f}s/epoch)  train {r['n_train']} / val {r['n_val']} rows")
    print(f"VAL  unseen-CE {m['unseen_ce']:.4f}   vs  hand-crafted {m['handcrafted_ce']:.4f}  "
          f"({(m['handcrafted_ce'] - m['unseen_ce']):+.4f}, the tell-reading gain)  |  uniform "
          f"{m['uniform_ce']:.4f}")
    print(f"     seen-CE {m['seen_ce']:.4f} (sanity, want ~0)   Brier {m['unseen_brier']:.4f}   "
          f"top-1 zone {m['unseen_top1'] * 100:.1f}%")

    ud = r["history"]["ud"]
    if ud:
        mid = ud[len(ud) // 3: 2 * len(ud) // 3]
        names = sorted(ud[0])
        agg = {n: float(np.nanmedian([u[n] for u in mid])) for n in names}
        bad = {n: v for n, v in agg.items() if not (-4.0 <= v <= -2.0)}
        print(f"\nUPDATE-TO-DATA at the plateau (want ~ -3.0):  median "
              f"{np.median(list(agg.values())):+.2f} over {len(names)} tensors"
              + (f"  |  {len(bad)} outside [-4,-2]" if bad else "  |  all inside [-4,-2]"))

    if len(m["by_corpus"]) > 1:
        print("\nval unseen-CE by source corpus:")
        for c in sorted(m["by_corpus"]):
            b = m["by_corpus"][c]
            print(f"  {b['name']:>22} {b['rows']:>7} rows   unseen-CE {b['unseen_ce']:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"belief_d{cfg.d_model}_L{cfg.n_layers}.pt")
    save_attention(out, r["model"], meta={
        "target": "belief", "npz": args.npz, "seconds": r["seconds"], "config": vars(args),
        "metrics": {k: m[k] for k in ("val_ce", "seen_ce", "unseen_ce", "uniform_ce", "handcrafted_ce",
                                      "unseen_brier", "unseen_top1")},
        "baselines": {"uniform_ce": m["uniform_ce"], "handcrafted_ce": m["handcrafted_ce"]},
        "by_corpus": m["by_corpus"], "history": r["history"]})
    print(f"saved {out}   (history: {len(r['history']['step'])} steps, {len(r['history']['epoch'])} epochs "
          f"-- plot with plot_training.py)")


if __name__ == "__main__":
    main()
