"""AlphaZero-style bootstrapping loop: NN-MCTS self-play -> retrain -> repeat.

    python -m imposterkings.machine_learning.loop --iterations 3 --games 2000 --mode fixed --k 600 \
        --arch 256 --temp-plies 6 --init-checkpoint models/mlp_32.pt --out-dir runs/az1 --workers 10

Each iteration: (1) generate `--games` self-play games with the current net as an NN-MCTS (PUCT) value/
policy head; (2) build training tensors; (3) train `--arch` on them -> a new checkpoint; (4) measure the
climb by pitting the new NN-MCTS bot against a fixed rollout-MCTS reference. Everything but the loop glue is
the existing datagen/dataset/train/evaluator machinery.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import List, Tuple

import numpy as np

from ..arena import play_game
from ..data_analysis import datagen
from ..data_analysis.budget_scaling import make_agent, spec_label
from . import dataset
from .benchmark import parse_spec
from .checkpoint import save as save_checkpoint
from .train import _arch_str, game_split, load_npz, parse_archs, train_one


def _eval_chunk(ckpt: str, spec: Tuple, opp: Tuple, seeds: List[int]) -> float:
    """NN-MCTS(ckpt) vs a rollout-MCTS opponent over mirrored paired deals -> challenger wins (sum)."""
    from .evaluator import build_evaluator
    ev = build_evaluator(ckpt)
    wins = 0.0
    for seed in seeds:
        for cs in (0, 1):                                 # challenger (NN-MCTS) in each seat
            agents = [None, None]
            agents[cs] = make_agent(spec, evaluator=ev)
            agents[1 - cs] = make_agent(opp)
            winner, _, _ = play_game(agents, np.random.default_rng(seed),
                                     play_rng=np.random.default_rng([seed, cs]))
            wins += int(winner == cs)
    return wins


def evaluate(ckpt: str, spec: Tuple, opp: Tuple, deals: int, workers: int, base_seed: int = 10 ** 6) -> float:
    from joblib import Parallel, delayed
    seeds = [base_seed + i for i in range(deals)]
    step = max(1, deals // max(workers, 1))
    chunks = [seeds[i:i + step] for i in range(0, deals, step)]
    wins = sum(Parallel(n_jobs=workers)(delayed(_eval_chunk)(ckpt, spec, opp, ch) for ch in chunks))
    return wins / (2 * deals)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="AlphaZero-style NN-MCTS self-play bootstrapping loop.")
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--games", type=int, default=2000, help="self-play games per iteration")
    p.add_argument("--mode", choices=["hybrid", "branching", "fixed"], default="hybrid")
    p.add_argument("--k", type=int, default=20, help="NN-MCTS budget: N (fixed) or k (hybrid/branching)")
    p.add_argument("--l", type=int, default=3)
    p.add_argument("--sweep", default="32;64;128",
                   help="MLP archs to sweep each iter (';'-separated, ','-separated layers); "
                        "best by top-1 agreement is kept (e.g. '32;64;128' or '256')")
    p.add_argument("--temp-plies", type=int, default=6, help="temperature-sampled opening plies (diversity)")
    p.add_argument("--init-checkpoint", default=os.path.join("models", "mlp_32.pt"))
    p.add_argument("--out-dir", default=os.path.join("runs", "az1"))
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--chunk", type=int, default=25)
    p.add_argument("--eval-deals", type=int, default=50)
    p.add_argument("--eval-opponent", default="hybrid-k50-l3", help="fixed rollout-MCTS reference")
    p.add_argument("--epochs", type=int, default=50)
    args = p.parse_args(argv)

    spec = ("fixed", args.k) if args.mode == "fixed" else (args.mode, args.k, args.l)
    opp = parse_spec(args.eval_opponent)
    archs = parse_archs(args.sweep)
    hp = {"epochs": args.epochs, "batch": 1024, "lr": 1e-3, "dropout": 0.0, "patience": 5, "seed": 0}
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"loop: {args.iterations} iters | NN-MCTS={spec_label(spec)} | "
          f"arch sweep {args.sweep} (select by top-1) | {args.games} games/iter | "
          f"eval vs {args.eval_opponent} | init={args.init_checkpoint}")

    loop_t0 = time.perf_counter()
    current = args.init_checkpoint
    for it in range(1, args.iterations + 1):
        it_dir = os.path.join(args.out_dir, f"iter{it:02d}")
        corpus, npz = os.path.join(it_dir, "corpus"), os.path.join(it_dir, "train.npz")
        os.makedirs(corpus, exist_ok=True)
        t0 = time.perf_counter()

        print(f"\n=== iter {it}: self-play with {current} ===")
        datagen.run(spec, args.games, args.workers, args.chunk, base_seed=it * 10 ** 7,
                    temp_plies=args.temp_plies, out_dir=corpus, value_ckpt=current)
        dataset.build(corpus, npz)

        data = load_npz(npz)
        tr, va = game_split(data["game_id"], 0.1, 0)
        print(f"  {'arch':>10} {'params':>7} {'ep':>3} {'secs':>5} {'trn_mse':>8} {'val_mse':>8} "
              f"{'gap':>6} {'top1_q':>7} {'rec@2':>6} {'spear':>6}")
        results = []
        for arch in archs:
            r = train_one(data, tr, va, arch, hp)
            print(f"  {_arch_str(arch):>10} {r['params']:>7} {r['epochs']:>3} {r['seconds']:>5.1f} "
                  f"{r['train_mse']:>8.4f} {r['val_mse']:>8.4f} {r['val_mse']-r['train_mse']:>6.4f} "
                  f"{r['top1_bestq']*100:>6.1f}% {r['recall2']*100:>5.1f}% {r['spearman']:>6.3f}")
            save_checkpoint(os.path.join(it_dir, f"mlp_{_arch_str(arch)}.pt"), r["model"],
                            meta={"iter": it, "arch": arch, "from": current, "target": "q",
                                  "metrics": {k: r[k] for k in ("val_mse", "recall2", "top1_bestq", "spearman")}})
            results.append(r)

        best = max(results, key=lambda r: (r["top1_bestq"], -r["val_mse"]))   # top-1 primary, val_mse tiebreak
        new_ckpt = os.path.join(it_dir, "mlp.pt")
        save_checkpoint(new_ckpt, best.pop("model"),
                        meta={"iter": it, "arch": best["arch"], "from": current, "target": "q",
                              "selected_by": "top1_bestq",
                              "metrics": {k: best[k] for k in ("val_mse", "recall2", "top1_bestq", "spearman")}})
        print(f"  selected: arch={_arch_str(best['arch'])} top1={best['top1_bestq']*100:.1f}% "
              f"val_mse={best['val_mse']:.4f}")

        wr = evaluate(new_ckpt, spec, opp, args.eval_deals, args.workers)
        print(f"iter {it}: NN-MCTS({_arch_str(best['arch'])}) vs {args.eval_opponent}: {wr*100:.1f}%  "
              f"({time.perf_counter()-t0:.0f}s)")
        current = new_ckpt

    wall = time.perf_counter() - loop_t0
    print(f"\ndone. final checkpoint: {current}")
    print(f"total wall: {wall:.0f}s ({wall/60:.1f} min)")


if __name__ == "__main__":
    main()
