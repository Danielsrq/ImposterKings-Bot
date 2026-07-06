"""Benchmark an NN checkpoint vs parameterizable MCTS opponents (mirrored, paired seeds).

    python -m imposterkings.machine_learning.benchmark --model models/mlp_32.pt \
        --opponent hybrid-k20-l3 fixed500 hybrid-k30-l3 --deals 100 --workers 10

The NN plays greedily (highest predicted q); each game pairs it against one MCTS opponent, so games are
cheap (the NN is ~free). Mirrored seating + paired deal seeds + independent play-rng reduce variance the
same way ``data_analysis.budget_scaling`` does. Reports win-rate + CI95 + split-rate per opponent.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from ..agents import MCTSAgent
from ..arena import play_game
from ..budget import make_budget

Spec = Tuple


def parse_spec(s: str) -> Spec:
    """'fixed500' -> ('fixed',500); 'hybrid-k20-l3' -> ('hybrid',20,3)."""
    if s.startswith("fixed"):
        return ("fixed", int(s[len("fixed"):]))
    mode, k, l = s.split("-")
    return (mode, int(k.lstrip("k")), int(l.lstrip("l")))


def parse_opponent(s: str) -> Spec:
    """A `.pt` path -> ('nn', path) (an NN opponent, e.g. self-play); else an MCTS spec string."""
    return ("nn", s) if s.endswith(".pt") else parse_spec(s)


def _make_agent(spec: Spec, evaluator=None):
    if spec[0] == "nn":
        from .agent import NNAgent
        return NNAgent.from_checkpoint(spec[1])
    if spec[0] == "fixed":
        return MCTSAgent(iterations=spec[1], evaluator=evaluator)
    return MCTSAgent(budget=make_budget(spec[0], k=spec[1], l=spec[2]), evaluator=evaluator)


def _chunk(ckpt: str, label: str, spec: Spec, seeds: List[int], independent_rng: bool,
           nn_mcts: str = None) -> Dict:
    import torch
    torch.set_num_threads(1)                               # avoid thread oversubscription across workers
    t0 = time.perf_counter()
    if nn_mcts is not None:                               # challenger is the net AS AN MCTS eval/policy head
        from .evaluator import build_evaluator
        nn = _make_agent(parse_spec(nn_mcts), evaluator=build_evaluator(ckpt))
    else:                                                 # challenger is the greedy net (no search)
        from .agent import NNAgent
        nn = NNAgent.from_checkpoint(ckpt)
    opp = _make_agent(spec)                               # opponent (MCTS or NN); stateless -> reuse
    rows = []
    for seed in seeds:
        nn_wins = 0
        for ns in (0, 1):                                 # NN seat = ns, mirrored
            agents = [None, None]
            agents[ns] = nn
            agents[1 - ns] = opp
            play_rng = np.random.default_rng([seed, ns]) if independent_rng else None
            winner, _, _ = play_game(agents, np.random.default_rng(seed), play_rng=play_rng)
            nn_wins += int(winner == ns)
        rows.append({"opponent": label, "seed": seed, "nn_wins": nn_wins,
                     "split": int(nn_wins == 1), "score": nn_wins / 2.0})
    return {"label": label, "rows": rows, "seconds": time.perf_counter() - t0}


def run(ckpt: str, opponents: List[Tuple[str, Spec]], deals: int, workers: int, chunk: int,
        base_seed: int, independent_rng: bool, nn_mcts: str = None) -> List[Dict]:
    from joblib import Parallel, delayed
    from tqdm import tqdm

    seeds = [base_seed + i for i in range(deals)]
    chunks = [seeds[i:i + chunk] for i in range(0, deals, chunk)]
    jobs = [delayed(_chunk)(ckpt, label, spec, ch, independent_rng, nn_mcts)
            for label, spec in opponents for ch in chunks]
    results = list(tqdm(Parallel(n_jobs=workers, return_as="generator")(jobs),
                        total=len(jobs), desc="chunks", unit="chunk"))

    by = defaultdict(lambda: {"rows": [], "seconds": 0.0})
    for r in results:
        by[r["label"]]["rows"].extend(r["rows"])
        by[r["label"]]["seconds"] += r["seconds"]
    out = []
    for label, _ in opponents:
        rows = by[label]["rows"]
        arr = np.array([x["score"] for x in rows], dtype=float)
        n = len(rows)
        ci = float(1.96 * arr.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        out.append({"opponent": label, "deals": n, "games": 2 * n, "wins": int(round(arr.sum() * 2)),
                    "winrate": float(arr.mean()), "ci95": ci,
                    "splits": sum(x["split"] for x in rows), "seconds": by[label]["seconds"]})
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="NN checkpoint win-rate vs MCTS opponents.")
    p.add_argument("--model", default=os.path.join("models", "mlp_32.pt"))
    p.add_argument("--opponent", nargs="+", default=["hybrid-k20-l3", "fixed500", "hybrid-k30-l3"],
                   help="opponents: fixed<N> | hybrid-k<k>-l<l> | branching-k<k>-l<l> | a .pt path (NN, e.g. self-play)")
    p.add_argument("--deals", type=int, default=100, help="deals per opponent; each mirrored -> 2x games")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--chunk", type=int, default=10)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--shared-rng", action="store_true")
    p.add_argument("--nn-mcts", default=None,
                   help="use the net AS AN MCTS eval/policy head at this budget (e.g. fixed500, hybrid-k20-l3) "
                        "instead of greedy play")
    p.add_argument("--out", default=None, help="optional CSV of the per-opponent results")
    args = p.parse_args(argv)

    opponents = [(f"nn:{os.path.basename(s)}" if s.endswith(".pt") else s, parse_opponent(s))
                 for s in args.opponent]
    challenger = f"NN-MCTS({args.nn_mcts})" if args.nn_mcts else "greedy-NN"
    print(f"benchmark {challenger} <- {args.model}  vs {[o[0] for o in opponents]}  | {args.deals} deals "
          f"x2 mirrored = {args.deals*2} games/opponent | workers={args.workers}")
    t0 = time.perf_counter()
    rows = run(args.model, opponents, args.deals, args.workers, args.chunk, args.base_seed,
               not args.shared_rng, nn_mcts=args.nn_mcts)
    wall = time.perf_counter() - t0

    print(f"\n  {'opponent':>16} {'win%':>6} {'ci95':>5} {'wins':>10} {'split%':>6} {'s/game':>7} {'secs':>7}")
    for r in rows:
        print(f"  {r['opponent']:>16} {r['winrate']*100:>5.1f}% {r['ci95']*100:>4.1f}% "
              f"{r['wins']:>4}/{r['games']:<5} {r['splits']/r['deals']*100:>5.0f}% "
              f"{r['seconds']/r['games']:>7.3f} {r['seconds']:>7.0f}")
    print(f"\ntotal wall: {wall:.0f}s")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print("saved " + args.out)


if __name__ == "__main__":
    main()
