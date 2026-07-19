"""Godmode study: the perfect-information MAIN-GAME tempo bias -- is the second player structurally favoured
once both setups are fixed and the rest is played to optimality?

Full-game solving (including the setup hide/discard fan-out) is intractable in pure Python (see the godmode
probe), so we fix BOTH players' setup with a strong fixed policy (the v3c_gen25 value net, greedy) and then
**solve the resulting main game EXACTLY**. Over many deals, mirrored on the starter, the fraction of solved
games the SECOND player wins is the structural main-game bias -- to set against the MCTS ~60/40.

    python -m imposterkings.data_analysis.godmode_study --deals 200 --setup-nn models/v3c_gen25/attn_d96_L2.pt \
        --time-limit 60 --out reports/godmode_maingame.csv

Deferred to a faster (Rust) engine: the full-game value with OPTIMAL setup, and hide-Assassin-at-setup.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from typing import List, Optional

import numpy as np

from ..godmode import Budget, solve
from ..state import GameState


def _post_setup(deal: GameState, setup_agent, rng) -> GameState:
    """Play the SETUP phase (both seats) with the fixed policy; return the first MAIN state (or a terminal
    state if the game somehow ended during setup, which it should not)."""
    s = deal
    while not s.is_terminal() and s.phase.name.startswith("SETUP"):
        view = s.information_set(s.to_play)
        s = s.apply(setup_agent.select_move(view, rng))
    return s


def _chunk(seeds: List[int], setup_ckpt: Optional[str], time_limit: float, order_depth: int,
           flip: bool = False) -> List[dict]:
    import torch
    torch.set_num_threads(1)
    from ..machine_learning.agent import NNAgent
    from ..agents import RandomAgent
    setup_agent = NNAgent.from_checkpoint(setup_ckpt) if setup_ckpt else RandomAgent()
    order = None
    if order_depth > 0 and setup_ckpt:                    # shallow value-net move ordering (exact accelerant)
        from ..godmode import make_nn_order
        from ..machine_learning.attention_model import build_evaluator
        order = make_nn_order(build_evaluator(setup_ckpt), max_depth=order_depth)
    rng = np.random.default_rng(0)

    rows = []
    for seed in seeds:
        for sp in (0, 1):                                 # mirror the starter on the SAME dealt hands
            deal = GameState.deal(np.random.default_rng(seed), starting_player=sp)
            s = _post_setup(deal, setup_agent, rng)
            if flip:                                      # net-play the STARTER's move(s) so the solve ROOT
                while not s.is_terminal() and s.to_play == sp:   # is the SECOND player -> flips which
                    s = s.apply(setup_agent.select_move(s.information_set(s.to_play), rng))  # outcome is cheap
            second = 1 - sp
            if s.is_terminal():
                rows.append(dict(seed=seed, starter=sp, second=second, winner=s.winner,
                                 solved=True, nodes=0, wall=0.0))
                continue
            r = solve(s, Budget(time_limit=time_limit), order=order)
            b = r.budget
            rows.append(dict(seed=seed, starter=sp, second=second,
                             winner=(None if r.value is None else r.winner),
                             solved=bool(b.solved), nodes=b.nodes, wall=b.wall))
    return rows


def _report(rows: List[dict], wall: float = 0.0) -> dict:
    solved = [r for r in rows if r["solved"]]
    n, ns = len(rows), len(solved)
    print(f"\n{'=' * 74}\nGODMODE main-game bias   ({n} solves = {n // 2} deals x 2 starters)\n{'=' * 74}")
    print(f"  solved within budget : {ns}/{n} = {ns / n * 100:.1f}%" if n else "  no rows")
    def second_rate(sub):
        if not sub:
            return None
        sec = np.array([1.0 if r["winner"] == r["second"] else 0.0 for r in sub])
        p = float(sec.mean())
        return p, 1.96 * math.sqrt(p * (1 - p) / len(sub)), len(sub)

    if ns:
        p, ci, _ = second_rate(solved)
        print(f"  SECOND-player win rate: {p * 100:.1f}%  +/- {ci * 100:.1f}   CI95 "
              f"[{(p - ci) * 100:.1f}, {(p + ci) * 100:.1f}]   (structural main-game bias vs MCTS ~60%)")
        # selection-bias probe: does the bias hold among the EASY (fast) vs HARD (slow) solved deals? If the
        # two halves agree, the 30% unsolved tail is less likely to swing the estimate.
        by_nodes = sorted(solved, key=lambda r: r["nodes"])
        half = len(by_nodes) // 2
        for lbl, sub in (("fast-solved half", by_nodes[:half]), ("slow-solved half", by_nodes[half:])):
            r = second_rate(sub)
            if r:
                print(f"    {lbl:>16}: 2nd-player {r[0] * 100:5.1f}% (n={r[2]})")
        nodes = sorted(r["nodes"] for r in solved if r["nodes"] > 0)
        walls = sorted(r["wall"] for r in solved if r["wall"] > 0)
        if nodes:
            med = lambda a: a[len(a) // 2]
            print(f"  solved-deal nodes: median {med(nodes):,} max {max(nodes):,}   "
                  f"wall: median {med(walls):.1f}s max {max(walls):.1f}s")
    if wall:
        print(f"  wall {wall:.0f}s")
    return {"n": len(rows), "solved": ns}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Godmode perfect-info main-game second-player-bias study.")
    p.add_argument("--deals", type=int, default=200)
    p.add_argument("--base-seed", type=int, default=70_000_000)
    p.add_argument("--setup-nn", default=os.path.join("models", "v3c_gen25", "attn_d96_L2.pt"),
                   help="value net (greedy) that plays the fixed setup for both seats; 'none' = random setup")
    p.add_argument("--time-limit", type=float, default=60.0, help="per-solve wall cap (s); unsolved reported")
    p.add_argument("--order-depth", type=int, default=8,
                   help="shallow value-net move-ordering depth (0 = off); the setup net is reused as orderer")
    p.add_argument("--flip", action="store_true",
                   help="net-play the starter's first move too, so the solve ROOT is the SECOND player "
                        "(flips which outcome is the cheap-to-prove direction)")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--chunk", type=int, default=5)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    setup_ckpt = None if args.setup_nn in ("none", "None") else args.setup_nn
    print(f"godmode study: {args.deals} deals x 2 starters, setup={'random' if not setup_ckpt else os.path.basename(setup_ckpt)}, "
          f"solve cap {args.time_limit}s, workers {args.workers}", flush=True)
    seeds = list(range(args.base_seed, args.base_seed + args.deals))
    chunks = [seeds[i:i + args.chunk] for i in range(0, len(seeds), args.chunk)]
    from joblib import Parallel, delayed
    from tqdm import tqdm
    t0 = time.perf_counter()
    jobs = [delayed(_chunk)(c, setup_ckpt, args.time_limit, args.order_depth, args.flip) for c in chunks]
    rows: List[dict] = []
    for r in tqdm(Parallel(n_jobs=args.workers, return_as="generator")(jobs), total=len(jobs),
                  desc="chunks", unit="chunk"):
        rows.extend(r)
    rows.sort(key=lambda r: (r["seed"], r["starter"]))
    _report(rows, time.perf_counter() - t0)

    if args.out:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["seed", "starter", "second", "winner", "solved", "nodes", "wall"])
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
