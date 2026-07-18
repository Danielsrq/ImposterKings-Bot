"""A/B: does belief-weighted determinization beat uniform? SAME agent on both sides -- the belief net is the
only difference -- so the delta is purely "which opponent worlds the search samples."

    # Test A -- pure-rollout N500, NO value net (isolates the determinization step)
    python -m imposterkings.data_analysis.belief_ab --spec fixed500 \
        --belief models/belief_5k/belief_d64_L2.pt --deals 100 --out reports/belief_ab_rolloutN500.csv

    # Test B -- d96 value net, hybrid-k20-l3
    python -m imposterkings.data_analysis.belief_ab --spec hybrid-k20-l3 \
        --value-nn models/v3c_gen25/attn_d96_L2.pt --belief models/belief_5k/belief_d64_L2.pt \
        --deals 100 --out reports/belief_ab_d96_k20l3.csv

Mirror = seat-swap: the belief agent plays seat 0 then seat 1 on the SAME deal (same deal RNG, independent
play RNG), which cancels both seat and starter advantage (the starter is seat-tied). Per-deal pair score in
{0, .5, 1}; winrate +/- 95% CI over the deals. Reuses the benchmark/forced_hide harness pattern.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from typing import List, Optional

import numpy as np

from ..arena import play_game
from ..machine_learning.benchmark import parse_spec
from .budget_scaling import make_agent, spec_label


def _chunk(seeds: List[int], spec, value_ckpt: Optional[str], belief_ckpt: str) -> List[dict]:
    """One worker: build both agents once, play the mirrored pair for each deal seed."""
    import torch
    torch.set_num_threads(1)
    from ..machine_learning.benchmark import _evaluator_for
    from ..machine_learning.attention_model import build_belief_sampler

    ev = _evaluator_for(value_ckpt) if value_ckpt else None      # same value net on BOTH sides (or none)
    sampler = build_belief_sampler(belief_ckpt)
    belief_agent = make_agent(spec, evaluator=ev, belief=sampler)
    base_agent = make_agent(spec, evaluator=ev, belief=None)

    rows = []
    for seed in seeds:
        wins = 0
        for bs in (0, 1):                                        # belief agent's seat, MIRRORED
            agents = [None, None]
            agents[bs] = belief_agent
            agents[1 - bs] = base_agent
            play_rng = np.random.default_rng([seed, bs])         # independent play randomness per arm
            winner, _, _ = play_game(agents, np.random.default_rng(seed), play_rng=play_rng)
            wins += int(winner == bs)
        rows.append({"seed": seed, "belief_wins": wins, "score": wins / 2.0, "split": int(wins == 1)})
    return rows


def _report(rows: List[dict], label: str, wall: float = 0.0) -> dict:
    arr = np.array([r["score"] for r in rows], float)
    n = len(rows)
    winrate = float(arr.mean()) if n else 0.0
    ci = 1.96 * arr.std(ddof=1) / math.sqrt(n) if n > 1 else 0.0
    wins = int(round(arr.sum() * 2))
    games = 2 * n
    splits = sum(r["split"] for r in rows)
    lo, hi = (winrate - ci) * 100, (winrate + ci) * 100
    print(f"\n{'=' * 74}\n{label}   ({n} mirrored deals = {games} games)\n{'=' * 74}")
    print(f"  belief-agent winrate : {winrate * 100:.1f}%  +/- {ci * 100:.1f}   CI95 [{lo:.1f}, {hi:.1f}]")
    print(f"  games                : {wins}/{games}   (deals split by seat: {splits}/{n})")
    verdict = ("belief WINS (CI clears 50%)" if lo > 50 else
               "belief LOSES (CI below 50%)" if hi < 50 else "no significant difference (CI straddles 50%)")
    print(f"  => {verdict}")
    if wall:
        print(f"  wall {wall:.0f}s")
    return {"label": label, "n": n, "winrate": winrate, "ci": ci, "wins": wins, "games": games}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="A/B belief-weighted vs uniform determinization (same agent).")
    p.add_argument("--spec", default="fixed500", help="fixed<N> (pure rollout) or hybrid-k<k>-l<l>")
    p.add_argument("--value-nn", default=None, help="value-net checkpoint applied to BOTH sides (default none)")
    p.add_argument("--belief", required=True, help="belief checkpoint driving the +belief side")
    p.add_argument("--deals", type=int, default=100, help="mirrored deals (each -> 2 games)")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--chunk", type=int, default=10)
    p.add_argument("--out", default=None, help="CSV path for the per-deal rows")
    args = p.parse_args(argv)

    spec = parse_spec(args.spec)
    value_ckpt = None if (args.value_nn in (None, "none", "None")) else args.value_nn
    label = (f"{spec_label(spec)}"
             + (f" + {os.path.basename(value_ckpt)}" if value_ckpt else " (pure rollout)")
             + f"  belief[{os.path.basename(args.belief)}] vs uniform")
    print(f"belief A/B: {label}\n  {args.deals} deals x 2 seats, workers={args.workers}", flush=True)

    seeds = list(range(args.base_seed, args.base_seed + args.deals))
    chunks = [seeds[i:i + args.chunk] for i in range(0, len(seeds), args.chunk)]
    from joblib import Parallel, delayed
    from tqdm import tqdm
    t0 = time.perf_counter()
    jobs = [delayed(_chunk)(c, spec, value_ckpt, args.belief) for c in chunks]
    rows: List[dict] = []
    for r in tqdm(Parallel(n_jobs=args.workers, return_as="generator")(jobs), total=len(jobs),
                  desc="chunks", unit="chunk"):
        rows.extend(r)
    rows.sort(key=lambda r: r["seed"])
    summary = _report(rows, label, time.perf_counter() - t0)

    out = args.out
    if out:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["seed", "belief_wins", "score", "split"])
            w.writeheader()
            w.writerows(rows)
        with open(out + ".summary.json", "w", encoding="utf-8") as fh:
            import json
            json.dump(summary, fh, indent=1)
        print(f"wrote {out} (+ .summary.json)")


if __name__ == "__main__":
    main()
