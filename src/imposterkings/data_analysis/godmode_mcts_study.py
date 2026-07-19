"""Godmode-MCTS: does PERFECT-INFORMATION play collapse the imperfect-info ~60/40 second-player edge?

Determinize ONCE to the true state, then plain MCTS with a fixed budget (no re-sampling, no flip handicap,
no outcome-dependent solve cost). Head-to-head, mirrored on seat. Two knobs per side (``ismcts``|``godmode``)
so the same harness runs:

    # both perfect-info -> the headline second-player winrate (value net leaf eval)
    python -m imposterkings.data_analysis.godmode_mcts_study --a godmode --b godmode \
        --eval models/v3c_gen25/attn_d96_L2.pt --sims 400 --deals 150 --out reports/gm_both_nn.csv
    # both perfect-info, PURE ROLLOUT (fully perfect-info, high N)
    python ... --a godmode --b godmode --eval none --sims 1500 --deals 150
    # asymmetric -> the VALUE of perfect information (godmode-agent winrate)
    python ... --a godmode --b ismcts --eval models/v3c_gen25/attn_d96_L2.pt --sims 400
    # baseline sanity -> reproduce ~60/40
    python ... --a ismcts --b ismcts --eval models/v3c_gen25/attn_d96_L2.pt --sims 400

Reports BOTH agent-A's winrate (for the asymmetric arm = value of perfect info) and the SECOND-player
(non-starter) winrate (for the symmetric arms = the structural bias), each +/- 95% CI over deals.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from typing import List, Optional

import numpy as np

from ..agents import MCTSAgent
from ..arena import play_game


def _chunk(seeds: List[int], a_mode: str, b_mode: str, eval_ckpt: Optional[str], sims: int,
           budget_spec: Optional[str] = None) -> List[dict]:
    import torch
    torch.set_num_threads(1)
    ev = None
    if eval_ckpt:
        from ..machine_learning.benchmark import _evaluator_for
        ev = _evaluator_for(eval_ckpt)
    budget = None
    if budget_spec:                                   # e.g. hybrid-k20-l3: adaptive iters (fixed FORMULA)
        from ..budget import make_budget
        from ..machine_learning.benchmark import parse_spec
        mode, k, l = parse_spec(budget_spec)
        budget = make_budget(mode, k=k, l=l)

    def mk(mode):
        return MCTSAgent(iterations=sims, budget=budget, evaluator=ev, godmode=(mode == "godmode"))

    agent_a, agent_b = mk(a_mode), mk(b_mode)

    rows = []
    for seed in seeds:
        for a_seat in (0, 1):                             # mirror A's seat -> cancels seat + starter advantage
            agents = [None, None]
            agents[a_seat] = agent_a
            agents[1 - a_seat] = agent_b
            winner, _, term = play_game(agents, np.random.default_rng(seed),
                                        play_rng=np.random.default_rng([seed, a_seat]))
            rows.append(dict(seed=seed, a_seat=a_seat, starter=term.starting_player, winner=winner,
                             a_win=int(winner == a_seat), second_win=int(winner != term.starting_player)))
    return rows


def _rate(per_deal: List[float]):
    a = np.array(per_deal, float)
    n = len(a)
    p = float(a.mean()) if n else 0.0
    ci = 1.96 * a.std(ddof=1) / math.sqrt(n) if n > 1 else 0.0
    return p, ci, n


def _report(rows: List[dict], label: str, wall: float = 0.0) -> dict:
    by_deal = {}
    for r in rows:
        by_deal.setdefault(r["seed"], []).append(r)
    a_scores = [np.mean([g["a_win"] for g in gs]) for gs in by_deal.values()]
    s_scores = [np.mean([g["second_win"] for g in gs]) for gs in by_deal.values()]
    ap, aci, n = _rate(a_scores)
    sp, sci, _ = _rate(s_scores)
    print(f"\n{'=' * 74}\n{label}   ({n} deals x 2 seatings = {len(rows)} games)\n{'=' * 74}")
    print(f"  agent-A winrate       : {ap * 100:5.1f}%  +/- {aci * 100:.1f}   "
          f"CI95 [{(ap - aci) * 100:.1f}, {(ap + aci) * 100:.1f}]   (asymmetric arm = value of perfect info)")
    print(f"  SECOND-player winrate : {sp * 100:5.1f}%  +/- {sci * 100:.1f}   "
          f"CI95 [{(sp - sci) * 100:.1f}, {(sp + sci) * 100:.1f}]   (symmetric arm = structural bias vs ~60%)")
    if wall:
        print(f"  wall {wall:.0f}s")
    return {"label": label, "n": n, "a_winrate": ap, "second_winrate": sp}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Godmode-MCTS: perfect-info second-player-bias / value of PI.")
    p.add_argument("--a", default="godmode", choices=["ismcts", "godmode"], help="seat-A search mode")
    p.add_argument("--b", default="godmode", choices=["ismcts", "godmode"], help="seat-B search mode")
    p.add_argument("--eval", default="none", help="value-net ckpt for BOTH sides, or 'none' (pure rollout)")
    p.add_argument("--sims", type=int, default=400, help="fixed simulations per decision (if --budget unset)")
    p.add_argument("--budget", default=None,
                   help="adaptive budget spec, e.g. hybrid-k20-l3 (overrides --sims; fixed FORMULA)")
    p.add_argument("--deals", type=int, default=150)
    p.add_argument("--base-seed", type=int, default=70_000_000)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--chunk", type=int, default=5)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    eval_ckpt = None if args.eval in ("none", "None") else args.eval
    budget_spec = None if args.budget in (None, "none", "None") else args.budget
    label = (f"A={args.a} vs B={args.b}  "
             + (f"eval={os.path.basename(eval_ckpt)}" if eval_ckpt else "PURE ROLLOUT")
             + (f"  budget={budget_spec}" if budget_spec else f"  sims={args.sims}"))
    print(f"godmode-MCTS: {label}\n  {args.deals} deals x 2 seatings, workers {args.workers}", flush=True)

    seeds = list(range(args.base_seed, args.base_seed + args.deals))
    chunks = [seeds[i:i + args.chunk] for i in range(0, len(seeds), args.chunk)]
    from joblib import Parallel, delayed
    from tqdm import tqdm
    t0 = time.perf_counter()
    jobs = [delayed(_chunk)(c, args.a, args.b, eval_ckpt, args.sims, budget_spec) for c in chunks]
    rows: List[dict] = []
    for r in tqdm(Parallel(n_jobs=args.workers, return_as="generator")(jobs), total=len(jobs),
                  desc="chunks", unit="chunk"):
        rows.extend(r)
    rows.sort(key=lambda r: (r["seed"], r["a_seat"]))
    _report(rows, label, time.perf_counter() - t0)

    if args.out:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["seed", "a_seat", "starter", "winner", "a_win", "second_win"])
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
