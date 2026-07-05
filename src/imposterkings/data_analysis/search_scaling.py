"""Search-scaling study: MCTS@N vs a fixed MCTS@baseline, with starting-position evals + calibration.

    python -m imposterkings.data_analysis.search_scaling                 # 25..500 step 25, 50 deals each
    python -m imposterkings.data_analysis.search_scaling --deals 100 --workers 8

For each N and each deal we collect three things:
  1. **Win-rate** of MCTS@N vs MCTS@baseline (mirrored seating, paired seeds).
  2. **Starting-position evals** by BOTH bots of the initial dealt position (mover's perspective):
     the root value (visit-weighted mean Q) and the top-2 root moves' Q values. Since the baseline is
     fixed and each deal's eval uses a fixed rng seed, MCTS@baseline's eval for a deal is identical at
     every N, and everything converges (diff -> 0) as N -> baseline -- so you can watch the top-2
     evaluations of MCTS@N approach MCTS@baseline's.
  3. **Calibration**: was each bot's verdict right? A bot "predicts a win" when its starting eval > 0;
     we then check the actual game in which THAT bot played the evaluated (starting) seat and record
     whether the starting seat won. ``correct`` = (predicted win == actually won).

Variance reduction: paired per-deal seeds (identical deal across all N) + mirrored seating (each deal
played in both seatings). Embarrassingly parallel by N-level (one joblib task per level).
"""
from __future__ import annotations

import argparse
import math
import os
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np

from ..agents import MCTSAgent
from ..arena import play_game
from ..mcts import SearchConfig, search
from ..state import GameState


def _summarize(result):
    """(root value, best-move Q, 2nd-best-move Q, best Action) from a search (stats visit-sorted)."""
    stats = result.stats
    q1 = stats[0].mean_q if stats else 0.0
    q2 = stats[1].mean_q if len(stats) > 1 else q1
    return result.root_value(), q1, q2, result.best_move


_KNOW_COLS = [f"g{cs}_p{s}_first_{lvl}"
              for cs in (0, 1) for s in (0, 1) for lvl in ("binary", "perfect")]


def _knowledge_tracker():
    """An ``on_decision`` hook plus a dict recording the first ply each seat first reaches binary /
    perfect knowledge of the opponent's hand (-1 = never). Cost is O(combinations) per ply."""
    ms = {(s, lvl): -1 for s in (0, 1) for lvl in ("binary", "perfect")}
    ply = [0]

    def on_decision(seat, view, move, agent, state):
        for s in (0, 1):
            lvl = state.information_set(s).knowledge_level()
            if lvl in ("binary", "perfect") and ms[(s, lvl)] < 0:
                ms[(s, lvl)] = ply[0]
        ply[0] += 1

    return on_decision, ms


def _eval_level(n: int, deals: int, baseline: int, base_seed: int, collect_eval: bool = True,
                independent_rng: bool = False, knowledge: bool = False) -> Dict:
    """Worker: MCTS@n vs MCTS@baseline over ``deals`` mirrored, paired-seed deals -> one curve point,
    plus (optionally) both bots' starting-position evals and win/prediction calibration per deal.

    ``independent_rng`` gives the two mirrored games independent *play* randomness (the deal stays
    identical); without it they share one rng, so at N == baseline the mirror collapses to identical
    games (always a split). Turn it on to measure the irreducible variance at equal strength."""
    t0 = time.perf_counter()
    pair_scores = []
    evals: List[Dict] = []
    for d in range(deals):
        seed = base_seed + d

        # --- the two mirrored games (challenger in each seat), same deal ---
        winners, game_ms = {}, {}
        for cs in (0, 1):
            agents = [None, None]
            agents[cs] = MCTSAgent(iterations=n)
            agents[1 - cs] = MCTSAgent(iterations=baseline)
            play_rng = np.random.default_rng([seed, cs]) if independent_rng else None
            on_dec, game_ms[cs] = _knowledge_tracker() if (knowledge and collect_eval) else (None, None)
            winners[cs], _, _ = play_game(agents, np.random.default_rng(seed),
                                          on_decision=on_dec, play_rng=play_rng)
        challenger_wins = int(winners[0] == 0) + int(winners[1] == 1)
        pair_scores.append(challenger_wins / 2.0)

        if not collect_eval:
            continue

        # --- both bots evaluate the initial position (fixed rng -> baseline identical across N) ---
        init = GameState.deal(np.random.default_rng(seed))
        p_seat = init.to_play                      # the starting player = mover being evaluated
        info = init.information_set(p_seat)
        res_n = search(info, SearchConfig(rng=np.random.default_rng(seed), iterations=n))
        res_b = search(info, SearchConfig(rng=np.random.default_rng(seed), iterations=baseline))
        val_n, q1_n, q2_n, bm_n = _summarize(res_n)
        val_b, q1_b, q2_b, bm_b = _summarize(res_b)

        # calibration: check each bot in the game where IT played the starting seat p_seat.
        # @N is challenger; @N sits at p_seat in the game cs == p_seat, @baseline in cs == 1 - p_seat.
        n_start_won = int(winners[p_seat] == p_seat)
        b_start_won = int(winners[1 - p_seat] == p_seat)
        km = {c: -1 for c in _KNOW_COLS}
        if knowledge and collect_eval:
            for cs in (0, 1):
                for s in (0, 1):
                    for lvl in ("binary", "perfect"):
                        km[f"g{cs}_p{s}_first_{lvl}"] = game_ms[cs][(s, lvl)]
        evals.append({
            "n": n, "deal": d, "seed": seed, "start_seat": p_seat,
            "eval_n": val_n, "eval_baseline": val_b, "diff": val_n - val_b,
            "q1_n": q1_n, "q2_n": q2_n, "q1_baseline": q1_b, "q2_baseline": q2_b,
            "bestmove_agree": int(bm_n == bm_b),
            "pred_n_win": int(val_n > 0), "n_start_won": n_start_won,
            "correct_n": int((val_n > 0) == bool(n_start_won)),
            "pred_baseline_win": int(val_b > 0), "baseline_start_won": b_start_won,
            "correct_baseline": int((val_b > 0) == bool(b_start_won)),
            **km,
        })

    arr = np.array(pair_scores, dtype=float)
    p = float(arr.mean())
    ci95 = float(1.96 * arr.std(ddof=1) / math.sqrt(deals)) if deals > 1 else 0.0
    return {
        "n": n, "deals": deals, "games": 2 * deals,
        "wins": int(round(arr.sum() * 2)), "winrate": p, "ci95": ci95,
        "seconds": time.perf_counter() - t0, "evals": evals,
    }


def run_sweep(n_values: List[int], deals: int, baseline: int, base_seed: int,
              workers: int, collect_eval: bool = True, independent_rng: bool = False,
              knowledge: bool = False) -> List[Dict]:
    """Run every N-level (in parallel when ``workers > 1``) and return rows sorted by N."""
    from joblib import Parallel, delayed
    from tqdm import tqdm

    jobs = (delayed(_eval_level)(n, deals, baseline, base_seed, collect_eval, independent_rng, knowledge)
            for n in n_values)
    gen = Parallel(n_jobs=workers, return_as="generator")(jobs)
    rows = list(tqdm(gen, total=len(n_values), desc="N-levels", unit="level"))
    rows.sort(key=lambda r: r["n"])
    return rows


def flatten_evals(rows: List[Dict]) -> List[Dict]:
    return [e for r in rows for e in r.get("evals", [])]


# --- reporting -------------------------------------------------------------------------

def _mean(vals):
    return float(np.mean(vals)) if len(vals) else float("nan")


def format_table(rows: List[Dict], baseline: int) -> str:
    lines = [f"  {'N':>4}  {'win%':>6}  {'ci95%':>5}  {'wins':>8}  "
             f"{'evalN':>6}  {'accN%':>5}  {'agree%':>6}  {'secs':>6}"]
    for r in rows:
        evs = r.get("evals", [])
        en = _mean([e["eval_n"] for e in evs])
        acc = _mean([e["correct_n"] for e in evs]) * 100
        agr = _mean([e["bestmove_agree"] for e in evs]) * 100
        lines.append(f"  {r['n']:>4}  {r['winrate']*100:>5.1f}%  {r['ci95']*100:>4.1f}%  "
                     f"{r['wins']:>3}/{r['games']:<4}  {en:>6.3f}  {acc:>4.0f}%  {agr:>5.0f}%  "
                     f"{r['seconds']:>6.0f}")
    return "\n".join(lines)


def save_csv(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("n,deals,games,wins,winrate,ci95,seconds\n")
        for r in rows:
            fh.write(f"{r['n']},{r['deals']},{r['games']},{r['wins']},"
                     f"{r['winrate']},{r['ci95']},{r['seconds']}\n")


_EVAL_COLS = ["n", "deal", "seed", "start_seat", "eval_n", "eval_baseline", "diff",
              "q1_n", "q2_n", "q1_baseline", "q2_baseline", "bestmove_agree",
              "pred_n_win", "n_start_won", "correct_n",
              "pred_baseline_win", "baseline_start_won", "correct_baseline"] + _KNOW_COLS


def save_eval_csv(path: str, eval_rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(_EVAL_COLS) + "\n")
        for e in eval_rows:
            fh.write(",".join(str(e[c]) for c in _EVAL_COLS) + "\n")


def _by_n(eval_rows, key):
    d = defaultdict(list)
    for e in eval_rows:
        d[e["n"]].append(e[key] if isinstance(key, str) else key(e))
    return d


def _agg(eval_rows, keyfn):
    d = defaultdict(list)
    for e in eval_rows:
        d[e["n"]].append(keyfn(e))
    ns = sorted(d)
    means = [float(np.mean(d[n])) for n in ns]
    sems = [float(np.std(d[n], ddof=1) / math.sqrt(len(d[n]))) if len(d[n]) > 1 else 0.0 for n in ns]
    return ns, means, sems


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def save_plot(path: str, rows: List[Dict], baseline: int) -> bool:
    plt = _plt()
    if plt is None:
        return False
    xs = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(xs, [r["winrate"] * 100 for r in rows], yerr=[r["ci95"] * 100 for r in rows],
                marker="o", capsize=3, label=f"MCTS@N vs MCTS@{baseline}")
    ax.axhline(50, ls="--", color="gray", label=f"50% (== MCTS@{baseline})")
    ax.set_xlabel("MCTS simulations per decision (N)")
    ax.set_ylabel(f"win-rate vs MCTS@{baseline} (%)")
    ax.set_title("ImposterKings MCTS search-scaling (mirrored, paired seeds)")
    ax.set_ylim(0, 100); ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(path, dpi=120); plt.close(fig)
    return True


def save_eval_plot(path: str, eval_rows: List[Dict], baseline: int) -> bool:
    """Convergence of MCTS@N's starting evals to MCTS@baseline's: |root|, |top-1 Q|, |top-2 Q| diffs."""
    plt = _plt()
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, fn in [("|root value diff|", lambda e: abs(e["diff"])),
                      ("|top-1 Q diff|", lambda e: abs(e["q1_n"] - e["q1_baseline"])),
                      ("|top-2 Q diff|", lambda e: abs(e["q2_n"] - e["q2_baseline"]))]:
        ns, means, sems = _agg(eval_rows, fn)
        ax.errorbar(ns, means, yerr=sems, marker="o", capsize=3, label=label)
    ax.axhline(0, ls="--", color="gray")
    ax.set_xlabel("MCTS simulations per decision (N)")
    ax.set_ylabel(f"starting-eval difference vs MCTS@{baseline}")
    ax.set_title(f"Starting-eval convergence to MCTS@{baseline}")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(path, dpi=120); plt.close(fig)
    return True


def save_calibration_plot(path: str, eval_rows: List[Dict], baseline: int) -> bool:
    """Did the bots' verdicts come true? Prediction accuracy of @N and @baseline, and best-move agreement."""
    plt = _plt()
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, key in [(f"MCTS@N prediction accuracy", "correct_n"),
                       (f"MCTS@{baseline} prediction accuracy", "correct_baseline"),
                       ("best-move agreement (N vs baseline)", "bestmove_agree")]:
        ns, means, sems = _agg(eval_rows, lambda e, k=key: e[k])
        ax.errorbar(ns, [m * 100 for m in means], yerr=[s * 100 for s in sems],
                    marker="o", capsize=3, label=label)
    ax.axhline(50, ls="--", color="gray")
    ax.set_xlabel("MCTS simulations per decision (N)")
    ax.set_ylabel("rate (%)")
    ax.set_title("Outcome prediction & best-move agreement vs N")
    ax.set_ylim(0, 100); ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(path, dpi=120); plt.close(fig)
    return True


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="MCTS@N vs MCTS@baseline scaling + eval/calibration sweep.")
    p.add_argument("--min", type=int, default=25)
    p.add_argument("--max", type=int, default=500)
    p.add_argument("--step", type=int, default=25)
    p.add_argument("--deals", type=int, default=50,
                   help="distinct deals per level; each is played twice (mirrored) -> 2x games")
    p.add_argument("--baseline", type=int, default=500)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-eval", action="store_true", help="skip eval + calibration collection")
    p.add_argument("--independent-rng", action="store_true",
                   help="give the two mirrored games independent play randomness (same deal)")
    p.add_argument("--knowledge", action="store_true",
                   help="record the first ply each seat reaches binary/perfect hand-knowledge (per game)")
    args = p.parse_args(argv)

    n_values = list(range(args.min, args.max + 1, args.step))
    print(f"sweep N={n_values}\n  baseline=MCTS@{args.baseline} | {args.deals} deals x2 mirrored = "
          f"{args.deals*2} games/level | eval={'off' if args.no_eval else 'on'} | "
          f"rng={'independent' if args.independent_rng else 'shared'} | "
          f"workers={args.workers} | base_seed={args.base_seed} | {len(n_values)*args.deals*2} games total")

    t0 = time.perf_counter()
    rows = run_sweep(n_values, args.deals, args.baseline, args.base_seed, args.workers,
                     collect_eval=not args.no_eval, independent_rng=args.independent_rng,
                     knowledge=args.knowledge)
    print(f"\n{format_table(rows, args.baseline)}")
    print(f"\ntotal wall time: {time.perf_counter() - t0:.0f}s")

    os.makedirs(args.out_dir, exist_ok=True)
    outputs = [os.path.join(args.out_dir, "search_scaling.csv")]
    save_csv(outputs[0], rows)

    eval_rows = flatten_evals(rows)
    if eval_rows:
        eval_csv = os.path.join(args.out_dir, "eval_scaling.csv")
        save_eval_csv(eval_csv, eval_rows)
        outputs.append(eval_csv)

    if not args.no_plot:
        if save_plot(os.path.join(args.out_dir, "search_scaling.png"), rows, args.baseline):
            outputs.append(os.path.join(args.out_dir, "search_scaling.png"))
        if eval_rows:
            for fn, name in [(save_eval_plot, "eval_scaling.png"),
                             (save_calibration_plot, "calibration.png")]:
                path = os.path.join(args.out_dir, name)
                if fn(path, eval_rows, args.baseline):
                    outputs.append(path)
    print("saved " + ", ".join(outputs))


if __name__ == "__main__":
    main()
