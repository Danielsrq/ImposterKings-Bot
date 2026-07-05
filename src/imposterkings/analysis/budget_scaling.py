"""Budget-scaling study: sweep a hybrid MCTS@k,l challenger against TWO fixed reference bots.

    python -m imposterkings.analysis.budget_scaling        # k=10..50 vs {fixed500, hybrid-k50}, 50 deals
    python -m imposterkings.analysis.budget_scaling --deals 50 --workers 5

Where ``search_scaling`` sweeps an integer-N challenger against one fixed-N baseline, this sweeps a
per-decision **budget** challenger -- ``hybrid(k, l)`` for ``k in {10,20,30,40,50}`` (see
:mod:`imposterkings.budget`) -- against a *list* of fixed reference opponents, by default:

  * ``fixed(500)``    -- the classic MCTS@500, and
  * ``hybrid(50, 3)`` -- the strongest point of the sweep itself (so k=50 vs this is self-play: the
                         deal-variance floor).

For every (challenger-k, baseline) pair and every deal we collect, with paired seeds + mirrored seating:

  1. **Win-rate** of MCTS@k vs each baseline (challenger sits in both seats on the same deal).
  2. **Per-deal outcome** -> the *seed split/sweep* analysis. In a mirror, challenger_wins in {0,1,2}:
     ``1`` == a **split** (each side won its seating -> the SEAT/DEAL decided the game, not skill);
     ``0``/``2`` == a **sweep** (one bot won regardless of seat -> skill decided it). The split-rate is
     the fraction of games "decided by the dealt cards". (Needs independent play RNG -- on by default --
     else the identical-strength mirror collapses to a trivial split.)
  3. **Starting-position evals** (root value + top-1 / top-2 root-move Q) by the challenger@k and by BOTH
     baselines, from the mover's perspective on the dealt position -- so you can watch @k's opening
     verdict approach the references as k grows.
  4. **Compute & shape** per game: plies, decisions, mean branching, and *actual MCTS iterations spent*
     by each seat -- so win-rate can be read against real compute, and game-length / branching numbers
     are recomputed on the current (bug-fixed) engine for NN dataset sizing.

Embarrassingly parallel by matchup (one joblib task per challenger x baseline) and by eval-spec.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from ..agents import MCTSAgent
from ..arena import play_game
from ..budget import branching, hybrid
from ..mcts import SearchConfig, search
from ..state import GameState
from .search_scaling import _mean, _plt, _summarize

Spec = Tuple  # ("fixed", n) | ("hybrid", k, l) | ("branching", k, l)


# --- spec -> agent / label / root-budget ----------------------------------------------

def _budget_of(spec: Spec):
    if spec[0] == "hybrid":
        return hybrid(spec[1], spec[2])
    if spec[0] == "branching":
        return branching(spec[1], spec[2])
    return None


def make_agent(spec: Spec, *, evaluate_forced: bool = False) -> MCTSAgent:
    if spec[0] == "fixed":
        return MCTSAgent(iterations=spec[1], evaluate_forced=evaluate_forced)
    return MCTSAgent(budget=_budget_of(spec), evaluate_forced=evaluate_forced)


def spec_label(spec: Spec) -> str:
    return f"fixed{spec[1]}" if spec[0] == "fixed" else f"{spec[0]}-k{spec[1]}-l{spec[2]}"


def _root_iters(spec: Spec, view, moves) -> int:
    if spec[0] == "fixed":
        return spec[1]
    return _budget_of(spec)(view, moves)


# --- per-game shape / compute accounting ----------------------------------------------

def _gs() -> Dict[str, int]:
    return {"plies": 0, "decisions": 0, "legal_sum": 0, "iters": 0}


def _cost_hook(stats: Dict[int, Dict[str, int]]):
    """``on_decision`` hook: per seat count plies, real decisions (>1 legal), branching, iters spent."""
    def f(seat, view, move, agent, state):
        s = stats[seat]
        s["plies"] += 1
        moves = view.legal_moves()
        n = len(moves)
        if n > 1:                                   # a genuine decision (else forced -> no search)
            s["decisions"] += 1
            s["legal_sum"] += n
            s["iters"] += agent.budget(view, moves) if agent.budget is not None else agent.iterations
    return f


# --- work unit: a CHUNK of mirrored deals -------------------------------------------
# The schedulable job is ~`chunk` deals (each 2 mirrored games) for one challenger x baseline,
# not a whole matchup -- so the heavy hybrid-k50 work spreads across workers instead of pinning
# one core for the full matchup -- while a chunk of ~10 still amortizes joblib's per-task overhead.

def _one_deal(challenger: Spec, baseline: Spec, seed: int, deal: int,
              independent_rng: bool) -> Dict:
    cwins = 0
    g_plies, g_dec, g_legal, ch_iters, bl_iters = [], [], [], [], []
    for cs in (0, 1):                               # challenger seat = cs, mirrored
        agents = [None, None]
        agents[cs] = make_agent(challenger)
        agents[1 - cs] = make_agent(baseline)
        stats = {0: _gs(), 1: _gs()}
        play_rng = np.random.default_rng([seed, cs]) if independent_rng else None
        winner, _, _ = play_game(agents, np.random.default_rng(seed),
                                 on_decision=_cost_hook(stats), play_rng=play_rng)
        cwins += int(winner == cs)
        g_plies.append(stats[0]["plies"] + stats[1]["plies"])
        g_dec.append(stats[0]["decisions"] + stats[1]["decisions"])
        g_legal.append(stats[0]["legal_sum"] + stats[1]["legal_sum"])
        ch_iters.append(stats[cs]["iters"])
        bl_iters.append(stats[1 - cs]["iters"])
    outcome = "split" if cwins == 1 else ("sweep_challenger" if cwins == 2 else "sweep_baseline")
    return {
        "challenger": spec_label(challenger), "baseline": spec_label(baseline), "k": challenger[1],
        "seed": seed, "deal": deal, "challenger_wins": cwins, "outcome": outcome,
        "split": int(cwins == 1), "pair_score": cwins / 2.0,
        "plies": _mean(g_plies), "decisions": _mean(g_dec),
        "branching": (sum(g_legal) / sum(g_dec)) if sum(g_dec) else float("nan"),
        "iters_challenger": _mean(ch_iters), "iters_baseline": _mean(bl_iters),
    }


def _chunk_task(challenger: Spec, baseline: Spec, deals: List[int], base_seed: int,
                independent_rng: bool) -> Dict:
    """Run one chunk of deals for a matchup; return its deal-rows + the compute-seconds it spent."""
    t0 = time.perf_counter()
    rows = [_one_deal(challenger, baseline, base_seed + d, d, independent_rng) for d in deals]
    return {"deal_rows": rows, "seconds": time.perf_counter() - t0}


def _aggregate_matchup(challenger: Spec, baseline: Spec, deal_rows: List[Dict],
                       seconds: float) -> Dict:
    deal_rows = sorted(deal_rows, key=lambda r: r["deal"])
    arr = np.array([r["pair_score"] for r in deal_rows], dtype=float)
    n = len(deal_rows)
    ci95 = float(1.96 * arr.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return {
        "challenger": spec_label(challenger), "baseline": spec_label(baseline),
        "k": challenger[1], "l": challenger[2], "deals": n, "games": 2 * n,
        "wins": int(round(arr.sum() * 2)), "winrate": float(arr.mean()), "ci95": ci95,
        "splits": sum(r["split"] for r in deal_rows),
        "plies": _mean([r["plies"] for r in deal_rows]),
        "decisions": _mean([r["decisions"] for r in deal_rows]),
        "branching": _mean([r["branching"] for r in deal_rows]),
        "iters_challenger": _mean([r["iters_challenger"] for r in deal_rows]),
        "iters_baseline": _mean([r["iters_baseline"] for r in deal_rows]),
        "seconds": seconds, "deal_rows": deal_rows,
    }


# --- starting-position evals: one search per (deal, spec), also chunked ---------------

def _eval_chunk(spec: Spec, deals: List[int], base_seed: int) -> List[Dict]:
    rows: List[Dict] = []
    label = spec_label(spec)
    for d in deals:
        seed = base_seed + d
        init = GameState.deal(np.random.default_rng(seed))
        p_seat = init.to_play
        info = init.information_set(p_seat)
        iters = _root_iters(spec, info, info.legal_moves())
        res = search(info, SearchConfig(rng=np.random.default_rng(seed), iterations=iters))
        val, q1, q2, _ = _summarize(res)
        rows.append({"spec": label, "k": spec[1], "deal": d, "seed": seed, "start_seat": p_seat,
                     "iters": iters, "root": val, "q1": q1, "q2": q2})
    return rows


# --- driver ---------------------------------------------------------------------------

def _chunks(deals: int, size: int) -> List[List[int]]:
    return [list(range(i, min(i + size, deals))) for i in range(0, deals, size)]


def run_study(k_values: List[int], l: int, baselines: List[Spec], deals: int, base_seed: int,
              workers: int, independent_rng: bool, collect_eval: bool, chunk: int = 10) -> Dict:
    from joblib import Parallel, delayed
    from tqdm import tqdm

    challengers = [("hybrid", k, l) for k in k_values]
    matchups = [(c, b) for b in baselines for c in challengers]
    deal_chunks = _chunks(deals, chunk)

    # Flatten (matchup x deal-chunk) into one balanced job list; a chunk-size job is the unit of work.
    jobs, keys = [], []
    for c, b in matchups:
        for ch in deal_chunks:
            jobs.append(delayed(_chunk_task)(c, b, ch, base_seed, independent_rng))
            keys.append((spec_label(c), spec_label(b)))
    tm = time.perf_counter()
    results = list(tqdm(Parallel(n_jobs=workers, return_as="generator")(jobs),
                        total=len(jobs), desc="chunks", unit="chunk"))
    matchup_wall = time.perf_counter() - tm

    grouped: Dict = defaultdict(lambda: {"deal_rows": [], "seconds": 0.0})
    for key, res in zip(keys, results):
        grouped[key]["deal_rows"].extend(res["deal_rows"])
        grouped[key]["seconds"] += res["seconds"]
    rows = [_aggregate_matchup(c, b, grouped[(spec_label(c), spec_label(b))]["deal_rows"],
                               grouped[(spec_label(c), spec_label(b))]["seconds"])
            for c, b in matchups]
    rows.sort(key=lambda r: (r["baseline"], r["k"]))

    eval_rows: List[Dict] = []
    eval_wall = 0.0
    if collect_eval:
        specs = challengers + baselines
        ejobs = [delayed(_eval_chunk)(s, ch, base_seed) for s in specs for ch in deal_chunks]
        te = time.perf_counter()
        for part in tqdm(Parallel(n_jobs=workers, return_as="generator")(ejobs),
                         total=len(ejobs), desc="evals", unit="chunk"):
            eval_rows.extend(part)
        eval_wall = time.perf_counter() - te
    return {"rows": rows, "eval_rows": eval_rows, "baselines": [spec_label(b) for b in baselines],
            "timing": {"matchup_wall": matchup_wall, "eval_wall": eval_wall, "workers": workers}}


def _agg_by(rows: List[Dict], xkey: str, valfn):
    """Mean +/- SEM of ``valfn`` grouped by ``rows[xkey]`` (x sorted). Local so x can be 'k', not 'n'."""
    d = defaultdict(list)
    for e in rows:
        d[e[xkey]].append(valfn(e))
    xs = sorted(d)
    means = [float(np.mean(d[x])) for x in xs]
    sems = [float(np.std(d[x], ddof=1) / math.sqrt(len(d[x]))) if len(d[x]) > 1 else 0.0 for x in xs]
    return xs, means, sems


# --- reporting ------------------------------------------------------------------------

def format_table(rows: List[Dict]) -> str:
    hdr = (f"  {'challenger':>16}  {'vs baseline':>16}  {'win%':>6}  {'ci95%':>5}  {'wins':>8}  "
           f"{'split%':>6}  {'plies':>5}  {'branch':>6}  {'itC':>6}  {'itB':>6}  {'secs':>5}")
    lines = [hdr]
    for r in rows:
        lines.append(
            f"  {r['challenger']:>16}  {r['baseline']:>16}  {r['winrate']*100:>5.1f}%  "
            f"{r['ci95']*100:>4.1f}%  {r['wins']:>3}/{r['games']:<4}  "
            f"{r['splits']/r['deals']*100:>5.0f}%  {r['plies']:>5.1f}  {r['branching']:>6.2f}  "
            f"{r['iters_challenger']:>6.0f}  {r['iters_baseline']:>6.0f}  {r['seconds']:>5.0f}")
    return "\n".join(lines)


def _write_csv(path: str, cols: List[str], rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")


_WIN_COLS = ["challenger", "baseline", "k", "l", "deals", "games", "wins", "winrate", "ci95",
             "splits", "plies", "decisions", "branching", "iters_challenger", "iters_baseline",
             "seconds"]
_DEAL_COLS = ["challenger", "baseline", "k", "seed", "deal", "challenger_wins", "outcome", "split",
              "plies", "decisions", "branching", "iters_challenger", "iters_baseline"]
_EVAL_COLS = ["spec", "k", "deal", "seed", "start_seat", "iters", "root", "q1", "q2"]


def save_all(out_dir: str, study: Dict, no_plot: bool) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    rows, eval_rows = study["rows"], study["eval_rows"]
    deal_rows = [dr for r in rows for dr in r["deal_rows"]]
    out = []
    for name, cols, data in [("winrate.csv", _WIN_COLS, rows),
                             ("deal_outcomes.csv", _DEAL_COLS, deal_rows),
                             ("evals.csv", _EVAL_COLS, eval_rows)]:
        if data:
            p = os.path.join(out_dir, name)
            _write_csv(p, cols, data)
            out.append(p)
    if not no_plot:
        out += save_plots(out_dir, study)
    return out


def save_plots(out_dir: str, study: Dict) -> List[str]:
    plt = _plt()
    if plt is None:
        return []
    rows, eval_rows, baselines = study["rows"], study["eval_rows"], study["baselines"]
    out = []

    # (1) win-rate vs k, one curve per baseline; (2) split-rate (deal-decided) vs k.
    fig, (axw, axs) = plt.subplots(1, 2, figsize=(12, 4.5))
    for bl in baselines:
        br = [r for r in rows if r["baseline"] == bl]
        ks = [r["k"] for r in br]
        axw.errorbar(ks, [r["winrate"] * 100 for r in br], yerr=[r["ci95"] * 100 for r in br],
                     marker="o", capsize=3, label=f"vs {bl}")
        axs.plot(ks, [r["splits"] / r["deals"] * 100 for r in br], marker="s", label=f"vs {bl}")
    axw.axhline(50, ls="--", color="gray"); axw.set_ylim(0, 100)
    axw.set_xlabel("challenger k (hybrid, l=3)"); axw.set_ylabel("win-rate (%)")
    axw.set_title("MCTS@k win-rate vs references"); axw.legend(); axw.grid(alpha=0.3)
    axs.set_ylim(0, 100); axs.set_xlabel("challenger k"); axs.set_ylabel("split-rate (%)")
    axs.set_title("Games decided by the deal (mirror split)"); axs.legend(); axs.grid(alpha=0.3)
    fig.tight_layout(); p = os.path.join(out_dir, "winrate_split.png"); fig.savefig(p, dpi=120)
    plt.close(fig); out.append(p)

    # (3) starting-eval top-1 / top-2 vs k, with baseline reference lines.
    if eval_rows:
        ch = [e for e in eval_rows if e["spec"].startswith("hybrid")]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for key, lab in [("q1", "top-1 root Q"), ("q2", "top-2 root Q")]:
            ns, means, sems = _agg_by(ch, "k", lambda e, k=key: e[k])
            ax.errorbar(ns, means, yerr=sems, marker="o", capsize=3, label=f"@k {lab}")
        for bl in baselines:
            be = [e for e in eval_rows if e["spec"] == bl]
            if be:
                ax.axhline(_mean([e["q1"] for e in be]), ls="--", alpha=0.6, label=f"{bl} top-1")
        ax.set_xlabel("challenger k"); ax.set_ylabel("starting-move eval (Q)")
        ax.set_title("Opening top-1/top-2 eval vs k"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); p = os.path.join(out_dir, "opening_eval.png"); fig.savefig(p, dpi=120)
        plt.close(fig); out.append(p)
    return out


def print_seed_analysis(study: Dict) -> None:
    """Per-baseline split/sweep tallies, and the seeds that are deal-locked across the whole k-sweep."""
    rows = study["rows"]
    print("\nseed split/sweep analysis (split == decided by the deal/seat, not skill):")
    for bl in study["baselines"]:
        br = [r for r in rows if r["baseline"] == bl]
        by_seed = defaultdict(list)
        for r in br:
            for dr in r["deal_rows"]:
                by_seed[dr["seed"]].append(dr["challenger_wins"])
        n_seeds = len(by_seed)
        always_split = [s for s, v in by_seed.items() if all(x == 1 for x in v)]
        always_sweep = [s for s, v in by_seed.items() if all(x in (0, 2) for x in v)]
        overall = _mean([r["splits"] / r["deals"] for r in br]) * 100
        print(f"  vs {bl:>16}: mean split-rate {overall:4.0f}%  |  "
              f"deal-locked seeds (split at every k): {len(always_split)}/{n_seeds}  |  "
              f"skill-locked (sweep at every k): {len(always_sweep)}/{n_seeds}")
        if always_split:
            print(f"      always-split seeds: {sorted(always_split)[:20]}")


def print_timing(study: Dict) -> None:
    """Wall vs 1-core-equivalent compute, realized speedup, and per-game cost for extrapolation.

    ``seconds`` per matchup is compute-time (sum of its chunk-task durations); summed it is the
    single-core equivalent, so ``compute / matchup_wall`` is the speedup actually achieved on the
    worker pool, and ``compute / games`` is the per-game cost to project a larger sweep."""
    rows, t = study["rows"], study.get("timing", {})
    compute = sum(r["seconds"] for r in rows)
    games = sum(r["games"] for r in rows)
    deals = rows[0]["deals"] if rows else 0
    mwall, ewall = t.get("matchup_wall", 0.0), t.get("eval_wall", 0.0)
    workers = t.get("workers", 1)
    lines = ["\ntiming:",
             f"  games:           {games}  ({compute / games:.2f}s/game, single-core)",
             f"  matchup compute: {compute:>6.0f}s  (single-core equivalent)"]
    if mwall > 0:
        lines.append(f"  matchup wall:    {mwall:>6.0f}s  "
                     f"(speedup {compute / mwall:.1f}x on {workers} workers)")
    if ewall > 0:
        lines.append(f"  eval wall:       {ewall:>6.0f}s")
    lines.append(f"  total wall:      {mwall + ewall:>6.0f}s")
    if rows:
        slow = max(rows, key=lambda r: r["seconds"] / r["games"])
        lines.append(f"  heaviest matchup: {slow['challenger']} vs {slow['baseline']} "
                     f"@ {slow['seconds'] / slow['games']:.2f}s/game")
    if mwall > 0 and deals:
        # compute grows ~linearly with deals; speedup ~constant -> wall scales ~linearly too.
        lines.append(f"  projection: ~{mwall / deals:.1f}s matchup wall per deal "
                     f"(at these k / baselines / {workers} workers)")
    print("\n".join(lines))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Hybrid MCTS@k,l sweep vs fixed references + seed analysis.")
    p.add_argument("--k", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    p.add_argument("--l", type=int, default=3)
    p.add_argument("--deals", type=int, default=50,
                   help="distinct deals per matchup; each mirrored -> 2x games (default 50 -> 100 games)")
    p.add_argument("--fixed-baseline", type=int, default=500)
    p.add_argument("--hybrid-baseline", type=int, default=50, help="k of the hybrid reference baseline")
    p.add_argument("--baselines", choices=["both", "fixed", "hybrid"], default="both",
                   help="which reference opponents to run against (default both)")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--chunk", type=int, default=10,
                   help="deals per parallel job (default 10 = 20 mirrored games/job)")
    p.add_argument("--out-dir", default=os.path.join("results", "budget_scaling"))
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--shared-rng", action="store_true",
                   help="share play RNG across the mirror (default: independent, needed for split rates)")
    args = p.parse_args(argv)

    all_baselines = {"fixed": ("fixed", args.fixed_baseline),
                     "hybrid": ("hybrid", args.hybrid_baseline, args.l)}
    keys = ["fixed", "hybrid"] if args.baselines == "both" else [args.baselines]
    baselines: List[Spec] = [all_baselines[k] for k in keys]
    n_match = len(args.k) * len(baselines)
    n_jobs = n_match * len(_chunks(args.deals, args.chunk))
    print(f"budget sweep  challenger=hybrid-k{args.k}-l{args.l}\n"
          f"  baselines={[spec_label(b) for b in baselines]} | {args.deals} deals x2 mirrored = "
          f"{args.deals*2} games/matchup | {n_match} matchups = {n_match*args.deals*2} games | "
          f"chunk={args.chunk} -> {n_jobs} jobs | eval={'off' if args.no_eval else 'on'} | "
          f"rng={'shared' if args.shared_rng else 'independent'} | workers={args.workers}")

    study = run_study(args.k, args.l, baselines, args.deals, args.base_seed, args.workers,
                      independent_rng=not args.shared_rng, collect_eval=not args.no_eval,
                      chunk=args.chunk)
    print(f"\n{format_table(study['rows'])}")
    print_seed_analysis(study)
    print_timing(study)

    outputs = save_all(args.out_dir, study, args.no_plot)
    print("saved " + ", ".join(outputs))


if __name__ == "__main__":
    main()
