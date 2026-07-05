"""Slice/reshape the raw per-deal eval data (eval_scaling.csv) for readable analysis.

    python -m imposterkings.analysis.eval_slice --n 200        # per-seed table for N=200
    python -m imposterkings.analysis.eval_slice --n 200 --per-game  # 2 rows/seed (one per mirrored game)
    python -m imposterkings.analysis.eval_slice --sweeps       # who swept each deal, across all N

Each seed is a mirrored pair (2 games: challenger in each seat). Encoding both games in one row is
confusing, so ``--per-game`` emits one row per game with that game's starter, its own eval/prediction,
and its result. The ``pair`` column (and ``--sweeps``) classifies the mirrored pair:
``500`` = MCTS@500 won both games, ``N`` = MCTS@N won both (an upset), ``split`` = one each (the deal,
not skill, decided it -- the same *seat* won both games). "top1/top2" are top-1/top-2 root-move Q-values.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List

OUT_COLS = ["seed", "eval_n", "eval_500", "accuracy_n", "accuracy_500", "agree", "pair",
            "N_won_as_start", "500_won_as_start",
            "top1_by_n", "top2_by_n", "top1_by_500", "top2_by_500"]
PG_COLS = ["seed", "starter", "opponent", "starter_eval", "pred_win", "won", "correct", "top1", "top2"]


def _tf(cond) -> str:
    return "T" if cond else "F"


def _pair(n_start_won: int, baseline_start_won: int) -> str:
    """Mirror-pair result: 'N' (@N swept), '500' (@500 swept), or 'split' (deal-decided)."""
    n_pair_wins = n_start_won + (1 - baseline_start_won)   # @N's wins across the 2 games: 0, 1 or 2
    return {2: "N", 0: "500"}.get(n_pair_wins, "split")


def _read(in_path: str, n: int):
    with open(in_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if int(r["n"]) == n:
                yield r


def slice_rows(in_path: str, n: int) -> List[Dict]:
    """One row per seed for the given N (both games encoded via the pair/win columns)."""
    rows: List[Dict] = []
    for r in _read(in_path, n):
        nsw, bsw = int(r["n_start_won"]), int(r["baseline_start_won"])
        rows.append({
            "seed": int(r["seed"]),
            "eval_n": float(r["eval_n"]), "eval_500": float(r["eval_baseline"]),
            "accuracy_n": _tf(int(r["correct_n"])), "accuracy_500": _tf(int(r["correct_baseline"])),
            "agree": _tf(int(r["bestmove_agree"])), "pair": _pair(nsw, bsw),
            "N_won_as_start": _tf(nsw), "500_won_as_start": _tf(bsw),
            "top1_by_n": float(r["q1_n"]), "top2_by_n": float(r["q2_n"]),
            "top1_by_500": float(r["q1_baseline"]), "top2_by_500": float(r["q2_baseline"]),
        })
    rows.sort(key=lambda x: x["seed"])
    return rows


def per_game_rows(in_path: str, n: int, baseline: int) -> List[Dict]:
    """Two rows per seed -- one per mirrored game -- from each game's starter's perspective."""
    rows: List[Dict] = []
    for r in _read(in_path, n):
        seed = int(r["seed"])
        # game where @N started
        rows.append({"seed": seed, "starter": f"@{n}", "opponent": f"@{baseline}",
                     "starter_eval": float(r["eval_n"]), "pred_win": _tf(float(r["eval_n"]) > 0),
                     "won": _tf(int(r["n_start_won"])), "correct": _tf(int(r["correct_n"])),
                     "top1": float(r["q1_n"]), "top2": float(r["q2_n"])})
        # game where @baseline started
        rows.append({"seed": seed, "starter": f"@{baseline}", "opponent": f"@{n}",
                     "starter_eval": float(r["eval_baseline"]), "pred_win": _tf(float(r["eval_baseline"]) > 0),
                     "won": _tf(int(r["baseline_start_won"])), "correct": _tf(int(r["correct_baseline"])),
                     "top1": float(r["q1_baseline"]), "top2": float(r["q2_baseline"])})
    rows.sort(key=lambda x: (x["seed"], x["starter"] != f"@{n}"))  # N-start row first
    return rows


def sweep_summary(in_path: str) -> List[Dict]:
    """Per N: counts of @500-sweeps, @N-sweeps, and splits over all deals."""
    by_n: Dict[int, Dict] = {}
    with open(in_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            n = int(r["n"])
            d = by_n.setdefault(n, {"n": n, "sweep_500": 0, "sweep_N": 0, "split": 0, "deals": 0})
            d["deals"] += 1
            p = _pair(int(r["n_start_won"]), int(r["baseline_start_won"]))
            d["sweep_500" if p == "500" else "sweep_N" if p == "N" else "split"] += 1
    return [by_n[n] for n in sorted(by_n)]


def write_csv(path: str, rows: List[Dict], cols: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def format_table(rows: List[Dict], n: int, top: int = 0) -> str:
    shown = rows if not top else rows[:top]
    head = (f"  {'seed':>4}  {'eval_n':>7}  {'eval_500':>8}  {'accN':>4}  {'acc500':>6}  {'agr':>3}  "
            f"{'pair':>5}  {'t1_n':>6}  {'t2_n':>6}  {'t1_500':>6}  {'t2_500':>6}")
    lines = [f"N = {n}  ({len(rows)} seeds)", head]
    for r in shown:
        lines.append(f"  {r['seed']:>4}  {r['eval_n']:>7.3f}  {r['eval_500']:>8.3f}  "
                     f"{r['accuracy_n']:>4}  {r['accuracy_500']:>6}  {r['agree']:>3}  {r['pair']:>5}  "
                     f"{r['top1_by_n']:>6.3f}  {r['top2_by_n']:>6.3f}  "
                     f"{r['top1_by_500']:>6.3f}  {r['top2_by_500']:>6.3f}")
    if top and len(rows) > top:
        lines.append(f"  ... ({len(rows) - top} more)")
    s500 = sum(r["pair"] == "500" for r in rows)
    sN = sum(r["pair"] == "N" for r in rows)
    lines.append(f"  sweeps: @500 won both = {s500}   @N won both = {sN}   split (deal-decided) = "
                 f"{len(rows) - s500 - sN}")
    return "\n".join(lines)


def format_per_game(rows: List[Dict], n: int, top: int = 0) -> str:
    shown = rows if not top else rows[:top]
    head = (f"  {'seed':>4}  {'starter':>7}  {'opp':>7}  {'eval':>6}  {'pred':>4}  {'won':>3}  "
            f"{'ok':>2}  {'top1':>6}  {'top2':>6}")
    lines = [f"N = {n}  ({len(rows)//2} seeds x 2 games)", head]
    for r in shown:
        lines.append(f"  {r['seed']:>4}  {r['starter']:>7}  {r['opponent']:>7}  {r['starter_eval']:>6.3f}  "
                     f"{r['pred_win']:>4}  {r['won']:>3}  {r['correct']:>2}  {r['top1']:>6.3f}  {r['top2']:>6.3f}")
    if top and len(rows) > top:
        lines.append(f"  ... ({len(rows) - top} more)")
    return "\n".join(lines)


def format_sweeps(rows: List[Dict]) -> str:
    lines = ["cross-N mirror-pair outcomes (per deal, one bot winning BOTH seats = a skill sweep)",
             f"  {'N':>4}  {'@500 sweeps':>11}  {'@N sweeps':>9}  {'splits':>6}  {'deals':>5}"]
    for r in rows:
        lines.append(f"  {r['n']:>4}  {r['sweep_500']:>11}  {r['sweep_N']:>9}  {r['split']:>6}  {r['deals']:>5}")
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Slice/reshape eval_scaling.csv for analysis.")
    p.add_argument("--n", type=int, default=None, help="N level to slice (required unless --sweeps)")
    p.add_argument("--in", dest="in_path", default="results/eval_scaling.csv")
    p.add_argument("--out", default=None)
    p.add_argument("--baseline", type=int, default=500, help="baseline N (for --per-game labels)")
    p.add_argument("--per-game", action="store_true", help="two rows per seed, one per mirrored game")
    p.add_argument("--sweeps", action="store_true", help="cross-N sweep summary (ignores --n)")
    p.add_argument("--print", dest="show", type=int, default=25, help="rows to print (0 = all)")
    args = p.parse_args(argv)
    d = os.path.dirname(args.in_path) or "."

    if args.sweeps:
        rows = sweep_summary(args.in_path)
        out = args.out or os.path.join(d, "eval_sweeps.csv")
        write_csv(out, rows, ["n", "sweep_500", "sweep_N", "split", "deals"])
        print(format_sweeps(rows))
        print(f"\nsaved {out}")
        return

    if args.n is None:
        raise SystemExit("--n is required (unless --sweeps)")
    if args.per_game:
        rows = per_game_rows(args.in_path, args.n, args.baseline)
        if not rows:
            raise SystemExit(f"no rows for N={args.n} in {args.in_path}")
        out = args.out or os.path.join(d, f"eval_pergame_N{args.n}.csv")
        write_csv(out, rows, PG_COLS)
        print(format_per_game(rows, args.n, top=args.show))
    else:
        rows = slice_rows(args.in_path, args.n)
        if not rows:
            raise SystemExit(f"no rows for N={args.n} in {args.in_path}")
        out = args.out or os.path.join(d, f"eval_slice_N{args.n}.csv")
        write_csv(out, rows, OUT_COLS)
        print(format_table(rows, args.n, top=args.show))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
