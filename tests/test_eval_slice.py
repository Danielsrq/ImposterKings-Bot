"""Reshaping the raw eval CSV: per-seed slice, per-game rows, and cross-N sweep summary."""
from __future__ import annotations

import csv

from imposterkings.analysis.eval_slice import (
    OUT_COLS, PG_COLS, per_game_rows, slice_rows, sweep_summary, write_csv,
)
from imposterkings.analysis.search_scaling import _EVAL_COLS


def _rec(n, seed, *, nsw, bsw, agree=1, correct_n=1, correct_b=1):
    return {"n": n, "deal": seed, "seed": seed, "start_seat": 0,
            "eval_n": 0.1, "eval_baseline": 0.2, "diff": -0.1,
            "q1_n": 0.3, "q2_n": 0.1, "q1_baseline": 0.4, "q2_baseline": 0.2,
            "bestmove_agree": agree, "pred_n_win": 1, "n_start_won": nsw, "correct_n": correct_n,
            "pred_baseline_win": 1, "baseline_start_won": bsw, "correct_baseline": correct_b}


def _write(path, records):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_EVAL_COLS)
        w.writeheader()
        for r in records:
            w.writerow(r)


def test_slice_pair_pergame_and_sweeps(tmp_path):
    src = tmp_path / "eval.csv"
    _write(src, [
        _rec(100, 0, nsw=0, bsw=1, agree=1, correct_n=0),  # @500 swept (N lost both)
        _rec(100, 1, nsw=1, bsw=0, agree=0),               # @N swept (upset)
        _rec(200, 0, nsw=1, bsw=1),                        # split (starter won both)
    ])

    rows = slice_rows(str(src), 100)
    assert [r["seed"] for r in rows] == [0, 1]             # only N=100, sorted by seed
    assert rows[0]["pair"] == "500" and rows[1]["pair"] == "N"
    assert rows[0]["accuracy_n"] == "F" and rows[0]["accuracy_500"] == "T"
    out = tmp_path / "slice.csv"
    write_csv(str(out), rows, OUT_COLS)
    assert out.read_text().splitlines()[0] == ",".join(OUT_COLS)

    pg = per_game_rows(str(src), 100, 500)
    assert len(pg) == 4                                    # 2 seeds x 2 games
    assert pg[0]["seed"] == 0 and pg[0]["starter"] == "@100"   # N-start row first
    assert pg[1]["seed"] == 0 and pg[1]["starter"] == "@500"
    assert list(pg[0]) == PG_COLS

    sw = {r["n"]: r for r in sweep_summary(str(src))}
    assert sw[100]["sweep_500"] == 1 and sw[100]["sweep_N"] == 1 and sw[100]["split"] == 0
    assert sw[200]["split"] == 1
