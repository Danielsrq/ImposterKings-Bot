"""Fast, in-process checks of the search-scaling sweep (skipped if joblib/matplotlib are absent)."""
from __future__ import annotations

import pytest

pytest.importorskip("joblib")

from imposterkings.data_analysis.search_scaling import (  # noqa: E402
    _EVAL_COLS, _KNOW_COLS, flatten_evals, run_sweep, save_calibration_plot, save_csv, save_eval_csv,
    save_eval_plot, save_plot,
)


def test_run_sweep_in_process_shapes_and_csv(tmp_path):
    # Tiny budgets + 2 deals so this stays sub-second; workers=1 -> in-process (no spawn).
    rows = run_sweep(n_values=[8, 16], deals=2, baseline=8, base_seed=0, workers=1)
    assert [r["n"] for r in rows] == [8, 16]  # returned sorted by N
    for r in rows:
        assert r["deals"] == 2 and r["games"] == 4  # each deal is mirrored into 2 games
        assert 0 <= r["wins"] <= r["games"]
        assert 0.0 <= r["winrate"] <= 1.0
        assert r["ci95"] >= 0.0

    csv_path = tmp_path / "scaling.csv"
    save_csv(str(csv_path), rows)
    assert csv_path.read_text().splitlines()[0] == "n,deals,games,wins,winrate,ci95,seconds"


def test_eval_and_calibration_fields(tmp_path):
    rows = run_sweep(n_values=[8, 16], deals=2, baseline=8, base_seed=0, workers=1)
    evals = flatten_evals(rows)
    assert len(evals) == 2 * 2  # 2 levels x 2 deals
    for e in evals:
        assert set(_EVAL_COLS) <= set(e)                       # every column present
        assert -1.0 <= e["eval_n"] <= 1.0 and -1.0 <= e["eval_baseline"] <= 1.0
        # calibration is internally consistent with the recorded outcome
        assert e["correct_n"] == int((e["eval_n"] > 0) == bool(e["n_start_won"]))
        assert e["correct_baseline"] == int((e["eval_baseline"] > 0) == bool(e["baseline_start_won"]))
        assert e["bestmove_agree"] in (0, 1)

    # MCTS@baseline eval is a fixed function of the deal -> identical at every N level.
    for deal in (0, 1):
        assert len({e["eval_baseline"] for e in evals if e["deal"] == deal}) == 1

    # At N == baseline the challenger's eval/top-Qs equal the baseline's exactly (same search).
    e0 = flatten_evals(run_sweep(n_values=[8], deals=1, baseline=8, base_seed=0, workers=1))[0]
    assert e0["diff"] == 0.0 and e0["q1_n"] == e0["q1_baseline"] and e0["bestmove_agree"] == 1

    ecsv = tmp_path / "eval.csv"
    save_eval_csv(str(ecsv), evals)
    assert ecsv.read_text().splitlines()[0] == ",".join(_EVAL_COLS)


def test_shared_rng_collapses_at_baseline_but_independent_does_not():
    # Shared rng at N == baseline: identical mirrored games -> every pair is a split -> ci95 == 0.
    rows = run_sweep(n_values=[16], deals=4, baseline=16, base_seed=0, workers=1, collect_eval=False)
    assert rows[0]["winrate"] == 0.5 and rows[0]["ci95"] == 0.0
    # Independent play rng: the two mirrored games differ, so it need not collapse (just runs cleanly).
    rows2 = run_sweep(n_values=[16], deals=4, baseline=16, base_seed=0, workers=1,
                      collect_eval=False, independent_rng=True)
    assert 0.0 <= rows2[0]["winrate"] <= 1.0


def test_knowledge_milestone_columns():
    # With --knowledge the eval rows carry the per-game first-ply milestones (int, >= -1).
    evals = flatten_evals(run_sweep(n_values=[8], deals=2, baseline=8, base_seed=0, workers=1,
                                    knowledge=True))
    assert evals and len(_KNOW_COLS) == 8
    for e in evals:
        assert set(_KNOW_COLS) <= set(e)
        assert all(isinstance(e[c], int) and e[c] >= -1 for c in _KNOW_COLS)
    # Columns are part of the schema even without --knowledge (defaulted to -1) so the CSV is stable.
    off = flatten_evals(run_sweep(n_values=[8], deals=1, baseline=8, base_seed=0, workers=1))[0]
    assert all(off[c] == -1 for c in _KNOW_COLS)
    assert set(_KNOW_COLS) <= set(_EVAL_COLS)


def test_plots_write_png(tmp_path):
    pytest.importorskip("matplotlib")
    rows = run_sweep(n_values=[8, 16], deals=2, baseline=8, base_seed=1, workers=1)
    evals = flatten_evals(rows)
    for fn, args in [(save_plot, (rows, 8)), (save_eval_plot, (evals, 8)),
                     (save_calibration_plot, (evals, 8))]:
        png = tmp_path / f"{fn.__name__}.png"
        assert fn(str(png), *args) is True and png.stat().st_size > 0
