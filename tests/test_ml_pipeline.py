"""End-to-end ML pipeline: tiny corpus -> tensors -> MLP train (skipped if torch is absent)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from imposterkings import record  # noqa: E402
from imposterkings.data_analysis import datagen  # noqa: E402
from imposterkings.machine_learning import dataset, features  # noqa: E402
from imposterkings.machine_learning.mlp import MLP  # noqa: E402
from imposterkings.machine_learning.train import (  # noqa: E402
    corpus_of, game_split, load_npz, parse_archs, per_corpus_metrics, train_one)


def test_corpus_of_recovers_the_source_from_the_deal_seed():
    """Pooled datasets carry no corpus label. game_id == deal_seed and each corpus owns a disjoint --base-seed
    DECADE (the same property token_dataset._guard_seed already leans on), so the decade IS the corpus id."""
    gid = np.array([10_000_000, 10_001_999, 30_000_000, 30_001_999, 40_000_000, 40_000_999])
    assert corpus_of(gid).tolist() == [1, 1, 3, 3, 4, 4]


def test_per_corpus_metrics_splits_a_pooled_val_set_and_is_silent_on_a_single_corpus():
    """The point of the breakdown: the asymmetric `mixed` corpus labels only ONE seat (~58 rows/game vs ~119),
    so it is outnumbered ~4:1 and a single aggregate val_mse could hide a model that ignores it entirely."""
    gid = np.array([10_000_000] * 4 + [40_000_000] * 4)
    did = np.array([0, 0, 1, 1, 2, 2, 3, 3])                  # 2 candidates per decision
    y = np.array([0.5, -0.5, 0.5, -0.5, 0.9, -0.9, 0.9, -0.9], dtype=np.float32)
    chosen = np.array([1, 0, 1, 0, 1, 0, 1, 0])

    pred = y.copy()                                            # perfect on gen-1...
    pred[4:] = -y[4:]                                          # ...and exactly inverted on the mixed corpus
    by = per_corpus_metrics(pred, y, gid, did, chosen)

    assert set(by) == {1, 4}                                   # only the corpora actually present
    assert by[1]["name"] == "gen1_v3c (k20)" and by[4]["name"] == "mixed_v3c_vs_mcts500"
    assert by[1]["games"] == 1 and by[1]["rows"] == 4
    assert by[1]["val_mse"] == pytest.approx(0.0) and by[1]["top1_bestq"] == 1.0
    assert by[4]["val_mse"] > 1.0 and by[4]["top1_bestq"] == 0.0    # the failure the aggregate would mask

    # a single-corpus dataset gets NO breakdown -- there is nothing to compare, so the trainer stays quiet
    assert per_corpus_metrics(pred[:4], y[:4], gid[:4], did[:4], chosen[:4]) == {}


def test_parse_archs():
    assert parse_archs("16;32;64") == [[16], [32], [64]]
    assert parse_archs("16,16;32") == [[16, 16], [32]]
    assert parse_archs("") == [[]]                       # linear model


def test_mlp_any_shape_and_bounded_output():
    import torch
    for shape in ([16], [16, 16], []):
        m = MLP(features.FEATURE_DIM, shape)
        out = m(torch.zeros(4, features.FEATURE_DIM)).detach()
        assert out.shape == (4, 1) and float(out.abs().max()) <= 1.0
        assert m.param_count() > 0


def test_dataset_build_then_train(tmp_path):
    recs = [datagen.collect_game(("hybrid", 3, 3), seed=s, temp_plies=0, base_seed=0) for s in (1, 2, 3, 4)]
    ddir = tmp_path / "corpus"
    ddir.mkdir()
    record.write_jsonl(str(ddir / "games_00000.jsonl"), recs)
    npz = str(tmp_path / "t.npz")

    stats = dataset.build(str(ddir), npz)
    assert stats["feature_dim"] == features.FEATURE_DIM and stats["n_rows"] > 0

    data = load_npz(npz)
    assert data["X"].shape[1] == features.FEATURE_DIM
    assert (np.abs(data["y"]) <= 1.0 + 1e-6).all()        # q target in [-1,1]
    assert set(np.unique(data["is_chosen"])) <= {0, 1}

    tr, va = game_split(data["game_id"], val_frac=0.25, seed=0)
    r = train_one(data, tr, va, [8],
                  {"epochs": 2, "batch": 128, "lr": 1e-3, "dropout": 0.0, "patience": 2, "seed": 0})
    assert {"val_mse", "val_mae", "top1_bestq", "top1_chosen", "spearman", "params"} <= set(r)
    assert r["val_mse"] >= 0.0 and 0.0 <= r["top1_bestq"] <= 1.0
