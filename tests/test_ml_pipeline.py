"""End-to-end ML pipeline: tiny corpus -> tensors -> MLP train (skipped if torch is absent)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from imposterkings import record  # noqa: E402
from imposterkings.data_analysis import datagen  # noqa: E402
from imposterkings.machine_learning import dataset, features  # noqa: E402
from imposterkings.machine_learning.mlp import MLP  # noqa: E402
from imposterkings.machine_learning.train import game_split, load_npz, parse_archs, train_one  # noqa: E402


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
