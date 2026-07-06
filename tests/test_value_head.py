"""NN value/policy head in ISMCTS: evaluator correctness, PUCT skips rollout, classic path unchanged."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from imposterkings import mcts as M  # noqa: E402
from imposterkings import scenario as sb  # noqa: E402
from imposterkings.machine_learning import checkpoint, features  # noqa: E402
from imposterkings.machine_learning.evaluator import build_evaluator  # noqa: E402
from imposterkings.machine_learning.mlp import MLP  # noqa: E402
from imposterkings.mcts import SearchConfig, search  # noqa: E402
from imposterkings.state import GameState  # noqa: E402


def _ckpt(tmp_path) -> str:
    p = str(tmp_path / "m.pt")
    checkpoint.save(p, MLP(features.FEATURE_DIM, [8]))
    return p


def test_evaluator_value_and_priors(tmp_path):
    ev = build_evaluator(_ckpt(tmp_path))
    st = sb.build(hand0=["Judge", "Queen", "Fool"], hand1=["Warlord", "Elder"],
                  stack=["Zealot"], turn_player=0)
    value, priors = ev(st)
    assert len(value) == 2 and abs(value[0] + value[1]) < 1e-6 and -1.0 <= value[0] <= 1.0  # zero-sum
    assert abs(sum(priors.values()) - 1.0) < 1e-5 and set(priors) == set(st.legal_moves())


def test_puct_search_skips_rollout(tmp_path, monkeypatch):
    ev = build_evaluator(_ckpt(tmp_path))
    st = GameState.deal(np.random.default_rng(1))
    view = st.information_set(st.to_play)

    def _boom(*a, **k):
        raise AssertionError("_rollout must not run when an evaluator is attached")
    monkeypatch.setattr(M, "_rollout", _boom)

    res = search(view, SearchConfig(rng=np.random.default_rng(1), iterations=60, evaluator=ev))
    assert res.best_move in view.legal_moves() and len(res.stats) >= 1
    assert -1.0 <= res.root_value() <= 1.0


def test_classic_search_unchanged(tmp_path):
    st = GameState.deal(np.random.default_rng(2))
    view = st.information_set(st.to_play)
    res = search(view, SearchConfig(rng=np.random.default_rng(2), iterations=60))  # no evaluator
    assert res.best_move in view.legal_moves() and len(res.stats) >= 1
