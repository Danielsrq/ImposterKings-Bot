"""NN bot: checkpoint round-trip, the reusable picker/agent, and the benchmark chunk (skip if no torch)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from imposterkings import scenario as sb  # noqa: E402
from imposterkings.machine_learning import checkpoint, features  # noqa: E402
from imposterkings.machine_learning.agent import NNAgent, NNPolicy  # noqa: E402
from imposterkings.machine_learning.benchmark import _chunk, parse_spec  # noqa: E402
from imposterkings.machine_learning.mlp import MLP  # noqa: E402


def _view():
    st = sb.build(hand0=["Judge", "Queen", "Fool"], hand1=["Warlord", "Elder"],
                  stack=["Zealot"], turn_player=0)
    return st.information_set(0)


def test_checkpoint_round_trip(tmp_path):
    import torch
    m = MLP(features.FEATURE_DIM, [8, 8])
    p = str(tmp_path / "m.pt")
    checkpoint.save(p, m, meta={"target": "q"})
    m2, meta = checkpoint.load(p)
    assert m2.in_dim == features.FEATURE_DIM and m2.hidden_dims == [8, 8] and meta["target"] == "q"
    x = torch.randn(3, features.FEATURE_DIM)
    assert torch.allclose(m(x), m2(x))                      # reloaded model predicts identically


def test_policy_and_agent_pick_a_legal_move():
    pol = NNPolicy(MLP(features.FEATURE_DIM, [8]))
    view = _view()
    scored = pol.evaluate(view)
    assert len(scored) == len(view.legal_moves()) and all(isinstance(q, float) for _, q in scored)
    assert pol.best_move(view) in view.legal_moves()
    assert NNAgent(pol).select_move(view, np.random.default_rng(0)) in view.legal_moves()


def test_parse_spec():
    assert parse_spec("fixed500") == ("fixed", 500)
    assert parse_spec("hybrid-k20-l3") == ("hybrid", 20, 3)
    assert parse_spec("branching-k30-l3") == ("branching", 30, 3)


def test_chunk_vs_tiny_mcts(tmp_path):
    p = str(tmp_path / "m.pt")
    checkpoint.save(p, MLP(features.FEATURE_DIM, [8]))
    res = _chunk(p, "hybrid-k3-l3", ("hybrid", 3, 3), [0, 1], independent_rng=True)
    assert len(res["rows"]) == 2
    for r in res["rows"]:
        assert r["nn_wins"] in (0, 1, 2) and r["score"] in (0.0, 0.5, 1.0)
