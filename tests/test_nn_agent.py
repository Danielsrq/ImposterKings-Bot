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


@pytest.mark.parametrize("feat", ["v1", "v2"])
def test_greedy_play_works_for_the_ATTENTION_net_not_just_the_mlp(tmp_path, feat):
    """NNPolicy hardcoded `features.encode` + `self.model(x)` -- the flat-216 MLP path -- so pointing the
    greedy benchmark at an attention checkpoint died with `KeyError: 'feature_dim'`. Greedy is how we ask
    'has the net learned the game, or is search doing the work', and it silently did not exist for the
    model the project actually ships."""
    import torch
    from imposterkings.machine_learning import attention_model as AM

    torch.manual_seed(0)
    m = AM.AttentionModel(AM.AttnConfig(d_model=32, n_layers=1, n_heads=4, feat=feat))
    p = tmp_path / "attn.pt"
    AM.save(str(p), m)

    pol = NNPolicy.from_checkpoint(str(p))          # dispatches on model_type, NOT on a feature_dim key
    assert pol.attn and pol.feat == feat
    view = _view()
    scored = pol.evaluate(view)
    assert len(scored) == len(view.legal_moves()) and all(isinstance(q, float) for _, q in scored)
    assert all(-1.0 <= q <= 1.0 for _, q in scored)              # bounded head
    assert NNAgent(pol).select_move(view, np.random.default_rng(0)) in view.legal_moves()


def test_greedy_scores_the_view_not_the_hidden_state():
    """Greedy must featurize the INFORMATION SET. attention_model.evaluator_from_model takes a GameState
    (MCTS calls it on an already-determinized node) -- reusing that seam here would hand the net the
    opponent's real hand and quietly inflate every greedy result."""
    import inspect

    from imposterkings.machine_learning.features2 import tokenize_state
    src = inspect.signature(tokenize_state).parameters["view"]
    assert src is not None                                        # the featurizer is view-native
    pol = NNPolicy(MLP(features.FEATURE_DIM, [8]))
    assert "view" in inspect.signature(pol.evaluate).parameters   # ...and so is the greedy entry point


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
