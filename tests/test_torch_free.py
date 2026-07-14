"""Prove the shipped game runs on a machine with NO torch.

This is the test the release actually rests on, and no other test can stand in for it: the dev suite runs
WITH torch installed, so every import succeeds there whether or not it should. Here we sabotage the import
system to make `import torch` raise ImportError -- exactly what a player's machine looks like -- and then
demand that the app still imports, still picks moves, and still explains them.

If this passes, `--exclude-module torch` in the PyInstaller spec is safe. If it fails, the frozen .exe
crashes on someone else's computer and we find out from a bug report.
"""
import builtins
import importlib
import sys

import numpy as np
import pytest

from imposterkings.actions import StepKind
from imposterkings.state import GameState

# these are the modules that must survive torch's absence; drop them so they re-import under the sabotage
_UNDER_TEST = ("imposterkings.machine_learning.explain",
               "imposterkings.machine_learning.npz_infer",
               "imposterkings.ui.app", "imposterkings.ui.review")


@pytest.fixture
def no_torch(monkeypatch):
    """Make `import torch` raise, as it would on a machine that never installed it."""
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("No module named 'torch' (simulated: the shipped game has no torch)")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    for m in list(sys.modules):
        if m == "torch" or m.startswith("torch.") or m in _UNDER_TEST:
            monkeypatch.delitem(sys.modules, m, raising=False)
    yield


@pytest.fixture(scope="module")
def npz(tmp_path_factory):
    """A tiny attention net exported to .npz -- the format the release ships."""
    torch = pytest.importorskip("torch")                 # building the fixture needs torch; USING it must not
    from imposterkings.machine_learning import attention_model as AM
    from imposterkings.machine_learning.export_npz import export
    torch.manual_seed(0)
    m = AM.AttentionModel(AM.AttnConfig(d_model=32, n_layers=2, n_heads=4, feat="v2")).eval()
    p = tmp_path_factory.mktemp("m") / "attn.pt"
    AM.save(str(p), m)
    return export(str(p))


def test_torch_really_is_blocked(no_torch):
    with pytest.raises(ImportError):
        importlib.import_module("torch")


def test_the_app_and_the_explainer_import_without_torch(no_torch):
    """ui.app is the game's entry point and explain.py draws the attention panel. Neither may need torch."""
    for name in _UNDER_TEST:
        importlib.import_module(name)                    # raises if any of them reaches torch at import time


def test_an_npz_bot_picks_a_move_without_torch(no_torch, npz):
    """The full bot path: load .npz -> numpy leaf evaluator -> PUCT search -> a legal move."""
    from imposterkings.agents import MCTSAgent
    from imposterkings.machine_learning.benchmark import _evaluator_for   # its .npz branch must not touch torch

    st = GameState.deal(np.random.default_rng(0), starting_player=0)
    while st.phase in (StepKind.SETUP_HIDE, StepKind.SETUP_DISCARD):
        st = st.apply(st.legal_moves()[0])
    view = st.information_set(st.to_play)

    agent = MCTSAgent(iterations=40, evaluator=_evaluator_for(npz))
    move = agent.select_move(view, np.random.default_rng(0))
    assert move in view.legal_moves()


def test_the_attention_drawer_explains_a_move_without_torch(no_torch, npz):
    """The explainability feature -- the whole point of shipping the attention net -- must work torch-free."""
    from imposterkings.machine_learning.explain import explain
    from imposterkings.machine_learning.npz_infer import load

    st = GameState.deal(np.random.default_rng(1), starting_player=0)
    while st.phase in (StepKind.SETUP_HIDE, StepKind.SETUP_DISCARD):
        st = st.apply(st.legal_moves()[0])
    view = st.information_set(st.to_play)

    ex = explain(view, st.legal_moves()[0], load(npz), all_layers=True, attribution=True)
    assert -1.0 <= ex.q <= 1.0
    assert ex.attn.shape[0] == ex.n_heads and len(ex.per_layer) == ex.n_layers
    assert ex.attribution is not None and ex.zone_posterior is not None   # the belief block renders too
    assert "torch" not in sys.modules                                     # nothing pulled it in behind our back
