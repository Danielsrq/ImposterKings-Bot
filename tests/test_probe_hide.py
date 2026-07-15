"""The hide-Assassin search probe.

The probe answers a causal question the corpus cannot: does an evaluator trained on hidden-Assassin games
make the SEARCH start choosing that move? These tests pin the two things that would make the probe lie:

  1. It must only measure positions where hiding the Assassin is actually LEGAL (the seat holds it).
  2. Same position + same search seed for every net -> the search is reproducible, so a later difference is
     attributable to the evaluator and nothing else.

They use a tiny torch MLP as a stand-in evaluator (fast, no checkpoint) via NNPolicy -- the probe's logic is
evaluator-agnostic.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from imposterkings.actions import Action, ActionKind, StepKind  # noqa: E402
from imposterkings.agents import MCTSAgent  # noqa: E402
from imposterkings.data_analysis.forced_hide import ASSASSIN, assassin_seat  # noqa: E402
from imposterkings.data_analysis.probe_hide import _setup_hide_node, probe_one  # noqa: E402
from imposterkings.machine_learning import attention_model as AM  # noqa: E402
from imposterkings.machine_learning.agent import NNPolicy  # noqa: E402
from imposterkings.state import GameState  # noqa: E402


def _net():
    torch.manual_seed(0)
    m = AM.AttentionModel(AM.AttnConfig(d_model=32, n_layers=2, n_heads=4, feat="v2"))
    return m, AM.evaluator_from_model(m), NNPolicy(m)


def test_setup_hide_node_lands_on_the_assassin_holders_hide_decision():
    seed = next(s for s in range(200) if assassin_seat(s) is not None)
    seat = assassin_seat(seed)
    st = _setup_hide_node(seed, seat)
    assert st.to_play == seat and st.phase == StepKind.SETUP_HIDE
    assert ASSASSIN in st.hands[seat]                               # the mover really holds it
    assert Action(ActionKind.HIDE_CARD, card=ASSASSIN) in st.information_set(seat).legal_moves()


def test_probe_returns_none_when_hiding_the_assassin_is_not_legal():
    """A seat that does NOT hold the Assassin cannot hide it -- the probe must skip, not fabricate a row."""
    seed = next(s for s in range(200) if assassin_seat(s) is not None)
    holder = assassin_seat(seed)
    other = 1 - holder
    st = _other_hide_node(seed, other)
    if st is None:
        pytest.skip("no clean opposite-seat hide node for this seed")
    m, ev, pol = _net()
    agent = MCTSAgent(iterations=30, evaluator=ev)
    assert probe_one(st, other, agent, pol, ev, rng_seed=0) is None


def _other_hide_node(seed, seat):
    """The hide node of the seat that does NOT hold the Assassin (or None if it can't be reached cleanly)."""
    st = GameState.deal(np.random.default_rng(seed))
    for _ in range(9):
        if st.to_play == seat and st.phase == StepKind.SETUP_HIDE:
            return st if ASSASSIN not in st.hands[seat] else None
        moves = st.legal_moves()
        if not moves:
            return None
        st = st.apply(moves[0])
    return None


def test_probe_row_has_every_channel_and_flip_is_a_bool():
    seed = next(s for s in range(200) if assassin_seat(s) is not None)
    seat = assassin_seat(seed)
    m, ev, pol = _net()
    agent = MCTSAgent(iterations=40, evaluator=ev)
    r = probe_one(_setup_hide_node(seed, seat), seat, agent, pol, ev, rng_seed=0)
    assert r is not None
    assert set(r) >= {"prior_q", "v_hidden", "share", "q_backed", "q_gap", "flipped", "best_card"}
    assert isinstance(r["flipped"], bool)
    assert 0.0 <= r["share"] <= 1.0
    assert -1.0 <= r["prior_q"] <= 1.0 and -1.0 <= r["v_hidden"] <= 1.0


def test_same_seed_makes_the_search_reproducible():
    """The whole design rests on this: identical position + identical search seed -> identical result, so a
    later difference is the EVALUATOR, never search variance."""
    seed = next(s for s in range(200) if assassin_seat(s) is not None)
    seat = assassin_seat(seed)
    m, ev, pol = _net()
    agent = MCTSAgent(iterations=50, evaluator=ev)
    a = probe_one(_setup_hide_node(seed, seat), seat, agent, pol, ev, rng_seed=7)
    b = probe_one(_setup_hide_node(seed, seat), seat, agent, pol, ev, rng_seed=7)
    assert a["share"] == b["share"] and a["q_backed"] == b["q_backed"] and a["flipped"] == b["flipped"]
