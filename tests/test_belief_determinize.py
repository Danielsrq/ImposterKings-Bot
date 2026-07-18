"""Belief-weighted determinization: the opt-in sampler reweights which opponent worlds SO-ISMCTS searches,
while the pinned/has/lacks hard constraints and world legality are preserved. The determinize tests are
engine-only; the factory/engine-smoke tests need torch + the belief checkpoint and skip if either is absent.
"""
from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

from imposterkings.cards import card_name
from imposterkings.infoset import InformationSet
from imposterkings.state import GameState

_CKPT = os.path.join("models", "belief_5k", "belief_d64_L2.pt")


def _midgame_view(seed=3, plies=6):
    s = GameState.deal(np.random.default_rng(seed))
    for _ in range(plies):
        if s.is_terminal():
            break
        s = s.apply(s.legal_moves()[0])
    return s, InformationSet.from_state(s, s.to_play)


def _uniform_w(view):
    return {c: np.array([1.0, 1.0, 1.0, 1.0]) for c in view.unknown_cards()}


# --- the sampler reweights, but stays legal and honours hard constraints ----------------------------

def test_belief_weight_pulls_a_card_into_the_opponent_hand():
    s, view = _midgame_view()
    opp = 1 - view.observer
    fav = view.unknown_cards()[0]
    bw = _uniform_w(view)
    bw[fav] = np.array([50.0, 0.1, 0.1, 0.1])                 # strongly favour fav in their_hand

    def rate(weights):
        return sum(fav in view.determinize(np.random.default_rng(1000 + i), belief_weights=weights).hands[opp]
                   for i in range(300))

    assert rate(bw) > rate(None) + 30                        # belief lands fav in hand far more than uniform


def test_uniform_path_is_deterministic_and_legal():
    s, view = _midgame_view()
    opp = 1 - view.observer
    a = view.determinize(np.random.default_rng(7))
    b = view.determinize(np.random.default_rng(7))
    assert a.hands[opp] == b.hands[opp]                       # belief_weights=None -> unchanged uniform draw
    assert len(a.hands[opp]) == view.opp_hand_count
    w = view.determinize(np.random.default_rng(7), belief_weights=_uniform_w(view))
    assert len(w.hands[opp]) == view.opp_hand_count          # belief path also builds a legal hand


def test_lacks_overrides_a_contrary_belief():
    s, view = _midgame_view()
    opp = 1 - view.observer
    nm = card_name(view.unknown_cards()[0])
    vv = replace(view, opp_hand_lacks=frozenset({nm}))
    bw = _uniform_w(view)
    for c in view.unknown_cards():                           # scream that nm is in their_hand
        if card_name(c) == nm:
            bw[c] = np.array([99.0, 0.01, 0.01, 0.01])
    for i in range(200):
        st = vv.determinize(np.random.default_rng(i), belief_weights=bw)
        assert all(card_name(c) != nm for c in st.hands[opp])   # a lacked name is NEVER in the hand


def test_has_forces_a_name_in_despite_a_contrary_belief():
    s, view = _midgame_view()
    opp = 1 - view.observer
    nm = card_name(view.unknown_cards()[0])
    vv = replace(view, opp_hand_has=frozenset({nm}))
    bw = _uniform_w(view)
    for c in view.unknown_cards():                           # belief says nm is NOT in their hand
        if card_name(c) == nm:
            bw[c] = np.array([0.001, 1.0, 1.0, 1.0])
    for i in range(200):
        st = vv.determinize(np.random.default_rng(i), belief_weights=bw)
        assert any(card_name(c) == nm for c in st.hands[opp])   # a has-name is ALWAYS present


def test_every_card_is_placed_exactly_once():
    s, view = _midgame_view()
    obs, opp = view.observer, 1 - view.observer
    bw = _uniform_w(view)
    bw[view.unknown_cards()[0]] = np.array([0.1, 9.0, 0.1, 0.1])   # push one toward their_hidden
    st = view.determinize(np.random.default_rng(11), belief_weights=bw)
    placed = list(st.hands[0]) + list(st.hands[1])
    for h in (st.hidden[obs], st.hidden[opp], st.setup_discard[obs], st.setup_discard[opp],
              st.leftover_faceup, st.leftover_facedown):
        if h is not None and h >= 0:
            placed.append(h)
    placed += [sc.card for sc in st.stack] + list(st.discard)
    for ante in st.antechambers:
        placed += list(ante)
    assert sorted(placed) == list(range(18))                 # a legal, complete world


# --- the factory + the engine smoke (need torch + the belief checkpoint) ----------------------------

def test_belief_sampler_factory_output():
    pytest.importorskip("torch")
    if not os.path.exists(_CKPT):
        pytest.skip("belief checkpoint not on disk")
    from imposterkings.machine_learning.attention_model import build_belief_sampler
    sampler = build_belief_sampler(_CKPT)
    s, view = _midgame_view()
    w = sampler(view)
    assert set(w) == set(view.unknown_cards())               # keyed by exactly the unknown ids
    for v in w.values():
        assert v.shape == (4,) and abs(float(v.sum()) - 1.0) < 1e-5   # a simplex over the 4 hidden zones
    seen = set(range(18)) - set(view.unknown_cards())
    assert seen and all(c not in w for c in seen)            # seen cards carry no belief entry


def test_mcts_agent_with_belief_returns_a_legal_move():
    pytest.importorskip("torch")
    if not os.path.exists(_CKPT):
        pytest.skip("belief checkpoint not on disk")
    from imposterkings.agents import MCTSAgent
    from imposterkings.machine_learning.attention_model import build_belief_sampler
    sampler = build_belief_sampler(_CKPT)
    s, view = _midgame_view()
    agent = MCTSAgent(iterations=30, belief=sampler)
    mv = agent.select_move(view, np.random.default_rng(0))
    assert mv in view.legal_moves() and agent.name.endswith("+belief")
