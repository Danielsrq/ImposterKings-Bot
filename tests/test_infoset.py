"""InformationSet projection, near-perfect-info determinization, and the arena driver."""
from __future__ import annotations

import numpy as np

from imposterkings import cards
from imposterkings.actions import StepKind
from imposterkings.agents import RandomAgent
from imposterkings.arena import run_arena
from imposterkings.state import GameState


def _dealt_to_main(seed: int) -> GameState:
    rng = np.random.default_rng(seed)
    st = GameState.deal(rng, starting_player=0)
    while st.phase in (StepKind.SETUP_HIDE, StepKind.SETUP_DISCARD):
        st = st.apply(st.legal_moves()[0])
    return st


def test_determinize_is_a_consistent_full_world():
    st = _dealt_to_main(7)
    view = st.information_set(st.to_play)
    world = view.determinize(np.random.default_rng(99))
    # Observer's own concealed info is preserved exactly.
    assert world.hands[view.observer] == view.own_hand
    assert world.hidden[view.observer] == view.own_hidden
    assert len(world.hands[1 - view.observer]) == view.opp_hand_count
    # Every one of the 18 cards appears exactly once across all zones.
    ids = (list(world.hands[0]) + list(world.hands[1])
           + [h for h in world.hidden if h is not None]
           + [d for d in world.setup_discard if d is not None]
           + list(world.discard) + [sc.card for sc in world.stack]
           + [c for ante in world.antechambers for c in ante]
           + [world.leftover_faceup, world.leftover_facedown])
    assert sorted(ids) == list(range(cards.DECK_SIZE))


def test_observer_legal_moves_are_world_independent():
    st = _dealt_to_main(3)
    view = st.information_set(st.to_play)
    # The observer's own options never depend on the opponent's concealed hand.
    assert set(view.legal_moves()) == set(st.legal_moves())


def test_determinize_varies_opponent_hand():
    st = _dealt_to_main(11)
    view = st.information_set(st.to_play)
    worlds = {view.determinize(np.random.default_rng(s)).hands[1 - view.observer] for s in range(20)}
    assert len(worlds) > 1  # sampling actually explores different opponent hands


def test_arena_random_vs_random_tallies_all_games():
    rng = np.random.default_rng(0)
    wins = run_arena([RandomAgent(), RandomAgent()], games=30, rng=rng, swap=True)
    assert sum(wins) == 30
