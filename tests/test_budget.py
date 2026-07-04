"""Per-decision budget policies and MCTSAgent budget wiring."""
from __future__ import annotations

import numpy as np

from imposterkings.actions import Action, ActionKind, StepKind
from imposterkings.agents import MCTSAgent
from imposterkings.budget import _effective_n, branching, fixed, hybrid, make_budget, opp_cards
from imposterkings.state import GameState

from .helpers import cid, mainstate


def _view(hand1=(cid("Soldier"), cid("Judge"), cid("Fool"))):
    # opp (seat 1) holds 3 cards, no hidden -> opp_cards == 3
    st = mainstate(hand0=(cid("Queen"),), hand1=hand1)
    return st.information_set(0)


def test_opp_cards_counts_hand_plus_hidden():
    v = _view()
    assert opp_cards(v) == 3                                  # 3 hand, no hidden
    v2 = mainstate(hand0=(cid("Queen"),), hand1=(cid("Soldier"),),
                   hidden=(None, cid("Warlord"))).information_set(0)
    assert opp_cards(v2) == 2                                 # 1 hand + 1 hidden


def test_fixed_and_branching_and_hybrid_math_and_clamp():
    v = _view()                                              # opp_cards == 3
    assert fixed(800)(v, 5) == 800
    assert branching(50)(v, 6) == 300                        # 50*6
    assert branching(50)(v, 1) == 64                         # floor
    assert branching(50)(v, 100) == 4096                     # ceil
    assert hybrid(100)(v, 5) == 100 * 5 * (1 + 3)            # 2000
    assert hybrid(1)(v, 1) == 64                             # 1*1*4=4 -> floor 64
    assert hybrid(100)(v, 15) == 4096                        # 100*15*4=6000 -> ceil 4096
    assert hybrid(100).label == "hybrid-k100-l3" and branching(50).label == "branching-k50-l3"


def test_effective_n_weights_sub_decision_cards():
    soldier = Action(ActionKind.PLAY_CARD, card=cid("Soldier"))   # heavy (mandatory guess)
    elder = Action(ActionKind.PLAY_CARD, card=cid("Elder"))       # plain
    flip = Action(ActionKind.FLIP_KING)                           # not PLAY_CARD -> 1
    moves = [soldier, elder, flip]
    assert _effective_n(moves, 3) == 3 + 1 + 1                    # Soldier(l=3) + Elder(1) + flip(1)
    assert _effective_n(moves, 1) == 3                           # l=1 -> no weighting
    assert _effective_n(7, 3) == 7                               # bare int passes through
    # the budget sizes by the weighted count: Soldier+Mystic (both heavy) -> 2*l, not 2
    v = _view()                                                  # opp_cards == 3
    heavy2 = [Action(ActionKind.PLAY_CARD, card=cid("Soldier")),
              Action(ActionKind.PLAY_CARD, card=cid("Mystic"))]
    assert branching(50, 3)(v, heavy2) == 300                    # 50 * (2*3)
    assert hybrid(100, 3)(v, heavy2) == 100 * 6 * (1 + 3)        # 2400


def test_make_budget_dispatch():
    v = _view()
    assert make_budget("hybrid", k=100)(v, 5) == hybrid(100)(v, 5)
    assert make_budget("branching", k=50)(v, 6) == branching(50)(v, 6)
    try:
        make_budget("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown mode")


def test_mctsagent_uses_the_budget():
    rng = np.random.default_rng(0)
    st = GameState.deal(rng, starting_player=0)
    while st.phase in (StepKind.SETUP_HIDE, StepKind.SETUP_DISCARD):
        st = st.apply(st.legal_moves()[0])
    view = st.information_set(st.to_play)
    b = hybrid(50)
    agent = MCTSAgent(budget=b)
    assert agent.name == "mcts-hybrid-k50-l3"
    moves = view.legal_moves()
    agent.select_move(view, rng)
    assert agent.last_result.iterations == b(view, moves)   # searched at the (move-weighted) budget
