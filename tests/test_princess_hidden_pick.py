"""Princess swap: the give-card is HIDDEN from the responder (ruling 2026-10-18) -- neither player knows
what the other will give until the swap resolves. The responder's information set must not carry the
Princess player's pick, and determinize must re-sample it from the sampled Princess hand."""
from __future__ import annotations

import numpy as np

from imposterkings import scenario as sb
from imposterkings.actions import DECLARE, DECLINE_REACTION, StepKind


def _at_swap_respond():
    st = sb.build(hand0=["Princess", "Fool", "Elder"], hand1=["Soldier", "Zealot", "Judge"],
                  stack=["Soldier"], hidden=("Sentry", "Mystic"), turn_player=0)
    st = sb.play(st, sb.play_card("Princess"), DECLARE, DECLINE_REACTION, sb.choose_hand_card("Fool"))
    assert st.pending[-1].kind == StepKind.ABILITY_SWAP_RESPOND and st.pending[-1].actor == 1
    return st


def test_responder_does_not_see_the_give_card():
    st = _at_swap_respond()
    assert st.pending[-1].picked == sb.cid("Fool")                 # the truth carries it
    assert st.information_set(1).pending[-1].picked is None        # the responder's view does not
    assert st.information_set(0).pending[-1].picked == sb.cid("Fool")   # the Princess player knows her own


def test_determinize_resamples_the_give_card_from_the_sampled_hand():
    info = _at_swap_respond().information_set(1)
    picks = set()
    for s in range(60):
        w = info.determinize(np.random.default_rng(s))
        p = w.pending[-1].picked
        assert p is not None and p in w.hands[0]                    # a card the Princess player holds
        picks.add(p)
        w.apply(w.legal_moves()[0])                                 # the swap resolves in every world
    assert len(picks) > 3                                           # not pinned to the true Fool
