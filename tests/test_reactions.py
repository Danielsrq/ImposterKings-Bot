"""Reaction windows: King's Hand counters a "may", and the nested Assassin / King's-Hand on a flip."""
from __future__ import annotations

from imposterkings import cards
from imposterkings.actions import (
    Action, ActionKind, DECLARE, DECLINE_REACTION, FLIP_KING, REVEAL_ASSASSIN,
    REVEAL_KINGSHAND, StepKind,
)
from imposterkings.state import StackCard

from .helpers import mainstate, run, cid, sc


def _play(c):
    return Action(ActionKind.PLAY_CARD, card=c)


def test_kingshand_counters_may_and_discards_both():
    # Player0 plays Princess and declares the swap; player1 reveals King's Hand.
    st = mainstate(hand0=(cid("Princess"), cid("Fool")), hand1=(cid("KingsHand"), cid("Soldier")),
                   stack=(sc("Elder"),))
    st = run(st, _play(cid("Princess")), DECLARE)
    assert st.phase == StepKind.REACTION_KINGSHAND
    assert REVEAL_KINGSHAND in st.legal_moves()
    st = run(st, REVEAL_KINGSHAND)
    # Both the Princess and the King's Hand are discarded; Princess left the stack.
    assert all(cards.card_name(s.card) != "Princess" for s in st.stack)
    assert cid("Princess") in st.discard and cid("KingsHand") in st.discard
    assert cid("KingsHand") not in st.hands[1]


def test_kingshand_unavailable_when_muted():
    st = mainstate(hand0=(cid("Princess"),), hand1=(cid("KingsHand"),),
                   stack=(sc("Elder"),), muted={8})  # value 8 muted -> reaction tag stripped
    st = run(st, _play(cid("Princess")), DECLARE)
    assert REVEAL_KINGSHAND not in st.legal_moves()


def _flip_setup(hand0=(cid("Fool"),), hand1=()):
    # Player0 will flip the king; player1 may hold an Assassin. Hidden card present to be taken.
    return mainstate(hand0=hand0, hand1=hand1, stack=(sc("Queen"),),
                     hidden=(cid("Zealot"), None), kings=(False, False))


def test_assassin_uncountered_wins_immediately():
    st = _flip_setup(hand0=(cid("Fool"),), hand1=(cid("Assassin"),))
    st = run(st, FLIP_KING)
    assert st.phase == StepKind.REACTION_ASSASSIN
    st = run(st, REVEAL_ASSASSIN, DECLINE_REACTION)  # flipper has no King's Hand to counter
    assert st.is_terminal() and st.winner == 1


def test_assassin_countered_by_kingshand_flip_proceeds():
    st = _flip_setup(hand0=(cid("Fool"), cid("KingsHand")), hand1=(cid("Assassin"),))
    st = run(st, FLIP_KING, REVEAL_ASSASSIN)
    assert st.phase == StepKind.REACTION_KH_VS_ASSASSIN
    st = run(st, REVEAL_KINGSHAND)
    assert st.winner is None
    assert st.kings[0] is True                       # flip completed
    assert cid("Zealot") in st.hands[0]              # hidden card taken
    assert st.stack[-1].disgraced                    # top disgraced to 0
    assert cid("Assassin") in st.discard and cid("KingsHand") in st.discard


def test_flip_without_assassin_just_proceeds():
    st = _flip_setup(hand0=(cid("Fool"),), hand1=(cid("Soldier"),))
    st = run(st, FLIP_KING)
    # Opponent holds no Assassin -> the only reaction is to decline, then the flip resolves.
    assert st.legal_moves() == [DECLINE_REACTION]
    st = run(st, DECLINE_REACTION)
    assert st.kings[0] is True and st.winner is None and st.stack[-1].disgraced
