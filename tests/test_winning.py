"""The win condition and its outs: a usable king or a queued antechamber card both avert a loss."""
from __future__ import annotations

from imposterkings import cards
from imposterkings.actions import FLIP_KING, StepKind

from .helpers import mainstate, cid, sc


def test_loss_only_when_no_play_and_king_used():
    # Player0 cannot beat the Queen and has already used their king -> player1 wins.
    st = mainstate(hand0=(cid("Fool"),), hand1=(cid("Queen"),),
                   stack=(sc("Queen"),), kings=(True, False))
    st = st._begin_turn(0)
    assert st.is_terminal() and st.winner == 1


def test_unused_king_is_always_an_out():
    st = mainstate(hand0=(cid("Fool"),), hand1=(cid("Queen"),),
                   stack=(sc("Queen"),), kings=(False, False))
    st = st._begin_turn(0)
    assert not st.is_terminal()
    assert FLIP_KING in st.legal_moves()  # forced flip is the only move


def test_antechamber_ascension_prevents_loss_and_is_the_turn():
    # Empty hand, king used -> would lose, but a queued card ascends instead and the turn passes.
    st = mainstate(hand0=(), hand1=(cid("Queen"),), stack=(sc("Soldier"),),
                   kings=(True, False), antechambers=((cid("Elder"),), ()))
    st = st._begin_turn(0)
    assert not st.is_terminal()
    # Ascension is surfaced as player 0's own forced turn (single legal move) before it resolves.
    assert st.phase == StepKind.ASCEND and st.to_play == 0 and len(st.legal_moves()) == 1
    st = st.apply(st.legal_moves()[0])
    assert cards.card_name(st.stack[-1].card) == "Elder"  # ascended to lead
    assert st.antechambers[0] == ()
    assert st.to_play == 1  # ascension consumed the whole turn
