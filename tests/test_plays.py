"""Play legality and the passive/override plays (Queen, Elder, Zealot, Oathbound, Warlord)."""
from __future__ import annotations

from imposterkings import cards
from imposterkings.actions import Action, ActionKind, FLIP_KING, StepKind
from imposterkings.state import StackCard

from .helpers import mainstate, run, cid, sc


def _play(card):
    return Action(ActionKind.PLAY_CARD, card=card)


def test_queen_disgraces_beneath_and_stays_leading():
    st = mainstate(hand0=(cid("Queen"),), hand1=(cid("Warlord"),),
                   stack=(sc("Soldier"), sc("Elder")))
    st = run(st, _play(cid("Queen")))
    # Everything beneath the Queen is disgraced; the Queen leads at 9; royalty is present.
    assert all(s.disgraced for s in st.stack[:-1])
    top = st.stack[-1]
    assert cards.card_name(top.card) == "Queen" and not top.disgraced
    assert st.effective_stack_value(top) == 9
    assert st.royalty_present()


def test_queen_is_not_kingshand_counterable():
    # Opponent holds King's Hand, but Queen is mandatory -> no reaction window is ever opened.
    st = mainstate(hand0=(cid("Queen"),), hand1=(cid("KingsHand"),), stack=(sc("Soldier"),))
    st = run(st, _play(cid("Queen")))
    assert st.phase != StepKind.REACTION_KINGSHAND
    assert cards.card_name(st.stack[-1].card) == "Queen"  # Queen was not discarded by a counter


def test_elder_plays_over_royalty():
    st = mainstate(hand0=(cid("Elder"),), hand1=(cid("Queen"),), stack=(sc("Queen"),))
    assert any(m.card == cid("Elder") for m in st.legal_moves() if m.kind == ActionKind.PLAY_CARD)
    st2 = run(st, _play(cid("Elder")))
    assert cards.card_name(st2.stack[-1].card) == "Elder"


def test_zealot_requires_own_flipped_king():
    # Leading non-royalty Warlord(7); Zealot(3) can only override if its own king is flipped.
    without = mainstate(hand0=(cid("Zealot"),), stack=(sc("Warlord"),), kings=(False, False))
    assert not any(m.card == cid("Zealot") for m in without.legal_moves()
                   if m.kind == ActionKind.PLAY_CARD)
    withking = mainstate(hand0=(cid("Zealot"),), stack=(sc("Warlord"),), kings=(True, False))
    assert any(m.card == cid("Zealot") for m in withking.legal_moves()
               if m.kind == ActionKind.PLAY_CARD)


def test_oathbound_override_disgraces_beaten_card_then_free_play():
    st = mainstate(hand0=(cid("Oathbound"), cid("Fool")), hand1=(cid("Queen"),),
                   stack=(sc("KingsHand"),))  # leading value 8 > 6
    st = run(st, _play(cid("Oathbound")))
    assert st.phase == StepKind.OATHBOUND_SECOND
    # The BEATEN card (King's Hand) is disgraced; the Oathbound is NOT disgraced and leads at 6.
    kh = next(s for s in st.stack if cards.card_name(s.card) == "KingsHand")
    ob = next(s for s in st.stack if cards.card_name(s.card) == "Oathbound")
    assert kh.disgraced and not ob.disgraced
    assert st.effective_stack_value(ob) == 6
    # The immediate second card can be ANY value (part of the ability) -- Fool(1) is offered.
    assert any(m.card == cid("Fool") for m in st.legal_moves())
    st = run(st, _play(cid("Fool")))
    assert cards.card_name(st.stack[-1].card) == "Fool"


def test_oathbound_from_antechamber_does_not_trigger_override():
    # Oathbound ascends from player 0's antechamber over a higher card; its override must NOT fire --
    # the antechamber mechanic already let it beat the card.
    st = mainstate(hand0=(cid("Fool"),), hand1=(cid("Queen"),), stack=(sc("KingsHand"),),
                   antechambers=((cid("Oathbound"),), ()))  # leading value 8 > 6
    st = st._begin_turn(0)
    assert st.phase == StepKind.ASCEND       # ascension is now a forced, recorded turn
    st = st.apply(st.legal_moves()[0])
    top = st.stack[-1]
    assert cards.card_name(top.card) == "Oathbound" and not top.disgraced
    assert st.effective_stack_value(top) == 6
    kh = next(s for s in st.stack if cards.card_name(s.card) == "KingsHand")
    assert not kh.disgraced                    # beaten card NOT disgraced -> override didn't fire
    assert st.phase != StepKind.OATHBOUND_SECOND
    assert st.to_play == 1                      # ascension consumed the turn; no follow-up card


def test_oathbound_as_last_card_cannot_override():
    # Lone Oathbound cannot override a higher card (no follow-up card to play) -> forced to flip.
    st = mainstate(hand0=(cid("Oathbound"),), hand1=(cid("Queen"),),
                   stack=(sc("KingsHand"),), kings=(False, False))  # leading value 8 > 6
    plays = {m.card for m in st.legal_moves() if m.kind == ActionKind.PLAY_CARD}
    assert cid("Oathbound") not in plays
    assert FLIP_KING in st.legal_moves()
    # With a second card in hand, the override becomes available.
    st2 = mainstate(hand0=(cid("Oathbound"), cid("Fool")), hand1=(cid("Queen"),),
                    stack=(sc("KingsHand"),), kings=(False, False))
    assert cid("Oathbound") in {m.card for m in st2.legal_moves() if m.kind == ActionKind.PLAY_CARD}


def test_warlord_lands_at_nine_with_royalty():
    # Royalty present (Princess at the bottom), low card leading so Warlord is playable.
    st = mainstate(hand0=(cid("Warlord"),), hand1=(cid("Queen"),),
                   stack=(sc("Princess"), sc("Elder")))
    st = run(st, _play(cid("Warlord")))
    top = st.stack[-1]
    assert cards.card_name(top.card) == "Warlord"
    assert st.effective_stack_value(top) == 9


def test_flip_king_offered_only_when_stack_nonempty_and_king_unused():
    empty = mainstate(hand0=(cid("Fool"),), kings=(False, False))
    assert FLIP_KING not in empty.legal_moves()  # nothing to disgrace on an empty stack
    nonempty = mainstate(hand0=(cid("Fool"),), stack=(sc("Queen"),), kings=(False, False))
    assert FLIP_KING in nonempty.legal_moves()
    used = mainstate(hand0=(cid("Fool"),), stack=(sc("Queen"),), kings=(True, False))
    assert FLIP_KING not in used.legal_moves()
