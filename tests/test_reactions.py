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
    # (owner keeps a second card: Princess with an empty hand no longer opens the ability window)
    st = mainstate(hand0=(cid("Princess"), cid("Fool")), hand1=(cid("KingsHand"),),
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


# --- King's Hand negates the ability -> the interaction is undone, the ACTIVE player replays ----------
# (Both cards expended, leading reverts, turn RETURNS to the player who was countered.)

from imposterkings import scenario as sb  # noqa: E402


def test_kingshand_vs_ability_returns_turn_to_active_player():
    # Leading Warlord=7; P1 plays Sentry(8) and declares; P0 counters with King's Hand.
    st = sb.build(hand0=["KingsHand"], hand1=["Sentry", "Queen"], stack=["Warlord"], turn_player=1)
    st = run(st, sb.play_card(cid("Sentry")), DECLARE)
    assert st.phase == StepKind.REACTION_KINGSHAND and st.to_play == 0
    st = run(st, REVEAL_KINGSHAND)
    # As though the exchange never happened: Sentry gone, leading back to 7, and it's P1's turn again.
    assert st.to_play == 1 and st.turn_player == 1 and st.phase == StepKind.MAIN
    assert st.leading_value() == 7 and all(cards.card_name(s.card) != "Sentry" for s in st.stack)
    assert cid("Sentry") in st.discard and cid("KingsHand") in st.discard
    assert cid("KingsHand") not in st.hands[0]


def test_kingshand_vs_oathbound_followup_reverts_to_oathbound():
    # P0 plays Oathbound over Sentry(8) (disgraces it, Oathbound live at 6), then the free follow-up
    # Inquisitor and guesses; P1 counters with King's Hand -> follow-up undone, leading = Oathbound(6).
    st = sb.build(hand0=["Oathbound", "Inquisitor", "Queen"], hand1=["Elder", "KingsHand"],
                  stack=["Sentry"], turn_player=0)
    st = run(st, sb.play_card(cid("Oathbound")))
    assert st.phase == StepKind.OATHBOUND_SECOND
    st = run(st, sb.play_card(cid("Inquisitor")), sb.guess("Elder"))
    assert st.phase == StepKind.REACTION_KINGSHAND and st.to_play == 1
    st = run(st, REVEAL_KINGSHAND)
    assert st.to_play == 0 and st.phase == StepKind.MAIN
    assert cards.card_name(st.leading.card) == "Oathbound" and st.leading_value() == 6
    assert cid("Inquisitor") in st.discard and cid("KingsHand") in st.discard
    # P0 resumes a NORMAL turn: must beat 6 (the free any-value follow-up is NOT re-granted).
    plays = [m.card for m in st.legal_moves() if m.kind == ActionKind.PLAY_CARD]
    assert plays and all(cards.card_value(c) >= 6 for c in plays)


def test_kingshand_declined_still_resolves_ability_and_passes_turn():
    # P0 plays Inquisitor over Elder(3), guesses Soldier (P1 holds it); P1 DECLINES the counter.
    # Inquisitor's effect applies and the turn passes to the OPPONENT (decline path unchanged by the fix).
    st = sb.build(hand0=["Inquisitor", "Queen"], hand1=["Soldier", "KingsHand"],
                  stack=["Elder"], turn_player=0)
    st = run(st, sb.play_card(cid("Inquisitor")), sb.guess("Soldier"))
    assert st.phase == StepKind.REACTION_KINGSHAND and st.to_play == 1
    st = run(st, DECLINE_REACTION)
    assert cid("Inquisitor") not in st.discard                # not countered -> Inquisitor stays in play
    assert st.turn_player == 1                                # turn passed to the opponent, not back to P0
    assert cid("Soldier") in st.antechambers[1]              # Inquisitor's effect applied


def test_kingshand_counter_forcing_loss_when_active_player_cannot_replay():
    # Leading Queen=9; P1 plays Princess(9, ties/plays as royalty over royalty) and declares; P0 counters.
    # After Princess is removed, leading reverts to Queen(9); P1 has only Fool(1) and no king -> P1 loses.
    st = sb.build(hand0=["KingsHand"], hand1=["Princess", "Fool"], stack=["Queen"], turn_player=1,
                  kings=(False, True))                         # P1's king already flipped -> no flip out
    st = run(st, sb.play_card(cid("Princess")), DECLARE)
    assert st.phase == StepKind.REACTION_KINGSHAND
    st = run(st, REVEAL_KINGSHAND)
    assert st.is_terminal() and st.winner == 0                # P1 can't beat Queen(9) and can't flip


def test_kingshand_removed_cards_leave_the_determinization_pool():
    import numpy as np
    st = sb.build(hand0=["Oathbound", "Inquisitor", "Queen"], hand1=["Elder", "KingsHand"],
                  stack=["Sentry"], turn_player=0)
    st = run(st, sb.play_card(cid("Oathbound")), sb.play_card(cid("Inquisitor")),
             sb.guess("Elder"), REVEAL_KINGSHAND)
    gone = {cid("Inquisitor"), cid("KingsHand")}
    for seat in (0, 1):
        iv = st.information_set(seat)
        assert gone <= set(iv.discard)                        # both are public discard
        assert not (gone & set(iv.unknown_cards()))           # excluded from the sampling pool
        rng = np.random.default_rng(3)
        for _ in range(100):                                  # never sampled into a hand/hidden/muck
            d = iv.determinize(rng)
            placed = (set(d.hands[0]) | set(d.hands[1])
                      | {x for x in d.hidden if x is not None}
                      | {x for x in d.setup_discard if x is not None})
            assert not (gone & placed)
