"""Optional on-play abilities: Mystic, Sentry, Soldier, Judge, Inquisitor, Fool, Princess.

Each is driven through its full chain: declare the "may", opponent declines the King's-Hand window
(opponents below hold no King's Hand), then the ability's own sub-decisions resolve.
"""
from __future__ import annotations

from imposterkings import cards
from imposterkings.actions import (
    Action, ActionKind, DECLARE, DECLINE_REACTION, REVEAL_KINGSHAND, STOP, StepKind,
)
from imposterkings.state import StackCard

from .helpers import mainstate, run, cid, sc


def _play(c):
    return Action(ActionKind.PLAY_CARD, card=c)


def _guess(name):
    return Action(ActionKind.GUESS_CARD, name=name)


def _hand(c):
    return Action(ActionKind.CHOOSE_HAND_CARD, card=c)


def _target(i):
    return Action(ActionKind.CHOOSE_STACK_TARGET, target=i)


def _num(n):
    return Action(ActionKind.CHOOSE_NUMBER, number=n)


def test_flattened_kingshand_reacts_after_the_declared_parameter():
    # Mystic: declaring "mute 7" goes straight to the King's-Hand window (no separate ABILITY_NUMBER step),
    # and revealing King's Hand there discards the Mystic and cancels the mute.
    st = mainstate(hand0=(cid("Mystic"),), hand1=(cid("KingsHand"),), stack=(sc("Warlord"),))
    st = run(st, _play(cid("Mystic")), _num(7))
    assert st.phase == StepKind.REACTION_KINGSHAND        # window is AFTER the number
    st = run(st, REVEAL_KINGSHAND)
    assert all(cards.card_name(s.card) != "Mystic" for s in st.stack)  # countered off the throne
    assert 7 not in st.muted_values                       # mute never applied


def test_mystic_mutes_retroactively_and_only_offers_1_to_8():
    # Warlord sits beneath; muting 7 should retroactively drop it to value 3.
    st = mainstate(hand0=(cid("Mystic"),), hand1=(cid("Queen"),),
                   stack=(sc("Warlord"), sc("Fool")))
    st = run(st, _play(cid("Mystic")))
    assert st.phase == StepKind.ABILITY_MAY               # declare+number are one decision now
    numbers = sorted(m.number for m in st.legal_moves() if m.kind == ActionKind.CHOOSE_NUMBER)
    assert numbers == list(range(1, 9))  # 9 (Queen/Princess) can never be targeted
    # declaring the mute value opens the King's-Hand window (which sees the number); opponent declines
    st = run(st, _num(7), DECLINE_REACTION)
    warlord_sc = next(s for s in st.stack if cards.card_name(s.card) == "Warlord")
    assert st.effective_stack_value(warlord_sc) == 3
    assert 7 in st.muted_values
    mystic_sc = next(s for s in st.stack if cards.card_name(s.card) == "Mystic")
    assert mystic_sc.disgraced  # Mystic self-disgraced


def test_sentry_swaps_exact_position_not_top():
    # Stack: Soldier(pos0), Warlord(pos1 leading). Sentry played -> leads at pos2.
    st = mainstate(hand0=(cid("Sentry"), cid("Fool")), hand1=(cid("Queen"),),
                   stack=(sc("Soldier"), sc("Warlord")))
    st = run(st, _play(cid("Sentry")), DECLARE, DECLINE_REACTION)
    assert st.phase == StepKind.ABILITY_STACK_TARGET
    # Grab the Soldier at position 0 and put our Fool in its exact place.
    st = run(st, _target(0), _hand(cid("Fool")))
    assert cards.card_name(st.stack[0].card) == "Fool"           # exact position replaced
    assert cards.card_name(st.stack[1].card) == "Warlord"        # leading unchanged
    assert cid("Soldier") in st.hands[0]                          # grabbed card returned to hand
    assert next(s for s in st.stack if cards.card_name(s.card) == "Sentry").disgraced


def test_soldier_guess_is_mandatory_no_decline():
    # Playing a Soldier goes straight to the guess -- there is no declare/decline step.
    st = mainstate(hand0=(cid("Soldier"),), hand1=(cid("Warlord"),), stack=(sc("Fool"),))
    st = run(st, _play(cid("Soldier")))
    assert st.phase == StepKind.ABILITY_GUESS
    kinds = {m.kind for m in st.legal_moves()}
    assert kinds == {ActionKind.GUESS_CARD}                        # only guesses; no DECLARE/DECLINE


def test_soldier_correct_guess_grants_plus_two_immediately_then_disgrace():
    st = mainstate(hand0=(cid("Soldier"),), hand1=(cid("Warlord"),), stack=(sc("Fool"),))
    # The guess is mandatory + made public; the defender (no King's Hand here) then gets the counter window.
    st = run(st, _play(cid("Soldier")), _guess("Warlord"))
    assert st.phase == StepKind.REACTION_KINGSHAND
    st = run(st, DECLINE_REACTION)
    # +2 is applied automatically on the correct guess; we go straight to the disgrace choice.
    assert st.phase == StepKind.ABILITY_STACK_TARGET
    soldier_sc = next(s for s in st.stack if cards.card_name(s.card) == "Soldier")
    assert st.effective_stack_value(soldier_sc) == 7  # +2, no extra "take the package" step
    # Disgrace the Fool at position 0, then stop.
    st = run(st, _target(0), STOP)
    assert next(s for s in st.stack if cards.card_name(s.card) == "Fool").disgraced


def test_soldier_plus_two_holds_even_when_disgracing_nothing():
    st = mainstate(hand0=(cid("Soldier"),), hand1=(cid("Warlord"),), stack=(sc("Fool"),))
    st = run(st, _play(cid("Soldier")), _guess("Warlord"), DECLINE_REACTION, STOP)
    soldier_sc = next(s for s in st.stack if cards.card_name(s.card) == "Soldier")
    assert st.effective_stack_value(soldier_sc) == 7  # +2 kept even with 0 cards disgraced
    assert not next(s for s in st.stack if cards.card_name(s.card) == "Fool").disgraced


def test_soldier_wrong_guess_offers_no_counter_and_does_nothing():
    st = mainstate(hand0=(cid("Soldier"),), hand1=(cid("Warlord"),), stack=(sc("Fool"),))
    st = run(st, _play(cid("Soldier")), _guess("Queen"))  # wrong -> no window, no effect
    soldier_sc = next(s for s in st.stack if cards.card_name(s.card) == "Soldier")
    assert st.effective_stack_value(soldier_sc) == 5  # no +2
    assert not next(s for s in st.stack if cards.card_name(s.card) == "Fool").disgraced


def test_judge_guess_is_mandatory_no_decline():
    # Playing a Judge goes straight to the guess -- like Soldier, there is no declare/decline step.
    st = mainstate(hand0=(cid("Judge"), cid("Fool")), hand1=(cid("Warlord"),), stack=(sc("Elder"),))
    st = run(st, _play(cid("Judge")))
    assert st.phase == StepKind.ABILITY_GUESS
    assert {m.kind for m in st.legal_moves()} == {ActionKind.GUESS_CARD}


def test_judge_correct_guess_queues_to_own_antechamber():
    st = mainstate(hand0=(cid("Judge"), cid("Queen")), hand1=(cid("Warlord"),), stack=(sc("Elder"),))
    st = run(st, _play(cid("Judge")), _guess("Warlord"), DECLINE_REACTION)
    assert st.phase == StepKind.ABILITY_HAND_CARD
    assert STOP in st.legal_moves()  # may decline to queue
    st = run(st, _hand(cid("Queen")))
    assert st.antechambers[0] == (cid("Queen"),)
    assert cid("Queen") not in st.hands[0]


def test_judge_cannot_schedule_base_value_1_fool():
    # The Judge may only queue a card with BASE value >= 2, so the Fool (value 1) is never offered --
    # not even the Judge+Fool "force my own Fool to ascend next turn" combo.
    st = mainstate(hand0=(cid("Judge"), cid("Fool"), cid("Queen")), hand1=(cid("Warlord"),),
                   stack=(sc("Elder"),))
    st = run(st, _play(cid("Judge")), _guess("Warlord"), DECLINE_REACTION)
    offered = {cards.card_name(m.card) for m in st.legal_moves() if m.card is not None}
    assert "Fool" not in offered and "Queen" in offered
    assert STOP in st.legal_moves()
    # base value is immutable: muting value 1 (Fool plays as 3) does NOT make it schedulable
    stm = mainstate(hand0=(cid("Judge"), cid("Fool"), cid("Queen")), hand1=(cid("Warlord"),),
                    stack=(sc("Elder"),), muted={1})
    stm = run(stm, _play(cid("Judge")), _guess("Warlord"), DECLINE_REACTION)
    assert "Fool" not in {cards.card_name(m.card) for m in stm.legal_moves() if m.card is not None}
    # with only a Fool to schedule, the player must decline (STOP only)
    sto = mainstate(hand0=(cid("Judge"), cid("Fool")), hand1=(cid("Warlord"),), stack=(sc("Elder"),))
    sto = run(sto, _play(cid("Judge")), _guess("Warlord"), DECLINE_REACTION)
    assert sto.legal_moves() == [STOP]


def test_inquisitor_forces_opponent_card_into_opponent_antechamber():
    # Player1 holds the Warlord; naming it (after surviving player1's counter window) forces it into
    # player1's antechamber, where it ascends on player1's turn, passing control back to player0.
    st = mainstate(hand0=(cid("Inquisitor"),), hand1=(cid("Warlord"), cid("Fool")),
                   stack=(sc("Zealot"),))
    st = run(st, _play(cid("Inquisitor")), _guess("Warlord"), DECLINE_REACTION)  # guess is the declaration
    assert cid("Warlord") not in st.hands[1]
    # Warlord is queued into P1's antechamber; its ascension is now P1's own forced turn.
    assert st.phase == StepKind.ASCEND and st.to_play == 1
    st = st.apply(st.legal_moves()[0])
    assert cards.card_name(st.stack[-1].card) == "Warlord"  # ascended to lead
    assert st.antechambers[1] == ()
    assert st.to_play == 0  # ascension consumed player1's whole turn


def test_ascension_surfaces_as_its_own_forced_decision():
    # Regression: a queued card's ascension must be a REAL decision (StepKind.ASCEND, single forced
    # PLAY_CARD) so the driver loop records it as its own turn -- not silently auto-resolved in apply().
    st = mainstate(hand0=(cid("Queen"),), hand1=(cid("Fool"),), stack=(sc("Soldier"),),
                   antechambers=((cid("Elder"),), ()))
    st = st._begin_turn(0)
    assert st.to_play == 0 and st.phase == StepKind.ASCEND
    assert st.legal_moves() == [Action(ActionKind.PLAY_CARD, card=cid("Elder"))]
    st = st.apply(st.legal_moves()[0])
    assert cards.card_name(st.stack[-1].card) == "Elder" and st.to_play == 1


def test_fool_takes_a_non_disgraced_card_and_cannot_take_disgraced_or_itself():
    # Leading card is a disgraced Soldier (value 0) so Fool is playable; a live Warlord sits beneath.
    st = mainstate(hand0=(cid("Fool"),), hand1=(cid("Queen"),),
                   stack=(sc("Warlord"), sc("Soldier", disgraced=True)))
    st = run(st, _play(cid("Fool")))
    assert st.phase == StepKind.ABILITY_MAY    # declare+which-card are one decision now
    targets = {m.target for m in st.legal_moves() if m.kind == ActionKind.CHOOSE_STACK_TARGET}
    warlord_pos = next(i for i, s in enumerate(st.stack) if cards.card_name(s.card) == "Warlord")
    soldier_pos = next(i for i, s in enumerate(st.stack) if cards.card_name(s.card) == "Soldier")
    fool_pos = next(i for i, s in enumerate(st.stack) if cards.card_name(s.card) == "Fool")
    assert targets == {warlord_pos}            # only the live Warlord; not the disgraced Soldier or itself
    assert soldier_pos not in targets and fool_pos not in targets
    st = run(st, _target(warlord_pos), DECLINE_REACTION)   # choosing the card opens the window
    assert cid("Warlord") in st.hands[0]
    assert all(cards.card_name(s.card) != "Warlord" for s in st.stack)


def test_princess_swaps_one_card_each():
    st = mainstate(hand0=(cid("Princess"), cid("Fool")), hand1=(cid("Soldier"),),
                   stack=(sc("Elder"),))
    st = run(st, _play(cid("Princess")), DECLARE, DECLINE_REACTION,
             _hand(cid("Fool")), _hand(cid("Soldier")))
    assert cid("Soldier") in st.hands[0]
    assert cid("Fool") in st.hands[1]


def test_princess_no_window_when_owner_has_no_card_left():
    # Princess as the owner's LAST card: a swap needs a card from each side, so the "Use ability?"
    # window is not offered at all -- the play just lands and the turn passes.
    st = mainstate(hand0=(cid("Princess"),), hand1=(cid("Soldier"),), stack=(sc("Elder"),))
    st = run(st, _play(cid("Princess")))
    assert st.pending and st.pending[-1].kind == StepKind.MAIN and st.to_play == 1
    assert st.hands[1] == (cid("Soldier"),)                        # nothing traded


def test_princess_no_window_when_opponent_hand_empty():
    st = mainstate(hand0=(cid("Princess"), cid("Fool")), hand1=(), stack=(sc("Elder"),))
    st = run(st, _play(cid("Princess")))
    assert st.pending and st.pending[-1].kind != StepKind.ABILITY_MAY   # no window; opponent to move
    assert cid("Fool") in st.hands[0]                              # nothing traded


# --- Mystic muting strips abilities EVERYWHERE (not just reactions) ---------------------------------

def test_muted_oathbound_loses_its_override():
    # muted 6s: the Oathbound is a bare value-3 card -- it must NOT play over a bigger card.
    st = mainstate(hand0=(cid("Oathbound"), cid("Fool")), hand1=(cid("Soldier"),),
                   stack=(sc("Warlord"),), muted={6})           # lead 7 > 3, no override
    plays = [a.card for a in st.legal_moves() if a.kind == ActionKind.PLAY_CARD]
    assert cid("Oathbound") not in plays
    # positive control: unmuted, the override applies (hand >= 2)
    st2 = mainstate(hand0=(cid("Oathbound"), cid("Fool")), hand1=(cid("Soldier"),),
                    stack=(sc("Warlord"),))
    assert cid("Oathbound") in [a.card for a in st2.legal_moves() if a.kind == ActionKind.PLAY_CARD]


def test_muted_elder_cannot_play_over_royalty():
    st = mainstate(hand0=(cid("Elder"), cid("Fool")), hand1=(cid("Soldier"),),
                   stack=(sc("Queen"),), muted={3})
    assert cid("Elder") not in [a.card for a in st.legal_moves() if a.kind == ActionKind.PLAY_CARD]


def test_muted_zealot_cannot_play_over_nonroyalty():
    st = mainstate(hand0=(cid("Zealot"), cid("Fool")), hand1=(cid("Soldier"),),
                   stack=(sc("Warlord"),), kings=(True, False), muted={3})
    assert cid("Zealot") not in [a.card for a in st.legal_moves() if a.kind == ActionKind.PLAY_CARD]


def test_muted_soldier_lands_without_guess():
    # muted 5s: Soldier plays as a bare 3 (legal over an Elder lead) and triggers NO guess step.
    st = mainstate(hand0=(cid("Soldier"), cid("Fool")), hand1=(cid("Judge"),),
                   stack=(sc("Elder"),), muted={5})
    st = run(st, _play(cid("Soldier")))
    assert st.pending and st.pending[-1].kind == StepKind.MAIN and st.to_play == 1   # turn just passes


def test_every_muted_card_lands_silently_and_at_value_3():
    # Systematic sweep: EVERY mutable card (base <= 8; 9s can never be muted), played while muted, must
    # land as a bare value-3 card -- no guess, no ability window, no override/follow-up, no value buff
    # (royalty sits beneath so a Warlord WOULD land buffed if muting failed to strip it).
    mutable = [n for n in cards.CARD_NAMES if cards.card_value(cid(n)) <= 8]
    for name in mutable:
        base = cards.card_value(cid(name))
        lead = sc("Zealot") if name == "Fool" else sc("Fool")        # a lead the muted 3 can beat
        filler = cid("Elder", 1) if name == "Elder" else cid("Elder")
        st = mainstate(hand0=(cid(name), filler), hand1=(cid("Judge"),),
                       stack=(sc("Queen"), lead), muted={base})
        st = run(st, _play(cid(name)))
        assert st.pending and st.pending[-1].kind == StepKind.MAIN and st.to_play == 1, \
            f"{name}: muted play must end the turn silently (got {st.pending[-1].kind})"
        assert st.leading_value() == 3, f"{name}: muted card must lead at 3 (got {st.leading_value()})"


def test_muted_card_ascends_silently():
    # A muted Soldier queued in the antechamber: the forced ascension must not trigger the guess.
    st = mainstate(hand0=(cid("Fool"),), hand1=(cid("Elder"),), stack=(sc("Zealot"),),
                   antechambers=((cid("Soldier"),), ()), muted={5}, to_play=1)
    st = run(st, _play(cid("Elder")))                                # P1 plays; P0's turn -> forced ASCEND
    assert st.pending and st.pending[-1].kind == StepKind.ASCEND
    st = run(st, st.legal_moves()[0])
    assert st.pending[-1].kind == StepKind.MAIN and st.to_play == 1  # landed silent; P1 to move
    assert st.leading_value() == 3                                   # ascended muted Soldier leads at 3
