"""Guess-leaked opponent-hand knowledge: recording, pruning, and determinize honoring it.

This is the ISMCTS determinization fix -- the search must not sample opponent hands that contradict
what a guess proved (a wrong Inquisitor/Soldier/Judge name -> the hand lacks it; a landed Soldier/Judge
-> the hand holds it).
"""
from __future__ import annotations

import numpy as np

from imposterkings import cards
from imposterkings.actions import Action, ActionKind, DECLARE, DECLINE_REACTION, StepKind
from imposterkings.cards import card_name
from imposterkings.mcts import SearchConfig, search
from imposterkings.state import GameState, PendingStep

from .helpers import cid, mainstate, run, sc


def _visible_all_except(*names):
    """All card ids except every copy of the given names (used to shrink the observer's unknown pool)."""
    excl = set()
    for n in names:
        excl.update(cards.card_ids_for_name(n))
    return tuple(c for c in range(cards.DECK_SIZE) if c not in excl)


def _play(c):
    return Action(ActionKind.PLAY_CARD, card=c)


def _hand(c):
    return Action(ActionKind.CHOOSE_HAND_CARD, card=c)


def _target(i):
    return Action(ActionKind.CHOOSE_STACK_TARGET, target=i)


def _opp_always_has(view, name, n=50):
    """True if every determinization puts a card named ``name`` in the opponent's hand."""
    opp = 1 - view.observer
    return all(any(card_name(c) == name for c in view.determinize(np.random.default_rng(i)).hands[opp])
               for i in range(n))


def _guessstate(owner: int, source: int, hand0, hand1) -> GameState:
    """A state whose top step is an ABILITY_GUESS by ``owner`` using ``source``."""
    return GameState(
        hands=(tuple(sorted(hand0)), tuple(sorted(hand1))),
        hidden=(None, None), kings=(False, False), antechambers=((), ()),
        stack=(), discard=(), leftover_faceup=-1, leftover_facedown=-1,
        muted_values=frozenset(), turn_player=owner, starting_player=0,
        pending=(PendingStep(StepKind.ABILITY_GUESS, owner, source=source),),
        history=(), winner=None,
    )


def _opp_hand(view, rng, use=True):
    det = view.determinize(rng, use_knowledge=use)
    return det.hands[1 - view.observer]


# --- determinize honors the facts --------------------------------------------------------

def test_determinize_never_samples_a_lacked_name_into_the_hand():
    st = mainstate(hand0=(cid("Soldier"), cid("Elder"), cid("Fool")),
                   hand1=(cid("Mystic"), cid("Warlord"), cid("Judge")),
                   hand_lacks=(frozenset({"Queen"}), frozenset()))
    view = st.information_set(0)
    assert view.opp_hand_lacks == frozenset({"Queen"})
    queen = cid("Queen")
    assert all(queen not in _opp_hand(view, np.random.default_rng(i)) for i in range(200))
    # ...but WITHOUT the knowledge, the uniform sampler does put the Queen in the hand sometimes.
    assert any(queen in _opp_hand(view, np.random.default_rng(i), use=False) for i in range(200))


def test_determinize_always_includes_a_held_name():
    st = mainstate(hand0=(cid("Elder"), cid("Fool"), cid("Zealot")),
                   hand1=(cid("Mystic"), cid("Warlord"), cid("Judge")),
                   hand_has=(frozenset({"Soldier"}), frozenset()))
    view = st.information_set(0)
    for i in range(200):
        opp = _opp_hand(view, np.random.default_rng(i))
        assert any(card_name(c) == "Soldier" for c in opp)


def test_determinize_backcompat_when_no_knowledge():
    st = mainstate(hand0=(cid("Elder"), cid("Fool")), hand1=(cid("Mystic"), cid("Warlord")))
    view = st.information_set(0)
    # No constraints -> the opp hand is a plausibly-varied draw (not a fixed set).
    hands = {_opp_hand(view, np.random.default_rng(i)) for i in range(50)}
    assert len(hands) > 1


# --- recording at the guess sites --------------------------------------------------------

def test_wrong_guess_records_lacks():
    st = _guessstate(0, cid("Soldier"), hand0=(cid("Soldier"),), hand1=(cid("Elder"), cid("Fool")))
    st2 = st.apply(Action(ActionKind.GUESS_CARD, name="Queen"))     # opponent has no Queen
    assert "Queen" in st2.hand_lacks[0] and "Queen" not in st2.hand_has[0]


def test_landed_guess_records_has():
    st = _guessstate(0, cid("Soldier"), hand0=(cid("Soldier"),), hand1=(cid("Queen"), cid("Fool")))
    st2 = st.apply(Action(ActionKind.GUESS_CARD, name="Queen"))     # opponent holds Queen -> KH window
    assert "Queen" in st2.hand_has[0] and "Queen" not in st2.hand_lacks[0]


def test_inquisitor_extract_records_lacks():
    st = _guessstate(0, cid("Inquisitor"),
                     hand0=(cid("Inquisitor"),), hand1=(cid("Elder"), cid("Elder", 1), cid("Fool")))
    st2 = st.apply(Action(ActionKind.GUESS_CARD, name="Elder"))     # held -> King's-Hand window
    st3 = st2.apply(Action(ActionKind.DECLINE_REACTION))            # no KH -> effect applies
    assert "Elder" in st3.hand_lacks[0]                             # all copies extracted
    assert all(card_name(c) != "Elder" for c in st3.hands[1])
    assert any(card_name(c) == "Elder" for c in st3.antechambers[1])


# --- pruning as cards move (reconciliation in with_) -------------------------------------

def test_has_dropped_when_the_card_leaves_the_hand():
    st = mainstate(hand0=(cid("Elder"),), hand1=(cid("Soldier"), cid("Fool")),
                   hand_has=(frozenset({"Soldier"}), frozenset()))
    st2 = st.with_(hands=(st.hands[0], (cid("Fool"),)))            # opp played/lost the Soldier
    assert "Soldier" not in st2.hand_has[0]


def test_lacks_cleared_on_concealed_hidden_pickup():
    hidden1 = cid("Queen")
    st = mainstate(hand0=(cid("Elder"),), hand1=(cid("Fool"),), hidden=(None, hidden1),
                   hand_lacks=(frozenset({"Warlord"}), frozenset()))
    st2 = st.with_(hands=(st.hands[0], tuple(sorted((cid("Fool"), hidden1)))))  # king-flip pickup
    assert st2.hand_lacks[0] == frozenset()


def test_public_pickup_updates_names():
    warlord = cid("Warlord")
    st = mainstate(hand0=(cid("Elder"),), hand1=(cid("Fool"),),
                   hand_lacks=(frozenset({"Warlord"}), frozenset()))
    st2 = st.with_(hands=(st.hands[0], tuple(sorted((cid("Fool"), warlord)))))  # public grab
    assert "Warlord" not in st2.hand_lacks[0] and "Warlord" in st2.hand_has[0]


# --- decisive end-to-end effect on the search ------------------------------------------

def test_search_eval_improves_when_opponent_is_known_to_lack_strong_cards():
    """The definitive test: in a fixed position, an unconstrained search sometimes hands the opponent
    strong cards (KingsHand/Assassin/Queen/Princess) that live in the unknown muck; proving (wrong
    guess) the opponent's hand lacks them raises the observer's evaluated win value substantially."""
    base = dict(hand0=(cid("Soldier"), cid("Mystic"), cid("Warlord")),
                hand1=(cid("Elder"), cid("Zealot"), cid("Fool")),
                stack=(sc("Oathbound"),), to_play=0)
    strong = frozenset({"KingsHand", "Assassin", "Queen", "Princess"})
    st_off = mainstate(**base)
    st_on = mainstate(**base, hand_lacks=(strong, frozenset()))

    def mean_root_value(state, use):
        vals = [search(state.information_set(0),
                       SearchConfig(rng=np.random.default_rng(s), iterations=500, use_knowledge=use))
                .root_value() for s in range(4)]
        return sum(vals) / len(vals)

    v_off = mean_root_value(st_off, use=False)
    v_on = mean_root_value(st_on, use=True)
    assert v_on - v_off > 0.12, f"expected a clear gain from the knowledge, got on={v_on:.3f} off={v_off:.3f}"


# --- non-guess cards that reveal a known card into the opponent's hand feed determinize too ------
# (Sentry/Fool/Princess move a PUBLIC card into a hand -> the generic with_ reconciliation records it
# as a 'has' fact, which determinize then honors. These prove that end-to-end.)

def test_sentry_grab_is_known_and_feeds_determinize():
    # Opponent (seat 1) plays a Sentry and swaps the live Warlord off the stack into hand.
    st = mainstate(hand0=(cid("Queen"),), hand1=(cid("Sentry"), cid("Fool")),
                   stack=(sc("Warlord"),), to_play=1)
    st = run(st, _play(cid("Sentry")), DECLARE, DECLINE_REACTION, _target(0), _hand(cid("Fool")))
    assert cid("Warlord") in st.hands[1] and "Warlord" in st.hand_has[0]
    assert _opp_always_has(st.information_set(0), "Warlord")


def test_fool_takeback_is_known_and_feeds_determinize():
    # Opponent (seat 1) plays a Fool over a disgraced lead and takes the live Warlord back to hand.
    st = mainstate(hand0=(cid("Queen"),), hand1=(cid("Fool"),),
                   stack=(sc("Warlord"), sc("Elder", disgraced=True)), to_play=1)
    st = run(st, _play(cid("Fool")), DECLARE, DECLINE_REACTION, _target(0))
    assert cid("Warlord") in st.hands[1] and "Warlord" in st.hand_has[0]
    assert _opp_always_has(st.information_set(0), "Warlord")


def test_princess_swap_is_known_and_feeds_determinize():
    # Observer (seat 0) plays Princess and gives its Fool to the opponent -> knows the opp now holds it.
    st = mainstate(hand0=(cid("Princess"), cid("Fool")), hand1=(cid("Soldier"),), stack=(sc("Elder"),))
    st = run(st, _play(cid("Princess")), DECLARE, DECLINE_REACTION, _hand(cid("Fool")), _hand(cid("Soldier")))
    assert cid("Fool") in st.hands[1] and "Fool" in st.hand_has[0]
    assert _opp_always_has(st.information_set(0), "Fool")


def test_landed_judge_guess_feeds_determinize():
    # A landed (correct) Judge guess proves the opponent holds the name -> determinize must include it.
    st = _guessstate(0, cid("Judge"), hand0=(cid("Judge"),), hand1=(cid("Warlord"), cid("Fool")))
    st = st.apply(Action(ActionKind.GUESS_CARD, name="Warlord"))
    assert "Warlord" in st.hand_has[0]
    assert _opp_always_has(st.information_set(0), "Warlord")


def test_determinize_never_leaks_own_hidden_card_to_opponent():
    # The observer's own hidden card can never be sampled into the opponent's hand.
    st = mainstate(hand0=(cid("Queen"), cid("Fool")),
                   hand1=(cid("Warlord"), cid("Mystic"), cid("Soldier")),
                   hidden=(cid("Zealot"), None))
    view = st.information_set(0)
    zealot = cid("Zealot")
    assert zealot not in view.unknown_cards()
    assert all(zealot not in view.determinize(np.random.default_rng(i)).hands[1] for i in range(300))


# --- knowledge level: perfect / binary / none -------------------------------------------

def test_knowledge_level_binary_perfect_and_none():
    disc = _visible_all_except("Queen", "Fool")            # only Queen + Fool unknown to observer 0
    v = mainstate(hand0=(), hand1=(cid("Queen"),), discard=disc).information_set(0)
    assert v.opp_hand_count == 1 and set(v.unknown_cards()) == {cid("Queen"), cid("Fool")}
    assert v.possible_opp_hands() == 2 and v.knowledge_level() == "binary"

    # Proving the opponent lacks the Fool leaves only the Queen -> perfect information.
    v2 = mainstate(hand0=(), hand1=(cid("Queen"),), discard=disc,
                   hand_lacks=(frozenset({"Fool"}), frozenset())).information_set(0)
    assert v2.possible_opp_hands() == 1 and v2.knowledge_level() == "perfect"

    # A third unknown card -> more than two possibilities -> no chip.
    v3 = mainstate(hand0=(), hand1=(cid("Queen"),),
                   discard=_visible_all_except("Queen", "Fool", "Zealot")).information_set(0)
    assert v3.possible_opp_hands() >= 3 and v3.knowledge_level() is None
