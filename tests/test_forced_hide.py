"""The forced-hide intervention, and the deal contract it leans on.

The experiment asks a causal question the corpus cannot: the bot picked "hide the Assassin" as its argmax
**0 times in 4,000 setup decisions**, but the threat that would dodge is real (an Inquisitor steals the
Assassin out of a hand in 20.3% of games) and the corpus only ever hid it 64 times by temperature accident,
at 54.7% [42.6, 66.3] against a 59.5% baseline -- an interval that contains the baseline. So we FORCE the
hide and play it out, paired on the deal and mirrored on the starter.

The load-bearing test here is ``test_passing_the_recorded_starter_is_a_no_op_for_an_ordinary_deal``. To
mirror a deal we must pass ``starting_player`` explicitly, and five sites re-derive games with
``GameState.deal(rng(seed))``. If passing the recorded starter perturbed the deal, EVERY corpus would decode
to different hands -- and ``token_dataset``'s canary DROPS desynced games silently rather than raising, so
4,500 recorded games would quietly empty instead of failing loudly.
"""
from __future__ import annotations

import numpy as np
import pytest

from imposterkings import cards
from imposterkings.actions import ActionKind, StepKind
from imposterkings.data_analysis.forced_hide import (ASSASSIN, ForceHideAgent, assassin_seat, report)
from imposterkings.data_analysis.replay import replay
from imposterkings.state import GameState


def test_passing_the_recorded_starter_is_a_no_op_for_an_ordinary_deal():
    """THE deal contract. deal() draws the 18-card permutation FIRST and flips the starter coin SECOND, so
    re-dealing with that same starter reproduces the identical hands. This is what lets the mirrored corpus
    replay while leaving all 4,500 existing recorded games byte-identical."""
    for seed in range(300):
        a = GameState.deal(np.random.default_rng(seed))                                  # the coin flip
        b = GameState.deal(np.random.default_rng(seed), starting_player=a.starting_player)
        assert a.hands == b.hands
        assert a.leftover_faceup == b.leftover_faceup and a.leftover_facedown == b.leftover_facedown
        assert a.starting_player == b.starting_player and a.pending == b.pending


def test_forcing_the_starter_keeps_the_hands_and_only_moves_the_tempo():
    """The mirror: the SAME deal played with either seat first. Hands must not move, or the two mirrored
    games are not the same experiment and the turn-order control is worthless."""
    base = GameState.deal(np.random.default_rng(7))
    a = GameState.deal(np.random.default_rng(7), starting_player=0)
    b = GameState.deal(np.random.default_rng(7), starting_player=1)
    assert a.hands == b.hands == base.hands            # identical cards...
    assert a.starting_player == 0 and b.starting_player == 1     # ...only the tempo differs
    assert a.pending != b.pending                      # (setup resolves starter-first, so the queue flips)


def test_the_forced_agent_hides_the_assassin_but_still_records_what_it_WANTED():
    """We override the played move, not the search. The recorded candidates must still be the bot's real
    preference -- that is how we learn it wanted something else, and it is what makes the corpus trainable."""
    from imposterkings.agents import MCTSAgent

    seed = next(s for s in range(200) if assassin_seat(s) is not None)
    seat = assassin_seat(seed)
    st = GameState.deal(np.random.default_rng(seed))
    while st.to_play != seat or st.phase != StepKind.SETUP_HIDE:
        st = st.apply(st.legal_moves()[0])

    inner = MCTSAgent(iterations=60)
    agent = ForceHideAgent(inner, ASSASSIN)
    view = st.information_set(seat)
    move = agent.select_move(view, np.random.default_rng(0))

    assert move.kind == ActionKind.HIDE_CARD and move.card == ASSASSIN     # we forced it
    assert agent.forced == 1
    assert agent.last_result is not None and len(agent.last_result.stats) > 1   # ...and the search still ran
    # the search's own preference is preserved, and it is (almost certainly) NOT the Assassin
    best = max(agent.last_result.stats, key=lambda s: s.visits)
    assert best.move in view.legal_moves()


def test_the_forced_agent_leaves_every_other_decision_alone():
    """Minimal intervention: only SETUP_HIDE is overridden. If it also touched the discard or the play, the
    measured effect would not be attributable to the hide."""
    from imposterkings.agents import MCTSAgent

    seed = next(s for s in range(200) if assassin_seat(s) is not None)
    seat = assassin_seat(seed)
    st = GameState.deal(np.random.default_rng(seed))
    agent = ForceHideAgent(MCTSAgent(iterations=40), ASSASSIN)

    # walk to this seat's SETUP_DISCARD (past its hide) and check the agent does not override there
    st = st.apply(GameState.deal(np.random.default_rng(seed)).legal_moves()[0]) if st.to_play != seat else st
    while not (st.to_play == seat and st.phase == StepKind.SETUP_DISCARD):
        if st.to_play == seat and st.phase == StepKind.SETUP_HIDE:
            st = st.apply(next(m for m in st.legal_moves() if m.card == ASSASSIN))
            continue
        st = st.apply(st.legal_moves()[0])

    view = st.information_set(seat)
    move = agent.select_move(view, np.random.default_rng(0))
    assert move.kind == ActionKind.DISCARD_CARD
    assert agent.forced == 0                       # the discard was NOT forced -- the bot chose it


def test_assassin_seat_finds_the_holder_and_admits_when_nobody_holds_it():
    """The Assassin is a singleton (id 16), so at most one seat holds it -- and 2/18 of the time it sits in
    the leftovers and nobody does. Those deals must be SKIPPED, not silently attributed to seat 0."""
    seats = [assassin_seat(s) for s in range(400)]
    assert set(seats) <= {0, 1, None}
    held = [s for s in seats if s is not None]
    assert len(held) / len(seats) > 0.8            # ~16/18 = 89% of deals put it in a hand
    assert seats.count(None) > 0                   # ...and the leftover case really happens

    seed = next(s for s in range(400) if assassin_seat(s) is not None)
    st = GameState.deal(np.random.default_rng(seed))
    seat = assassin_seat(seed)
    assert ASSASSIN in st.hands[seat] and ASSASSIN not in st.hands[1 - seat]
    assert cards.card_name(ASSASSIN) == "Assassin"


def test_the_report_pairs_within_a_deal_and_uses_mcnemar():
    """Unpaired, ~75% of the outcome variance is deal luck and it swamps the effect. Paired, only the deals
    where the two arms DISAGREE carry information -- which is exactly McNemar's test."""
    rows = []
    for seed in range(50):                          # forced wins every deal, free loses every deal
        for starter in (0, 1):
            rows.append({"seed": seed, "starter": starter, "arm": "forced", "holder_won": 1})
            rows.append({"seed": seed, "starter": starter, "arm": "free", "holder_won": 0})
    r = report(rows)
    assert r["n"] == 100                            # 50 deals x 2 starters = 100 pairs
    assert r["forced"] == 1.0 and r["free"] == 0.0
    assert r["diff"] == pytest.approx(1.0)
    assert r["z"] > 1.96                            # a total sweep is unambiguously significant

    tied = [{"seed": s, "starter": 0, "arm": a, "holder_won": 1}          # both arms win every deal
            for s in range(40) for a in ("forced", "free")]
    r2 = report(tied)
    assert r2["diff"] == 0.0 and r2["z"] == 0.0     # no disagreement -> no evidence, and no divide-by-zero
