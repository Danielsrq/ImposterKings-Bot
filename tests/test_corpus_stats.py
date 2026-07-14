"""Corpus statistics: the replay primitive, the combo enumerator, and the engine predicates.

The statistics are only as trustworthy as the claims they encode about the ENGINE, so the tests here are
mostly about those claims -- built from `scenario.py` positions rather than a corpus, so they hold on a
clean clone and pin behaviour rather than data.

The one that matters most is `test_a_desynced_game_is_caught_not_silently_replayed`: `GameState.apply`
does NOT validate legality, so a corpus recorded under older rules replays into garbage instead of raising.
`build_tables.py` swallowed 446 such games in a bare `except` and published tables built from 1,554 of
2,000 games with nothing on screen to say so. That is the failure this module exists to not repeat.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from imposterkings import cards, scenario
from imposterkings.actions import Action, ActionKind, StepKind
from imposterkings.data_analysis.corpus_stats import (Stats, override_fired, wilson, win_reason)
from imposterkings.data_analysis.replay import replay
from imposterkings.record import action_to_dict
from imposterkings.state import GameState

QUEEN, PRINCESS, KINGSHAND, WARLORD = 0, 1, 2, 4
ELDER_A, ELDER_B, ASSASSIN, FOOL = 14, 15, 16, 17


def _rec(seed=0, decisions=None, winner=0):
    st = GameState.deal(np.random.default_rng(seed))
    return {"deal_seed": seed, "starting_player": st.starting_player, "winner": winner,
            "rewards": [1.0, -1.0] if winner == 0 else [-1.0, 1.0],
            "gen": {"spec": "test", "self_play": True}, "decisions": decisions or []}


def _legal_playthrough(seed=0):
    """A real game, always taking the first legal move -- the ground truth a replay must reproduce."""
    st = GameState.deal(np.random.default_rng(seed))
    decisions = []
    while not st.is_terminal():
        mv = st.legal_moves()[0]
        decisions.append({"seat": st.to_play, "phase": st.phase.name,
                          "chosen": action_to_dict(mv), "candidates": []})
        st = st.apply(mv)
    return st, decisions


# --- the replay primitive ------------------------------------------------------------------------------

def test_replay_reproduces_the_recorded_game_exactly():
    """(deal_seed, chosen actions) fully determines a game: deal is seeded and apply has no rng."""
    term, decisions = _legal_playthrough(3)
    g = replay(_rec(3, decisions, winner=term.winner))

    assert not g.desynced
    assert g.terminal.winner == term.winner == g.winner
    assert len(g.plies) == len(decisions)
    assert g.plies[0].state.hands == g.dealt.hands           # ply 0 sees the untouched deal
    assert g.terminal.hands == term.hands


def test_a_desynced_game_is_caught_not_silently_replayed():
    """THE canary. apply() dispatches on pending[-1].kind without checking legality, so an illegal action
    mutates the state instead of raising -- exactly how a pre-rules-change corpus decays into garbage."""
    st = GameState.deal(np.random.default_rng(0))
    illegal = next(a for a in (Action(ActionKind.HIDE_CARD, card=c) for c in range(18))
                   if a not in st.legal_moves())
    g = replay(_rec(0, [{"seat": st.to_play, "phase": st.phase.name,
                         "chosen": action_to_dict(illegal), "candidates": []}]))
    assert g.desynced and g.desync_at == 0 and g.terminal is None


def test_the_ply_carries_the_state_BEFORE_the_action():
    """Every engine predicate we ask (royalty_present, active_royalty, "does the defender hold X") is a
    question about the position the mover FACED. A post-apply state would answer a different question."""
    term, decisions = _legal_playthrough(5)
    g = replay(_rec(5, decisions, winner=term.winner))
    for p in g.plies:
        assert p.action in p.state.legal_moves()             # the action was legal in the state we hand back
        assert p.state.to_play == p.seat or p.state.pending[-1].actor == p.seat


def test_preferred_is_the_argmax_visit_share_not_the_played_move():
    """datagen ran at temp_plies 6 and setup is plies 1-2, so `chosen` is a SAMPLE from the visit
    distribution. Reading it as the policy would misreport what the bot actually wants to hide."""
    st = GameState.deal(np.random.default_rng(0))
    played, wanted = st.legal_moves()[0], st.legal_moves()[1]
    g = replay(_rec(0, [{"seat": st.to_play, "phase": st.phase.name, "chosen": action_to_dict(played),
                         "candidates": [
                             {"move": action_to_dict(played), "visit_share": 0.2, "mean_q": 0.0},
                             {"move": action_to_dict(wanted), "visit_share": 0.8, "mean_q": 0.1}]}]))
    assert g.plies[0].action == played
    assert g.plies[0].preferred()["move"] == action_to_dict(wanted)


# --- the combo enumerator -------------------------------------------------------------------------------

def test_a_duplicate_card_is_not_double_counted():
    """A hand with both Elders offers two ways to pick the combo ('Elder',). Counting both would inflate
    every duplicate's denominator and silently drag its winrate toward 50%."""
    s = Stats("t", max_combo=2)
    keys = s._combos([QUEEN, PRINCESS, ELDER_A, ELDER_B])

    assert keys.count(("Elder",)) == 1                        # NOT 2, despite two Elder instances
    assert ("Elder", "Elder") in keys                         # ...but the PAIR is a real, distinct combo
    assert ("Princess", "Queen") in keys
    assert len(keys) == len(set(keys))
    assert all(len(k) <= 2 for k in keys)                     # max_combo respected


def test_the_combo_baseline_is_exactly_50_percent():
    """The invariant that proves the row unit is (game, seat) and not (game): every game has exactly one
    winning seat, so pooled over both seats the base winrate is 50% and a combo's distance from 50% IS
    its value. If this drifts, the unit is wrong and every combo number is meaningless."""
    s = Stats("t")
    for seed in range(40):
        term, decisions = _legal_playthrough(seed)
        s.add(replay(_rec(seed, decisions, winner=term.winner)))

    # THE invariant, read straight off the turn-order table (which counts every seat of every game once):
    rows = sum(n for n, _ in s.turn.values())
    wins = sum(w for _, w in s.turn.values())
    assert rows == 2 * s.games, "a hand-row must be (game, seat) -- two per game"
    assert wins == s.games, "exactly one seat wins each game, so the pooled base winrate is 50%"
    assert wins / rows == 0.5

    for (n, w, nf, wf) in s.combo.values():                    # no combo can win more often than it occurs
        assert 0 <= w <= n and 0 <= wf <= nf <= n


# --- the engine predicates ------------------------------------------------------------------------------

def test_warlord_fires_only_when_royalty_is_actually_on_the_stack():
    """abilities.py:139 -- `override = 9 if (ability == WARLORD and state.royalty_present())`. "Warlord
    played" is NOT "Warlord bonus applied", and a table that conflates them is simply wrong."""
    st = scenario.build(hand0=["Warlord", "Soldier"], hand1=["Elder", "Fool"], stack=["Queen"],
                        turn_player=0)
    assert st.royalty_present()

    plain = scenario.build(hand0=["Warlord", "Soldier"], hand1=["Elder", "Fool"], stack=["Soldier"],
                           turn_player=0)
    assert not plain.royalty_present()


def test_override_fired_names_the_passive_that_made_the_play_legal():
    """A passive has no decision window, so 'was it used' can only mean 'did its rule change what was
    legal'. Elder (a 3) onto an active Queen (a 9) is only playable BECAUSE it is an Elder."""
    st = scenario.build(hand0=["Elder", "Soldier"], hand1=["Fool", "Zealot"], stack=["Queen"],
                        turn_player=0)
    elder = next(c for c in st.hands[0] if cards.card_name(c) == "Elder")

    assert st.effective_hand_value(elder) < st.leading_value()      # a 3 cannot beat a 9...
    assert Action(ActionKind.PLAY_CARD, card=elder) in st.legal_moves()   # ...yet it is legal
    assert override_fired(st, 0, elder, ascended=False) == "elder"  # because of the Elder override

    soldier = next(c for c in st.hands[0] if cards.card_name(c) == "Soldier")
    assert override_fired(st, 0, soldier, ascended=False) is None   # a 5 under a 9: no override, just illegal


def test_override_fired_is_none_when_the_card_beats_the_throne_honestly():
    st = scenario.build(hand0=["Queen", "Soldier"], hand1=["Fool", "Elder"], stack=["Fool"], turn_player=0)
    queen = next(c for c in st.hands[0] if cards.card_name(c) == "Queen")
    assert st.effective_hand_value(queen) >= st.leading_value()
    assert override_fired(st, 0, queen, ascended=False) is None


def test_win_reason_separates_the_assassin_kill_from_a_stall_out():
    """The engine has exactly TWO winner= paths (state.py:238 stall-out, abilities.py:456 assassin), so the
    taxonomy is complete rather than heuristic. The assassin signature is the last ply, not the winner."""
    term, decisions = _legal_playthrough(3)
    g = replay(_rec(3, decisions, winner=term.winner))
    assert win_reason(g).startswith("stalled") or win_reason(g) in (
        "assassin_kills_king", "kingshand_counter_stalls_them")

    last = g.plies[-1]
    is_assassin = (last.phase == StepKind.REACTION_KH_VS_ASSASSIN
                   and last.action.kind == ActionKind.DECLINE_REACTION)
    assert (win_reason(g) == "assassin_kills_king") == is_assassin


# --- the interval ---------------------------------------------------------------------------------------

def test_combo_lift_is_measured_against_the_SEAT_baseline_not_against_50_percent():
    """50% is the wrong null once turn order exists. P1 wins ~41% of hand-rows and P2 ~59%, so a combo that
    wins 55% is +14pp for P1 but MINUS 4pp for P2 -- pooling them reports '55%, above average' and hides the
    fact that it is a bad hand for one of the two seats."""
    from imposterkings.data_analysis.corpus_stats import _rows_combo

    s = Stats("t")
    s.games = 100
    s.turn[("bot", True)] = [100, 40]              # P1: 40% baseline
    s.turn[("bot", False)] = [100, 60]             # P2: 60% baseline
    s.combo[("Queen",)] = [40, 22, 20, 11]         # 55% pooled; 55% as P1 AND 55% as P2

    r = next(x for x in _rows_combo(s) if x["combo"] == "Queen")
    assert r["winrate"] == 0.55
    assert r["first_wr"] == 0.55 and r["second_wr"] == 0.55
    assert r["lift_first"] == pytest.approx(0.15)    # +15pp against P1's 40% base -- a good hand for him
    assert r["lift_second"] == pytest.approx(-0.05)  # -5pp against P2's 60% base -- a BAD one for her
    assert r["interaction"] == pytest.approx(0.20)   # ...and the seats disagree, which pooling would erase


def test_stats_json_accumulates_corpora_instead_of_clobbering_them(tmp_path, monkeypatch):
    """stats.json is the COLLATION across every corpus -- but corpora are analysed one command at a time,
    so a plain overwrite leaves a 'collated' file holding only whatever ran last. (It did: three runs went
    in, and the file came out with one group.) A later run must ADD to it, and re-running one corpus must
    refresh that corpus without evicting the others."""
    from imposterkings.data_analysis import corpus_stats as cs

    calls = {"n": 0}

    def fake_replay_all(dirs, limit=None, progress=True):
        calls["n"] += 1
        for seed in range(6):
            term, decisions = _legal_playthrough(seed)
            yield replay(_rec(seed, decisions, winner=term.winner))

    monkeypatch.setattr(cs, "replay_all", fake_replay_all)
    out = str(tmp_path / "cs")

    cs.main(["--data", "datasets/alpha", "--out-dir", out, "--no-plot"])
    cs.main(["--data", "datasets/beta", "--out-dir", out, "--no-plot"])

    with open(os.path.join(out, "stats.json"), encoding="utf-8") as fh:
        doc = json.load(fh)
    assert set(doc) == {"alpha", "beta"}, "the second run evicted the first -- stats.json is not a collation"
    assert doc["alpha"]["games"] == 6 and doc["alpha"]["tables"]["turn_order"]

    cs.main(["--data", "datasets/alpha", "--out-dir", out, "--no-plot"])       # re-running refreshes in place
    with open(os.path.join(out, "stats.json"), encoding="utf-8") as fh:
        doc = json.load(fh)
    assert set(doc) == {"alpha", "beta"}


def test_wilson_stays_inside_the_unit_interval_at_small_n():
    """Why Wilson and not the normal interval: a 4-card combo lands in ~2% of hands, so n is small and the
    textbook interval runs off the end of [0,1] -- reporting a winrate range that includes 105%."""
    p, lo, hi = wilson(5, 5)                       # 5/5: the normal interval would give [1.0, 1.0]
    assert p == 1.0 and lo < 1.0 and hi <= 1.0     # Wilson admits the uncertainty

    p, lo, hi = wilson(0, 4)
    assert p == 0.0 and lo >= 0.0 and hi > 0.0

    p, lo, hi = wilson(500, 1000)                  # at large n it converges on the normal interval
    assert p == 0.5 and abs(lo - 0.469) < 0.005 and abs(hi - 0.531) < 0.005
    assert wilson(0, 0)[0] != wilson(0, 0)[0]      # n=0 -> nan, not a ZeroDivisionError
