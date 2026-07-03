"""MCTS sanity (fast) and strength vs Random (slow)."""
from __future__ import annotations

import numpy as np
import pytest

from imposterkings.actions import StepKind
from imposterkings.agents import MCTSAgent, RandomAgent
from imposterkings.arena import run_arena
from imposterkings.mcts import SearchConfig, SearchResult, search
from imposterkings.record import play_and_record
from imposterkings.state import GameState

from .helpers import mainstate, cid, sc


def _dealt_to_main(seed: int) -> GameState:
    rng = np.random.default_rng(seed)
    st = GameState.deal(rng, starting_player=0)
    while st.phase in (StepKind.SETUP_HIDE, StepKind.SETUP_DISCARD):
        st = st.apply(st.legal_moves()[0])
    return st


def test_mcts_returns_a_legal_move_with_stats():
    st = _dealt_to_main(1)
    view = st.information_set(st.to_play)
    agent = MCTSAgent(iterations=50)
    move = agent.select_move(view, np.random.default_rng(0))
    assert move in view.legal_moves()
    assert agent.last_result is not None
    assert sum(s.visits for s in agent.last_result.stats) == 50
    # policy target is a normalized distribution over root moves
    assert abs(sum(agent.last_result.policy_target().values()) - 1.0) < 1e-9


def test_principal_variations():
    st = _dealt_to_main(1)
    view = st.information_set(st.to_play)
    result = search(view, SearchConfig(rng=np.random.default_rng(0), iterations=200))
    assert result.root is not None
    lines = result.principal_variations(top=2, depth=4)
    assert 1 <= len(lines) <= 2
    assert lines[0][0].move == result.best_move              # first line starts with the chosen move
    for line in lines:
        assert 1 <= len(line) <= 4                           # depth respected
        for step in line:
            assert step.player in (0, 1) and step.visits > 0 and -1.0 <= step.mean_q <= 1.0
    # a result with no retained tree yields no lines (and doesn't error)
    empty = SearchResult(info=view, best_move=result.best_move, stats=[], iterations=0, elapsed=0.0)
    assert empty.principal_variations() == []


def test_mcts_evaluates_forced_turns_only_when_asked():
    # A forced turn: antechamber ascension has exactly one legal move. By default the agent skips search
    # (no decision to make); evaluate_forced still returns the forced move but computes a position eval.
    st = mainstate(hand0=(cid("Queen"),), hand1=(cid("Fool"),), stack=(sc("Soldier"),),
                   antechambers=((cid("Elder"),), ()))._begin_turn(0)
    view = st.information_set(0)
    forced = view.legal_moves()
    assert len(forced) == 1

    plain = MCTSAgent(iterations=30)
    assert plain.select_move(view, np.random.default_rng(0)) == forced[0]
    assert plain.last_result is None                        # default: no eval on a forced turn

    evaluated = MCTSAgent(iterations=30, evaluate_forced=True)
    assert evaluated.select_move(view, np.random.default_rng(0)) == forced[0]  # still the forced move
    assert evaluated.last_result is not None               # ...but now it carries a real eval
    assert len(evaluated.last_result.stats) == 1 and -1.0 <= evaluated.last_result.root_value() <= 1.0


def test_play_and_record_backfills_reward():
    rng = np.random.default_rng(2)
    rec = play_and_record([MCTSAgent(iterations=20), RandomAgent()], rng)
    assert rec.winner in (0, 1)
    assert rec.decisions and all(d.z in (-1.0, 1.0) for d in rec.decisions)


@pytest.mark.slow
def test_mcts_beats_random():
    rng = np.random.default_rng(0)
    wins = run_arena([MCTSAgent(iterations=120), RandomAgent()], games=40, rng=rng, swap=True)
    assert wins[0] > wins[1], f"MCTS should beat Random over a seat-swapped arena, got {wins}"
