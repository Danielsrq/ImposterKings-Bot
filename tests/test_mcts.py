"""MCTS sanity (fast) and strength vs Random (slow)."""
from __future__ import annotations

import numpy as np
import pytest

from imposterkings.actions import StepKind
from imposterkings.agents import MCTSAgent, RandomAgent
from imposterkings.arena import run_arena
from imposterkings.record import play_and_record
from imposterkings.state import GameState


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
