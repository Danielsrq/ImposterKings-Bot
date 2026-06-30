"""End-to-end invariants: random games always terminate with a single winner, reproducibly."""
from __future__ import annotations

import numpy as np

from imposterkings.state import GameState


def _play_random(seed: int):
    rng = np.random.default_rng(seed)
    st = GameState.deal(rng, starting_player=0)
    steps = 0
    while not st.is_terminal():
        moves = st.legal_moves()
        assert moves, f"no legal moves at phase {st.phase} for seat {st.to_play}"
        st = st.apply(moves[int(rng.integers(len(moves)))])
        steps += 1
        assert steps < 5000, "game failed to terminate"
    return st.winner, steps


def test_random_games_terminate_with_a_winner():
    for seed in range(200):
        winner, _ = _play_random(seed)
        assert winner in (0, 1)


def test_random_game_is_reproducible():
    assert _play_random(12345) == _play_random(12345)
