"""The game driver: a single agent-agnostic loop, plus a seat-swapped arena.

``play_game`` dispatches every micro-decision (including the setup hide/discard) to whichever agent
is seated -- human, random, or MCTS -- via the same ``select_move`` call. The ``on_decision`` hook is
the seam used for the live CLI display, the UI, and (later) dataset collection.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from .actions import Action
from .infoset import InformationSet
from .state import GameState

OnDecision = Callable[[int, InformationSet, Action, object, GameState], None]


def play_game(agents, rng: np.random.Generator,
              on_decision: Optional[OnDecision] = None,
              starting_player: Optional[int] = None,
              play_rng: Optional[np.random.Generator] = None,
              initial_state: Optional[GameState] = None) -> Tuple[int, List[float], GameState]:
    """Play one full game; return ``(winner, reward_vector, terminal_state)``.

    ``rng`` deals the cards (and picks the starter). Agents draw from ``play_rng`` if given, else
    ``rng`` -- so two games can share an identical deal but use independent play randomness.
    ``initial_state`` (e.g. from ``scenario.build``) plays from a constructed position instead of a deal."""
    state = initial_state if initial_state is not None else GameState.deal(rng, starting_player=starting_player)
    if play_rng is None:
        play_rng = rng
    while not state.is_terminal():
        seat = state.to_play
        view = state.information_set(seat)
        move = agents[seat].select_move(view, play_rng)
        if on_decision is not None:
            on_decision(seat, view, move, agents[seat], state)
        state = state.apply(move)
    return state.winner, state.result(), state


def run_arena(agents, games: int, rng: np.random.Generator, swap: bool = True) -> List[int]:
    """Play ``games`` games, optionally swapping seats every other game to cancel lead advantage.

    Returns wins indexed by the agents' *original* seat (so swapping is accounted for)."""
    wins = [0, 0]
    for g in range(games):
        swapped = swap and (g % 2 == 1)
        seated = [agents[1], agents[0]] if swapped else list(agents)
        winner, _, _ = play_game(seated, rng)
        original = (1 - winner) if swapped else winner
        wins[original] += 1
    return wins
