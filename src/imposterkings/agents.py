"""Agents. They receive only an InformationSet -- never the omniscient GameState.

A single structural ``Agent`` protocol (bigtwo's design) lets the same game loop seat humans, random
baselines, and the MCTS bot interchangeably. The MCTS agent is added in Phase 3; it will store its
last :class:`SearchResult` on ``last_result`` for explainability.
"""
from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from .actions import Action
from .infoset import InformationSet
from .mcts import DEFAULT_C, SearchConfig, SearchResult, search


class Agent(Protocol):
    def select_move(self, view: InformationSet, rng: np.random.Generator) -> Action:
        ...


class RandomAgent:
    """Control baseline: pick uniformly among legal actions."""

    name = "random"

    def select_move(self, view: InformationSet, rng: np.random.Generator) -> Action:
        moves = view.legal_moves()
        # Index by integer rather than rng.choice(moves): numpy would coerce the frozen Action
        # objects into an object array, which is slower and fiddly.
        return moves[int(rng.integers(len(moves)))]


class MCTSAgent:
    """SO-ISMCTS bot. After each move ``last_result`` holds the search stats for explainability.

    A single legal action is returned without searching (common at forced reaction windows)."""

    name = "mcts"

    def __init__(self, iterations: int = 1000, c: float = DEFAULT_C, use_knowledge: bool = True) -> None:
        self.iterations = iterations
        self.c = c
        self.use_knowledge = use_knowledge     # False = ignore guess-leaked facts (A/B benchmarking)
        self.last_result: Optional[SearchResult] = None

    def select_move(self, view: InformationSet, rng: np.random.Generator) -> Action:
        moves = view.legal_moves()
        if len(moves) == 1:
            self.last_result = None
            return moves[0]
        config = SearchConfig(rng=rng, iterations=self.iterations, c=self.c,
                              use_knowledge=self.use_knowledge)
        self.last_result = search(view, config)
        return self.last_result.best_move
