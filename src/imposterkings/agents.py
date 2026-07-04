"""Agents. They receive only an InformationSet -- never the omniscient GameState.

A single structural ``Agent`` protocol (bigtwo's design) lets the same game loop seat humans, random
baselines, and the MCTS bot interchangeably. The MCTS agent is added in Phase 3; it will store its
last :class:`SearchResult` on ``last_result`` for explainability.
"""
from __future__ import annotations

from typing import Callable, Optional, Protocol

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

    A single legal action is normally returned without searching (common at forced reaction windows),
    since there is no decision to make. With ``evaluate_forced`` the agent searches even then -- the
    move is still forced, but the search yields a real value estimate of the position (a position's
    eval is well-defined even when the policy is degenerate). Used for post-game review so forced turns
    (ascensions, sole reactions) still carry an eval; left off in live play to avoid needless latency.

    ``budget`` (a :mod:`imposterkings.budget` policy, ``(view, n_legal) -> iters``) overrides the fixed
    ``iterations`` per decision -- e.g. the hybrid branching/uncertainty schedule -- and sets ``name``."""

    name = "mcts"        # class default; per-instance name (set below) reflects the budget mode

    def __init__(self, iterations: int = 1000, c: float = DEFAULT_C, use_knowledge: bool = True,
                 evaluate_forced: bool = False, budget: Optional[Callable] = None) -> None:
        self.iterations = iterations
        self.c = c
        self.use_knowledge = use_knowledge     # False = ignore guess-leaked facts (A/B benchmarking)
        self.evaluate_forced = evaluate_forced
        self.budget = budget                   # None -> fixed self.iterations
        self.name = "mcts" if budget is None else f"mcts-{getattr(budget, 'label', 'budget')}"
        self.last_result: Optional[SearchResult] = None

    def select_move(self, view: InformationSet, rng: np.random.Generator) -> Action:
        moves = view.legal_moves()
        if len(moves) == 1 and not self.evaluate_forced:
            self.last_result = None
            return moves[0]
        iters = self.budget(view, len(moves)) if self.budget is not None else self.iterations
        config = SearchConfig(rng=rng, iterations=iters, c=self.c, use_knowledge=self.use_knowledge)
        self.last_result = search(view, config)
        # The move is still forced when there is only one; return it explicitly (best_move agrees).
        return moves[0] if len(moves) == 1 else self.last_result.best_move
