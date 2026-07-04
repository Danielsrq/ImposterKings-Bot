"""Per-decision MCTS budget policies: ``(view, n_legal) -> iterations``.

Fixed-N search spends the same budget on a 2-way decision and a 15-way guess, which under-samples
high-branching decisions and over-samples trivial ones. These policies let the bot scale its per-move
iteration budget by branching factor and/or opponent-card uncertainty. ``MCTSAgent`` takes one as its
``budget``; each factory returns a callable with a ``.label`` for the agent name / benchmark logging.

Pure module (only ``InformationSet`` for typing) so the game app and future benchmarks both import it.
"""
from __future__ import annotations

from typing import Callable

from .infoset import InformationSet

Budget = Callable[[InformationSet, int], int]


def opp_cards(view: InformationSet) -> int:
    """The opponent's remaining cards from ``view``: hand + (hidden, until their king flips). 7 at the
    start of a 2P game (6 hand + 1 hidden), decreasing to 0 -- a proxy for how much is still unknown."""
    return view.opp_hand_count + (1 if view.opp_has_hidden else 0)


def _clamp(x: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, x)))


def fixed(n: int) -> Budget:
    """Constant budget ``n`` every decision (the classic MCTS@N)."""
    def f(view: InformationSet, n_legal: int) -> int:
        return n
    f.label = f"fixed{n}"
    return f


def branching(k: int, lo: int = 64, hi: int = 4096) -> Budget:
    """``clamp(k * n_legal, lo, hi)`` -- ~k simulations per legal option, so a 15-way guess gets a real
    budget while binary decisions get little. ``k`` is the primary strength knob."""
    def f(view: InformationSet, n_legal: int) -> int:
        return _clamp(k * n_legal, lo, hi)
    f.label = f"branching-k{k}"
    return f


def hybrid(k1: int, lo: int = 64, hi: int = 4096) -> Budget:
    """``clamp(k1 * n_legal * (1 + opp_cards), lo, hi)`` -- branching AND opponent-card uncertainty, so the
    budget is highest early (many unknown cards) and on high-branching decisions, tapering late."""
    def f(view: InformationSet, n_legal: int) -> int:
        return _clamp(k1 * n_legal * (1 + opp_cards(view)), lo, hi)
    f.label = f"hybrid-k{k1}"
    return f


def make_budget(mode: str, k: int = 100, lo: int = 64, hi: int = 4096) -> Budget:
    """Build a budget from a mode name (for CLI / benchmark specs)."""
    if mode == "hybrid":
        return hybrid(k, lo, hi)
    if mode == "branching":
        return branching(k, lo, hi)
    raise ValueError(f"unknown budget mode {mode!r} (expected 'hybrid' or 'branching')")
