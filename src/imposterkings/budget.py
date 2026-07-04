"""Per-decision MCTS budget policies: ``(view, moves) -> iterations``.

Fixed-N search spends the same budget on a 2-way decision and a 15-way guess, which under-samples
high-branching decisions and over-samples trivial ones. These policies let the bot scale its per-move
iteration budget by branching factor and/or opponent-card uncertainty. ``MCTSAgent`` takes one as its
``budget``; each factory returns a callable with a ``.label`` for the agent name / benchmark logging.

``moves`` is the legal-action list (or a bare int count). At card selection, a card whose on-play ability
opens a sub-decision (guess / number / target / follow-up) counts as ``l`` effective moves so the root
search funds that subtree -- otherwise a 2-card Soldier+Mystic turn gets far too little budget.

Leaf module (imports only cards/actions/infoset) so the game app and future benchmarks both import it.
"""
from __future__ import annotations

from typing import Callable

from . import cards
from .actions import ActionKind
from .cards import Ability
from .infoset import InformationSet

Budget = Callable[[InformationSet, object], int]   # (view, moves|n_legal) -> iterations

# Cards whose ON-PLAY ability opens a sub-decision (guess / pick a number / select a target / follow-up)
# -- mirrors abilities._MANDATORY_GUESS | _OPTIONAL_ONPLAY | {OATHBOUND}. At card selection they lead one
# branch deeper, so each counts as ``l`` legal moves (see ``_effective_n``) to fund that subtree.
_HEAVY_ABILITIES = frozenset({Ability.SOLDIER, Ability.JUDGE, Ability.INQUISITOR, Ability.MYSTIC,
                              Ability.PRINCESS, Ability.SENTRY, Ability.FOOL, Ability.OATHBOUND})


def _effective_n(moves, l: int) -> int:
    """Effective branching for budget sizing. ``moves`` is an int (raw count -> used as-is) or the legal
    Action list, where each PLAY_CARD of a sub-decision card counts as ``l`` (else 1). So a Soldier+Mystic
    turn sizes as ``2*l``, not 2, funding their guess/number subtrees."""
    if isinstance(moves, int):
        return moves
    total = 0
    for m in moves:
        heavy = (m.kind == ActionKind.PLAY_CARD and m.card is not None
                 and cards.card_ability(m.card) in _HEAVY_ABILITIES)
        total += l if heavy else 1
    return total


def opp_cards(view: InformationSet) -> int:
    """The opponent's remaining cards from ``view``: hand + (hidden, until their king flips). 7 at the
    start of a 2P game (6 hand + 1 hidden), decreasing to 0 -- a proxy for how much is still unknown."""
    return view.opp_hand_count + (1 if view.opp_has_hidden else 0)


def _clamp(x: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, x)))


def fixed(n: int) -> Budget:
    """Constant budget ``n`` every decision (the classic MCTS@N)."""
    def f(view: InformationSet, moves) -> int:
        return n
    f.label = f"fixed{n}"
    return f


def branching(k: int, l: int = 3, lo: int = 64, hi: int = 4096) -> Budget:
    """``clamp(k * effective_n, lo, hi)`` -- ~k simulations per legal option (a sub-decision card counts as
    ``l``), so a 15-way guess and a Soldier/Mystic card selection both get a real budget. ``k`` is the
    primary strength knob; ``l`` funds sub-decision subtrees at card selection."""
    def f(view: InformationSet, moves) -> int:
        return _clamp(k * _effective_n(moves, l), lo, hi)
    f.label = f"branching-k{k}-l{l}"
    return f


def hybrid(k1: int, l: int = 3, lo: int = 64, hi: int = 4096) -> Budget:
    """``clamp(k1 * effective_n * (1 + opp_cards), lo, hi)`` -- branching (a sub-decision card counts as
    ``l``) AND opponent-card uncertainty, so the budget is highest early and on high-branching /
    sub-decision decisions, tapering late."""
    def f(view: InformationSet, moves) -> int:
        return _clamp(k1 * _effective_n(moves, l) * (1 + opp_cards(view)), lo, hi)
    f.label = f"hybrid-k{k1}-l{l}"
    return f


def make_budget(mode: str, k: int = 100, l: int = 3, lo: int = 64, hi: int = 4096) -> Budget:
    """Build a budget from a mode name (for CLI / benchmark specs)."""
    if mode == "hybrid":
        return hybrid(k, l, lo, hi)
    if mode == "branching":
        return branching(k, l, lo, hi)
    raise ValueError(f"unknown budget mode {mode!r} (expected 'hybrid' or 'branching')")
