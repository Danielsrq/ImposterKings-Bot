"""Shared test helpers: build concrete states and drive action sequences.

The general scenario builder now lives in ``imposterkings.scenario``; ``cid``/``sc``/``play`` are
re-exported from there so tests and the src utility can't drift. ``mainstate`` stays a thin MAIN-phase
convenience wrapper."""
from __future__ import annotations

from typing import Tuple

from imposterkings import cards
from imposterkings.actions import StepKind
from imposterkings.scenario import cid, sc  # noqa: F401  (re-exported for tests)
from imposterkings.scenario import play as run  # noqa: F401  (helpers.run == scenario.play)
from imposterkings.state import GameState, PendingStep, StackCard


def mainstate(
    hand0: Tuple[int, ...] = (),
    hand1: Tuple[int, ...] = (),
    *,
    stack: Tuple[StackCard, ...] = (),
    kings=(False, False),
    hidden=(None, None),
    antechambers=((), ()),
    muted=frozenset(),
    to_play: int = 0,
    discard: Tuple[int, ...] = (),
    hand_lacks=(frozenset(), frozenset()),
    hand_has=(frozenset(), frozenset()),
) -> GameState:
    """A state sitting at a MAIN decision for ``to_play`` (turn_player == to_play)."""
    return GameState(
        hands=(tuple(sorted(hand0)), tuple(sorted(hand1))),
        hidden=hidden,
        kings=kings,
        antechambers=antechambers,
        stack=stack,
        discard=discard,
        leftover_faceup=-1,
        leftover_facedown=-1,
        muted_values=frozenset(muted),
        turn_player=to_play,
        starting_player=0,
        pending=(PendingStep(StepKind.MAIN, to_play),),
        history=(),
        winner=None,
        hand_lacks=hand_lacks,
        hand_has=hand_has,
    )


def names_on_stack(state: GameState):
    return [(cards.card_name(s.card), state.effective_stack_value(s), s.disgraced) for s in state.stack]
