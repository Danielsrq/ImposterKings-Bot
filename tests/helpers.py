"""Shared test helpers: build concrete states and drive action sequences."""
from __future__ import annotations

from typing import Optional, Tuple

from imposterkings import cards
from imposterkings.actions import Action, StepKind
from imposterkings.state import GameState, PendingStep, StackCard


def cid(name: str, k: int = 0) -> int:
    """The k-th instance id carrying ``name`` (e.g. ``cid("Oathbound", 1)``)."""
    return cards.card_ids_for_name(name)[k]


def sc(name: str, k: int = 0, disgraced: bool = False, override: Optional[int] = None) -> StackCard:
    return StackCard(cid(name, k), disgraced=disgraced, value_override=override)


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
    )


def run(state: GameState, *actions: Action) -> GameState:
    for a in actions:
        state = state.apply(a)
    return state


def names_on_stack(state: GameState):
    return [(cards.card_name(s.card), state.effective_stack_value(s), s.disgraced) for s in state.stack]
