"""Board-test utility: construct an arbitrary mid-game scenario and play from it, on the real engine.

Building a concrete position directly is far easier than hunting a random seed that happens to reach a
particular edge case. Cards may be given by NAME (first matching instance) or by explicit instance id;
duplicates (2x Oathbound/Soldier/Inquisitor/Elder) are disambiguated with ``cid(name, k)`` or an id.

    from imposterkings import scenario as sb
    st = sb.build(hand0=["Oathbound", "Inquisitor", "Warlord"], hand1=["Elder", "KingsHand"],
                  stack=["Warlord"], turn_player=0)
    st = sb.play(st, sb.play_card(sb.cid("Oathbound")), ...)
    print(sb.show(st))
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union

from . import cards
from .actions import (Action, ActionKind, DECLARE, DECLINE, DECLINE_REACTION, FLIP_KING,
                      REVEAL_ASSASSIN, REVEAL_KINGSHAND, STOP, StepKind)
from .state import GameState, PendingStep, StackCard

CardRef = Union[int, str]                 # an instance id, or a card name (first matching instance)
StackRef = Union[int, str, StackCard]     # id / name / an explicit StackCard


def cid(name: str, k: int = 0) -> int:
    """The k-th instance id carrying ``name`` (e.g. ``cid("Oathbound", 1)`` = the 2nd Oathbound)."""
    return cards.card_ids_for_name(name)[k]


def sc(name: str, *, k: int = 0, disgraced: bool = False, override: Optional[int] = None) -> StackCard:
    """A ``StackCard`` by name (``override`` = a landed value like Warlord's 9)."""
    return StackCard(cid(name, k), disgraced=disgraced, value_override=override)


def _to_id(ref: CardRef) -> int:
    return ref if isinstance(ref, int) else cid(ref)


def _to_ids(refs: Iterable[CardRef]) -> Tuple[int, ...]:
    return tuple(sorted(_to_id(r) for r in refs))


def _to_stackcard(ref: StackRef) -> StackCard:
    if isinstance(ref, StackCard):
        return ref
    return StackCard(_to_id(ref))


def _to_stack(refs: Iterable[StackRef]) -> Tuple[StackCard, ...]:
    return tuple(_to_stackcard(r) for r in refs)     # order preserved; [-1] leads


def build(
    *,
    hand0: Iterable[CardRef] = (),
    hand1: Iterable[CardRef] = (),
    stack: Iterable[StackRef] = (),
    kings: Tuple[bool, bool] = (False, False),
    hidden: Tuple[Optional[CardRef], Optional[CardRef]] = (None, None),
    antechambers: Tuple[Iterable[CardRef], Iterable[CardRef]] = ((), ()),
    muted: Iterable[int] = (),
    discard: Iterable[CardRef] = (),
    turn_player: int = 0,
    pending: Optional[Tuple[PendingStep, ...]] = None,
    phase: StepKind = StepKind.MAIN,
    leftover_faceup: int = -1,
    leftover_facedown: int = -1,
    setup_discard: Tuple[Optional[CardRef], Optional[CardRef]] = (None, None),
    hand_lacks=(frozenset(), frozenset()),
    hand_has=(frozenset(), frozenset()),
    winner: Optional[int] = None,
) -> GameState:
    """Construct an arbitrary ``GameState``. Hands/stack/discard accept names or ids. By default the
    state sits at ``PendingStep(phase, turn_player)`` (so it starts at any decision, not just MAIN); pass
    an explicit ``pending`` tuple for a hand-built resolution stack (``pending[-1]`` is the current step).
    """
    def _opt(ref: Optional[CardRef]) -> Optional[int]:
        return None if ref is None else _to_id(ref)

    if pending is None:
        pending = (PendingStep(phase, turn_player),)
    return GameState(
        hands=(_to_ids(hand0), _to_ids(hand1)),
        hidden=(_opt(hidden[0]), _opt(hidden[1])),
        kings=kings,
        antechambers=(_to_ids(antechambers[0]), _to_ids(antechambers[1])),
        stack=_to_stack(stack),
        discard=tuple(_to_id(c) for c in discard),
        leftover_faceup=leftover_faceup,
        leftover_facedown=leftover_facedown,
        muted_values=frozenset(muted),
        turn_player=turn_player,
        starting_player=turn_player,
        pending=pending,
        history=(),
        winner=winner,
        setup_discard=(_opt(setup_discard[0]), _opt(setup_discard[1])),
        hand_lacks=hand_lacks,
        hand_has=hand_has,
    )


# --- action shorthands (re-exports + tiny constructors) ------------------------------------
# Re-exported payload-free singletons: DECLARE, DECLINE, STOP, FLIP_KING, REVEAL_KINGSHAND,
# REVEAL_ASSASSIN, DECLINE_REACTION.

def play_card(card: CardRef) -> Action:
    return Action(ActionKind.PLAY_CARD, card=_to_id(card))


def guess(name: str) -> Action:
    return Action(ActionKind.GUESS_CARD, name=name)


def choose_number(n: int) -> Action:
    return Action(ActionKind.CHOOSE_NUMBER, number=n)


def choose_hand_card(card: CardRef) -> Action:
    return Action(ActionKind.CHOOSE_HAND_CARD, card=_to_id(card))


def choose_stack_target(index: int) -> Action:
    return Action(ActionKind.CHOOSE_STACK_TARGET, target=index)


# --- driving + inspection ------------------------------------------------------------------

def play(state: GameState, *actions: Action) -> GameState:
    """Apply a sequence of actions in order and return the resulting state."""
    for a in actions:
        state = state.apply(a)
    return state


def _stack_repr(state: GameState) -> str:
    parts = []
    for s in state.stack:
        tag = "disgraced" if s.disgraced else str(state.effective_stack_value(s))
        parts.append(f"{cards.card_name(s.card)}[{tag}]")
    return " | ".join(parts) if parts else "(empty)"


def _hand_repr(ids: Tuple[int, ...]) -> str:
    return ", ".join(cards.format_card(c) for c in ids) if ids else "(none)"


def show(state: GameState) -> str:
    """A readable, omniscient dump of the board -- for eyeballing a constructed scenario."""
    lead = state.leading
    lead_s = "(empty)" if lead is None else f"{cards.card_name(lead.card)} = {state.leading_value()}"
    phase = state.pending[-1].kind.name if state.pending else "(none)"
    lines = [
        f"to_play=P{state.to_play}  turn_player=P{state.turn_player}  phase={phase}"
        + (f"  winner=P{state.winner}" if state.winner is not None else ""),
        f"leading: {lead_s}",
        f"stack:   {_stack_repr(state)}",
        f"P0 hand: {_hand_repr(state.hands[0])}   hidden={state.hidden[0]}  king_flipped={state.kings[0]}",
        f"P1 hand: {_hand_repr(state.hands[1])}   hidden={state.hidden[1]}  king_flipped={state.kings[1]}",
        f"antechambers: P0={[cards.card_name(c) for c in state.antechambers[0]]} "
        f"P1={[cards.card_name(c) for c in state.antechambers[1]]}",
        f"discard: {[cards.card_name(c) for c in state.discard]}   muted={sorted(state.muted_values)}",
    ]
    return "\n".join(lines)
