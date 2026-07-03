"""Legal-action enumeration: ``legal_moves(state)`` dispatched on the current decision point.

Every entry point asks "what may the actor on top of the resolution stack do *now*", so the MCTS
contract (one flat list of hashable :class:`~imposterkings.actions.Action`) holds at every
micro-decision -- exactly what bigtwo's ``search`` consumes. Choices among interchangeable
duplicate cards are deduplicated by name to keep the branching factor (and MCTS visit dilution) low.
"""
from __future__ import annotations

from typing import List, Tuple

from . import abilities, cards
from .actions import (
    Action, ActionKind, DECLARE, DECLINE, DECLINE_REACTION, FLIP_KING, REVEAL_ASSASSIN,
    REVEAL_KINGSHAND, STOP, StepKind,
)
from .cards import Ability
from .state import GameState

# Base values of the reaction cards -- a muted base value strips the reaction tag/ability.
_KINGSHAND_VALUE = cards.card_value(abilities.KINGSHAND_ID)   # 8
_ASSASSIN_VALUE = cards.card_value(abilities.ASSASSIN_ID)     # 2


def _dedupe_by_name(card_ids: Tuple[int, ...]) -> List[int]:
    out: List[int] = []
    seen = set()
    for c in card_ids:
        name = cards.card_name(c)
        if name not in seen:
            seen.add(name)
            out.append(c)
    return out


def _can_react(state: GameState, actor: int, card_id: int, base_value: int) -> bool:
    """A reaction is available only if the actor holds the card and it is not muted away."""
    return card_id in state.hands[actor] and base_value not in state.muted_values


def legal_moves(state: GameState) -> List[Action]:
    if state.winner is not None:
        return []

    step = state.pending[-1]
    k = step.kind
    actor = step.actor

    if k == StepKind.SETUP_HIDE:
        return [Action(ActionKind.HIDE_CARD, card=c) for c in _dedupe_by_name(state.hands[actor])]

    if k == StepKind.SETUP_DISCARD:
        return [Action(ActionKind.DISCARD_CARD, card=c) for c in _dedupe_by_name(state.hands[actor])]

    if k == StepKind.MAIN:
        moves = [Action(ActionKind.PLAY_CARD, card=c) for c in abilities.legal_play_cards(state, actor)]
        if (not state.kings[actor]) and state.stack:
            moves.append(FLIP_KING)
        return moves

    if k == StepKind.OATHBOUND_SECOND:
        return [Action(ActionKind.PLAY_CARD, card=c) for c in _dedupe_by_name(state.hands[actor])]

    if k == StepKind.ABILITY_MAY:
        ability = cards.card_ability(step.source)
        # Flattened abilities declare their PARAMETER here (King's Hand then reacts to it): one decision
        # instead of declare -> window -> parameter. Decline is always available.
        if ability == Ability.MYSTIC:
            from . import rules
            return [DECLINE] + [Action(ActionKind.CHOOSE_NUMBER, number=n)
                                for n in range(rules.MYSTIC_MIN, rules.MYSTIC_MAX + 1)]
        if ability == Ability.INQUISITOR:
            return [DECLINE] + [Action(ActionKind.GUESS_CARD, name=n) for n in cards.CARD_NAMES]
        if ability == Ability.FOOL:
            return [DECLINE] + [Action(ActionKind.CHOOSE_STACK_TARGET, target=i)
                                for i in abilities._fool_targets(state, step.source)]
        return [DECLARE, DECLINE]   # Sentry / Princess: King's Hand blocks the bare declaration

    if k == StepKind.ABILITY_GUESS:
        # NOTE: all names are offered on purpose. Naming a card the opponent cannot hold is a legitimate
        # *deliberate whiff* -- a landed guess opens a King's-Hand window that can discard the played
        # Soldier/Judge from the throne, so a guaranteed miss protects it. (This is why MCTS may "guess"
        # a card you hold, e.g. your own hidden card: it is choosing the safe miss, not a leaked belief.)
        return [Action(ActionKind.GUESS_CARD, name=n) for n in cards.CARD_NAMES]

    if k == StepKind.ABILITY_NUMBER:
        from . import rules
        return [Action(ActionKind.CHOOSE_NUMBER, number=n)
                for n in range(rules.MYSTIC_MIN, rules.MYSTIC_MAX + 1)]

    if k == StepKind.ABILITY_HAND_CARD:
        moves = [Action(ActionKind.CHOOSE_HAND_CARD, card=c) for c in _dedupe_by_name(state.hands[actor])]
        if cards.card_ability(step.source) == Ability.JUDGE:
            moves.append(STOP)  # may decline to queue a card
        return moves

    if k == StepKind.ABILITY_SWAP_RESPOND:
        return [Action(ActionKind.CHOOSE_HAND_CARD, card=c) for c in _dedupe_by_name(state.hands[actor])]

    if k == StepKind.ABILITY_STACK_TARGET:
        ability = cards.card_ability(step.source)
        if ability == Ability.FOOL:
            return [Action(ActionKind.CHOOSE_STACK_TARGET, target=i)
                    for i in abilities._fool_targets(state, step.source)]
        if ability == Ability.SENTRY:
            return [Action(ActionKind.CHOOSE_STACK_TARGET, target=i)
                    for i in abilities._sentry_targets(state, step.source)]
        if ability == Ability.SOLDIER:
            avail = [i for i, sc in enumerate(state.stack)
                     if not sc.disgraced and i not in step.chosen]
            moves = [Action(ActionKind.CHOOSE_STACK_TARGET, target=i) for i in avail]
            moves.append(STOP)  # finish disgracing (0..3 targets allowed)
            return moves

    if k == StepKind.REACTION_KINGSHAND:
        moves = []
        if _can_react(state, actor, abilities.KINGSHAND_ID, _KINGSHAND_VALUE):
            moves.append(REVEAL_KINGSHAND)
        moves.append(DECLINE_REACTION)
        return moves

    if k == StepKind.REACTION_ASSASSIN:
        moves = []
        if _can_react(state, actor, abilities.ASSASSIN_ID, _ASSASSIN_VALUE):
            moves.append(REVEAL_ASSASSIN)
        moves.append(DECLINE_REACTION)
        return moves

    if k == StepKind.REACTION_KH_VS_ASSASSIN:
        moves = []
        if _can_react(state, actor, abilities.KINGSHAND_ID, _KINGSHAND_VALUE):
            moves.append(REVEAL_KINGSHAND)
        moves.append(DECLINE_REACTION)
        return moves

    raise ValueError(f"legal_moves: unhandled step kind {k!r}")
