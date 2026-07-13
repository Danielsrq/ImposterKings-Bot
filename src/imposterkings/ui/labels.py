"""Action -> human-readable string. The one place the UI's wording for moves and decisions lives.

Kept apart from the painting so a label can be reworded without opening any drawing code (and so the tree
view and the review screen, which both need ``compact_action``, do not have to import the board renderer).
"""
from __future__ import annotations

from .. import cards
from ..actions import Action, ActionKind, StepKind

# Friendly labels for the decision header (the raw StepKind names are long/cryptic).
DECISION_LABELS = {
    StepKind.SETUP_HIDE: "Hide a card", StepKind.SETUP_DISCARD: "Discard a card",
    StepKind.MAIN: "Your turn", StepKind.ABILITY_MAY: "Use ability?",
    StepKind.ABILITY_CHOICE: "Take the effect?", StepKind.ABILITY_GUESS: "Name a card",
    StepKind.ABILITY_NUMBER: "Pick a value (1-8)", StepKind.ABILITY_HAND_CARD: "Choose a hand card",
    StepKind.ABILITY_STACK_TARGET: "Choose a stack card", StepKind.ABILITY_SWAP_RESPOND: "Card to swap",
    StepKind.OATHBOUND_SECOND: "Play a follow-up", StepKind.REACTION_KINGSHAND: "King's Hand?",
    StepKind.REACTION_ASSASSIN: "Assassin?", StepKind.REACTION_KH_VS_ASSASSIN: "King's Hand vs Assassin?",
}

# The flattened abilities declare their parameter at ABILITY_MAY, so give them a clearer header.
ABILITY_MAY_LABEL = {
    cards.Ability.MYSTIC: "Mystic: pick a value (or decline)",
    cards.Ability.INQUISITOR: "Interrogate: name a card (or decline)",
    cards.Ability.FOOL: "Fool: take a stack card (or decline)",
}

REACTION_KINDS = (StepKind.REACTION_KINGSHAND, StepKind.REACTION_ASSASSIN,
                  StepKind.REACTION_KH_VS_ASSASSIN)

SHORT_ACTION = {
    ActionKind.DECLARE_ABILITY: "declare", ActionKind.DECLINE_ABILITY: "decline",
    ActionKind.FLIP_KING: "flip-king", ActionKind.STOP: "stop",
    ActionKind.REVEAL_KINGSHAND: "KingsHand!", ActionKind.REVEAL_ASSASSIN: "Assassin!",
    ActionKind.DECLINE_REACTION: "no-react",
}

CARD_PREFIX = {ActionKind.PLAY_CARD: "", ActionKind.HIDE_CARD: "hide ",
               ActionKind.DISCARD_CARD: "discard ", ActionKind.CHOOSE_HAND_CARD: "give "}


def reaction_context(view) -> str:
    """A human-readable note about what a reaction window is reacting to (reaction steps only)."""
    if not view.pending:
        return ""
    step = view.pending[-1]
    if step.kind not in REACTION_KINDS or step.source is None:
        return ""
    src = cards.card_name(step.source)
    if step.guess is not None:
        return f"Counter {src}? (guessed {step.guess})"
    return f"Counter opponent's {src}?"


def compact_action(action: Action) -> str:
    """A short action label for the narrow reasoning panel (drops the play_card()/#id noise).
    Card actions keep their kind (hide/discard/give) so the setup phase reads correctly."""
    k = action.kind
    if k in CARD_PREFIX:
        cdef = cards.card_def(action.card)
        return f"{CARD_PREFIX[k]}{cdef.name}({cdef.value})"
    if k == ActionKind.GUESS_CARD:
        return f"guess {action.name}"
    if k == ActionKind.CHOOSE_NUMBER:
        return f"mute {action.number}"
    if k == ActionKind.CHOOSE_STACK_TARGET:
        return f"target@{action.target}"
    return SHORT_ACTION.get(k, k.name.lower())


def decision_label(view) -> str:
    """The panel's header for the pending decision (GAME OVER when there is none)."""
    kind = view.pending[-1].kind if view.pending else None
    if kind is None:
        return "GAME OVER"
    label = DECISION_LABELS.get(kind, "GAME OVER")
    if kind == StepKind.ABILITY_MAY and view.pending[-1].source is not None:
        return ABILITY_MAY_LABEL.get(cards.card_ability(view.pending[-1].source), label)
    return label
