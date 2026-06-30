"""The micro-decision action type and the decision-point kinds.

ImposterKings turns are not atomic (unlike bigtwo's one-combo turns): a single "play" may unfold
into several sub-choices, and reaction windows hand control to the opponent mid-resolution. We
therefore model every sub-choice as its own decision point. The engine's resolution stack stores
:class:`StepKind` steps; an agent answers the top step by returning an :class:`Action`.

Both enums and the frozen :class:`Action` live in this leaf module (it imports nothing from the
package) so ``state``/``generate``/``abilities`` can share them without an import cycle. ``Action``
is frozen+hashable so it can key the MCTS ``Dict[Action, Node]`` exactly like bigtwo's ``Move``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class StepKind(IntEnum):
    """The kind of decision sitting on top of the resolution stack (what is being decided now)."""
    SETUP_HIDE = 1            # choose the hidden card from the 8 dealt
    SETUP_DISCARD = 2         # choose the discard from the remaining 7
    MAIN = 3                  # normal turn: play a card or flip the king
    ABILITY_MAY = 4           # declare/decline an optional ("may") ability
    ABILITY_GUESS = 5         # name a card in the opponent's hand (Soldier/Judge/Inquisitor)
    ABILITY_NUMBER = 6        # pick a base value 1..8 (Mystic)
    ABILITY_HAND_CARD = 7     # choose one of your own hand cards (Princess give / Sentry in / Judge queue)
    ABILITY_STACK_TARGET = 8  # choose a stack position (Soldier disgrace / Sentry swap / Fool take)
    ABILITY_SWAP_RESPOND = 9  # opponent picks their hand card to swap (Princess)
    OATHBOUND_SECOND = 10     # play the immediate follow-up card after Oathbound self-disgrace
    REACTION_KINGSHAND = 11   # opponent may reveal King's Hand to counter a declared "may"
    REACTION_ASSASSIN = 12    # opponent may reveal Assassin in response to a king-flip
    REACTION_KH_VS_ASSASSIN = 13  # flipper may reveal King's Hand to counter the Assassin (nested)
    ABILITY_CHOICE = 14       # an inner accept/decline with NO reaction window (Soldier's package)


class ActionKind(IntEnum):
    """The kind of action an agent returns to answer the current step."""
    HIDE_CARD = 1
    DISCARD_CARD = 2
    PLAY_CARD = 3
    FLIP_KING = 4
    DECLARE_ABILITY = 5
    DECLINE_ABILITY = 6
    GUESS_CARD = 7
    CHOOSE_NUMBER = 8
    CHOOSE_HAND_CARD = 9
    CHOOSE_STACK_TARGET = 10
    STOP = 11                 # finish a variable-length multi-select (e.g. Soldier targets)
    REVEAL_KINGSHAND = 12
    REVEAL_ASSASSIN = 13
    DECLINE_REACTION = 14


@dataclass(frozen=True)
class Action:
    """One micro-decision. Optional payload fields are populated per ``kind``.

    - ``card``:   instance id 0..17 (play / hide / discard / hand-card choice / revealed reaction).
    - ``target``: a stack position index (for disgrace / swap / take targeting).
    - ``number``: a Mystic base value 1..8.
    - ``name``:   a guessed card name (Soldier / Judge / Inquisitor).
    Player is always derivable in the 2-player game, so there is no player field.
    """
    kind: ActionKind
    card: Optional[int] = None
    target: Optional[int] = None
    number: Optional[int] = None
    name: Optional[str] = None

    def __str__(self) -> str:  # pragma: no cover - convenience
        from .explain import format_action
        return format_action(self)


# Singletons for the payload-free actions (cheap to share because Action is frozen).
FLIP_KING = Action(ActionKind.FLIP_KING)
DECLARE = Action(ActionKind.DECLARE_ABILITY)
DECLINE = Action(ActionKind.DECLINE_ABILITY)
STOP = Action(ActionKind.STOP)
REVEAL_KINGSHAND = Action(ActionKind.REVEAL_KINGSHAND)
REVEAL_ASSASSIN = Action(ActionKind.REVEAL_ASSASSIN)
DECLINE_REACTION = Action(ActionKind.DECLINE_REACTION)
