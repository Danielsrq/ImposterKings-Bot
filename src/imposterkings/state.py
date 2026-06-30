"""GameState: the omniscient referee, driven by a LIFO resolution stack.

Unlike bigtwo (atomic one-combo turns), ImposterKings resolves a turn through a sequence of
micro-decisions and reaction windows. ``GameState.pending`` is a stack of :class:`PendingStep`s;
``pending[-1]`` is the decision being made *now* and ``to_play`` derives from its ``actor`` -- so
control can flip to the opponent mid-resolution (a reaction) without leaving the same engine loop.

``apply(action)`` is copy-on-write: it validates the action against the top step, produces a NEW
state (all fields are immutable tuples/frozensets), and may pop/push steps. When the stack empties
the turn is over and ``_begin_turn`` runs for the opponent (handling forced antechamber ascension
and the win check). This keeps bigtwo's exact MCTS contract -- ``deal / legal_moves / is_terminal /
apply / result / to_play / winner / information_set`` -- valid at *every* micro-decision.

All ability semantics live in :mod:`abilities` (imported lazily to avoid a cycle); this module owns
the state container, the value-derivation helpers, and the turn/stack plumbing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, Tuple

import numpy as np

from . import cards, rules
from .actions import Action, StepKind
from .cards import Tag

NUM_PLAYERS = rules.NUM_PLAYERS


@dataclass(frozen=True)
class StackCard:
    """A card occupying a position on the stack/throne. ``stack[-1]`` is the leading card."""
    card: int                              # instance id 0..17
    disgraced: bool = False                # value 0, name/value/ability/tags all stripped
    value_override: Optional[int] = None   # Warlord lands at 9; None otherwise


@dataclass(frozen=True)
class PendingStep:
    """One decision point on the resolution stack.

    Fields beyond ``kind``/``actor`` carry context forward between chained sub-decisions:
    ``source`` (the card whose ability is resolving), ``picked`` (a card chosen earlier, e.g. the
    Princess give-card or the Sentry stack position grabbed), ``chosen``/``limit`` (multi-select
    accumulator + remaining picks for Soldier), ``guess`` (a carried card name), ``against`` (the
    card a reaction window is reacting to).
    """
    kind: StepKind
    actor: int
    source: Optional[int] = None
    picked: Optional[int] = None
    chosen: Tuple[int, ...] = ()
    limit: int = 0
    guess: Optional[str] = None
    against: Optional[int] = None


class GameState:
    def __init__(
        self,
        hands: Tuple[Tuple[int, ...], ...],
        hidden: Tuple[Optional[int], ...],
        kings: Tuple[bool, ...],
        antechambers: Tuple[Tuple[int, ...], ...],
        stack: Tuple[StackCard, ...],
        discard: Tuple[int, ...],
        leftover_faceup: int,
        leftover_facedown: int,
        muted_values: FrozenSet[int],
        turn_player: int,
        starting_player: int,
        pending: Tuple[PendingStep, ...],
        history: Tuple[Tuple[int, Action], ...],
        winner: Optional[int],
        setup_discard: Tuple[Optional[int], ...] = (None, None),
    ) -> None:
        self.hands = hands
        self.hidden = hidden
        self.kings = kings
        self.antechambers = antechambers
        self.stack = stack
        self.discard = discard                  # public pile (King's-Hand/Assassin/countered cards)
        self.setup_discard = setup_discard      # each player's private setup discard (hidden)
        self.leftover_faceup = leftover_faceup
        self.leftover_facedown = leftover_facedown
        self.muted_values = muted_values
        self.turn_player = turn_player
        self.starting_player = starting_player
        self.pending = pending
        self.history = history
        self.winner = winner

    # --- construction ------------------------------------------------------------------

    @classmethod
    def deal(cls, rng: np.random.Generator, starting_player: Optional[int] = None) -> "GameState":
        """Deal 8 cards to each player (2 left over, 1 face-up), then queue the setup decisions.

        Each player will hide 1 card and discard 1 (the SETUP steps below), keeping 6 in hand.
        ``starting_player`` defaults to a coin flip. ``turn_player`` is initialised to the
        *non*-starter so that when the setup stack empties, the generic ``_begin_turn(1 -
        turn_player)`` rule lands on the true starting player with no special-casing.
        """
        perm = [int(c) for c in rng.permutation(cards.DECK_SIZE)]
        hand0 = tuple(sorted(perm[0:8]))
        hand1 = tuple(sorted(perm[8:16]))
        leftover_faceup = perm[16]
        leftover_facedown = perm[17]
        if starting_player is None:
            starting_player = int(rng.integers(NUM_PLAYERS))

        # Setup steps in resolution order: starter hides+discards, then the other player.
        other = 1 - starting_player
        order = [
            PendingStep(StepKind.SETUP_HIDE, starting_player),
            PendingStep(StepKind.SETUP_DISCARD, starting_player),
            PendingStep(StepKind.SETUP_HIDE, other),
            PendingStep(StepKind.SETUP_DISCARD, other),
        ]
        pending = tuple(reversed(order))  # first to resolve sits on top (index -1)

        return cls(
            hands=(hand0, hand1),
            hidden=(None, None),
            kings=(False, False),
            antechambers=((), ()),
            stack=(),
            discard=(),
            leftover_faceup=leftover_faceup,
            leftover_facedown=leftover_facedown,
            muted_values=frozenset(),
            turn_player=other,            # see docstring
            starting_player=starting_player,
            pending=pending,
            history=(),
            winner=None,
            setup_discard=(None, None),
        )

    # --- copy-on-write plumbing --------------------------------------------------------

    def with_(self, **changes) -> "GameState":
        """Return a new state with the given fields replaced (everything else shared)."""
        fields = dict(
            hands=self.hands, hidden=self.hidden, kings=self.kings,
            antechambers=self.antechambers, stack=self.stack, discard=self.discard,
            setup_discard=self.setup_discard,
            leftover_faceup=self.leftover_faceup, leftover_facedown=self.leftover_facedown,
            muted_values=self.muted_values, turn_player=self.turn_player,
            starting_player=self.starting_player, pending=self.pending,
            history=self.history, winner=self.winner,
        )
        fields.update(changes)
        return GameState(**fields)

    def replace_top(self, step: PendingStep, **changes) -> "GameState":
        """Replace the current top step in place (used by multi-select accumulation)."""
        return self.with_(pending=self.pending[:-1] + (step,), **changes)

    def advance(self, *new_steps: PendingStep, **changes) -> "GameState":
        """Pop the current top step, push ``new_steps`` (first listed resolves first).

        If the stack ends up empty (and the game is not already won), the turn is over and the
        opponent's turn begins -- which itself handles forced ascension and the win check.
        """
        new_pending = self.pending[:-1] + tuple(reversed(new_steps))
        st = self.with_(pending=new_pending, **changes)
        if st.winner is not None:
            return st
        if not new_pending:
            return st._begin_turn(1 - st.turn_player)
        return st

    # --- turn boundaries ---------------------------------------------------------------

    def _begin_turn(self, player: int) -> "GameState":
        """Start ``player``'s turn: forced antechamber ascension, else the win check + MAIN."""
        from . import abilities
        if self.antechambers[player]:
            return abilities.ascend(self.with_(turn_player=player), player)

        st = self.with_(turn_player=player,
                        pending=(PendingStep(StepKind.MAIN, player),))
        has_play = bool(abilities.legal_play_cards(st, player))
        has_flip = (not st.kings[player]) and bool(st.stack)
        if not has_play and not has_flip:
            # No card beats the leading card and no king to flip -> opponent wins.
            return st.with_(winner=1 - player, pending=())
        return st

    # --- queries -----------------------------------------------------------------------

    @property
    def to_play(self) -> int:
        if self.pending:
            return self.pending[-1].actor
        return self.turn_player

    @property
    def phase(self) -> StepKind:
        return self.pending[-1].kind

    def is_terminal(self) -> bool:
        return self.winner is not None

    def legal_moves(self) -> List[Action]:
        from .generate import legal_moves
        return legal_moves(self)

    def result(self, scaled: bool = True) -> List[float]:
        if self.winner is None:
            raise ValueError("result() called on a non-terminal state")
        cards_left = [len(h) for h in self.hands]
        return rules.terminal_rewards(self.winner, cards_left, scaled)

    def apply(self, action: Action) -> "GameState":
        from . import abilities
        return abilities.resolve(self, action)

    def information_set(self, observer: int):
        from .infoset import InformationSet
        return InformationSet.from_state(self, observer)

    # --- value derivation (single source of truth for "what is this worth now") --------

    @property
    def leading(self) -> Optional[StackCard]:
        return self.stack[-1] if self.stack else None

    def effective_stack_value(self, sc: StackCard) -> int:
        """Value of a stack card: 0 if disgraced; 3 if its base value is muted; else override/base.

        Precedence: disgraced (0) > Mystic mute (3) > Warlord override (9) > base value. Muting is
        keyed on base value, so it retroactively overrides even a Warlord that landed at 9.
        """
        if sc.disgraced:
            return 0
        base = cards.card_value(sc.card)
        if base in self.muted_values:
            return rules.MYSTIC_SET_VALUE
        if sc.value_override is not None:
            return sc.value_override
        return base

    def effective_hand_value(self, card: int) -> int:
        """Value of a hand card for legality: muted -> 3; Warlord -> 8 when royalty present."""
        base = cards.card_value(card)
        if base in self.muted_values:
            return rules.MYSTIC_SET_VALUE
        if cards.card_ability(card) == cards.Ability.WARLORD and self.royalty_present():
            return base + rules.WARLORD_BONUS  # 7 + 1 = 8 (does not stack per royalty)
        return base

    def leading_value(self) -> Optional[int]:
        lead = self.leading
        return None if lead is None else self.effective_stack_value(lead)

    def active_royalty(self, sc: StackCard) -> bool:
        """A stack card that still counts as royalty (not disgraced, not muted away)."""
        if sc.disgraced:
            return False
        if cards.card_value(sc.card) in self.muted_values:
            return False  # muting strips tags (moot for Queen/Princess at value 9, kept general)
        return Tag.ROYALTY in cards.card_def(sc.card).tags

    def royalty_present(self) -> bool:
        return any(self.active_royalty(sc) for sc in self.stack)
