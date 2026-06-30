"""InformationSet: what one player can legally observe, plus the determinizer.

This is the entire contract MCTS needs (bigtwo's design): ``GameState.information_set(observer)``
projects away the opponent's concealed cards; :meth:`determinize` is the stochastic inverse that
samples a concrete, consistent :class:`~imposterkings.state.GameState` to search.

ImposterKings is *near* perfect information: at the root the observer knows their 8 dealt cards and
the face-up leftover, leaving only 9 unknown ids (the opponent's hand plus the face-down leftover);
after the opponent hides 1 and discards 1, those 9 split into opp-hand / opp-hidden / opp-setup-
discard / face-down-leftover. The opponent's setup-discard and the face-down leftover never re-enter
play, so they are interchangeable "muck"; only the opponent's hand and (until their king flips)
hidden card matter for search. ``_consistent`` is the hook for tightening sampling with the
information that guesses leak (a wrong Inquisitor/Soldier/Judge name proves the opponent lacks it).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from . import cards
from .actions import Action, StepKind
from .state import GameState, PendingStep, StackCard


@dataclass
class InformationSet:
    observer: int
    own_hand: Tuple[int, ...]
    own_hidden: Optional[int]
    own_setup_discard: Optional[int]
    kings: Tuple[bool, ...]
    opp_hand_count: int
    opp_has_hidden: bool
    stack: Tuple[StackCard, ...]
    antechambers: Tuple[Tuple[int, ...], ...]
    discard: Tuple[int, ...]
    leftover_faceup: int
    muted_values: frozenset
    to_play: int
    turn_player: int
    starting_player: int
    pending: Tuple[PendingStep, ...]
    history: Tuple[Tuple[int, Action], ...] = field(default_factory=tuple)

    # --- projection --------------------------------------------------------------------

    @classmethod
    def from_state(cls, state: GameState, observer: int) -> "InformationSet":
        opp = 1 - observer
        return cls(
            observer=observer,
            own_hand=state.hands[observer],
            own_hidden=state.hidden[observer],
            own_setup_discard=state.setup_discard[observer],
            kings=state.kings,
            opp_hand_count=len(state.hands[opp]),
            opp_has_hidden=state.hidden[opp] is not None,
            stack=state.stack,
            antechambers=state.antechambers,
            discard=state.discard,
            leftover_faceup=state.leftover_faceup,
            muted_values=state.muted_values,
            to_play=state.to_play,
            turn_player=state.turn_player,
            starting_player=state.starting_player,
            pending=state.pending,
            history=state.history,
        )

    # --- queries -----------------------------------------------------------------------

    def legal_moves(self) -> List[Action]:
        """Legal actions for the observer. The observer's choices never depend on the opponent's
        concealed cards, so a fixed determinization yields the canonical list."""
        if self.to_play != self.observer:
            raise ValueError("legal_moves() requested when it is not the observer's turn")
        return self.determinize(np.random.default_rng(0)).legal_moves()

    def unknown_cards(self) -> List[int]:
        """Instance ids the observer cannot see (opponent hand + opponent hidden + opponent
        setup-discard + the face-down leftover)."""
        seen = set(self.own_hand)
        if self.own_hidden is not None:
            seen.add(self.own_hidden)
        if self.own_setup_discard is not None:
            seen.add(self.own_setup_discard)
        seen.update(sc.card for sc in self.stack)
        seen.update(self.discard)
        for ante in self.antechambers:
            seen.update(ante)
        seen.add(self.leftover_faceup)
        return [c for c in range(cards.DECK_SIZE) if c not in seen]

    def _pinned_opp_cards(self) -> set:
        """Hidden cards the resolution stack has committed to the OPPONENT's hand.

        A pending step can pin a still-hidden card to a player (e.g. Princess's chosen give-card,
        held by ``1 - ABILITY_SWAP_RESPOND.actor`` until the responder answers). Determinize must
        keep such cards in that hand, otherwise it can build a world the committed action can't apply.
        """
        opp = 1 - self.observer
        pinned = set()
        for step in self.pending:
            if step.kind == StepKind.ABILITY_SWAP_RESPOND and step.picked is not None:
                if (1 - step.actor) == opp:
                    pinned.add(step.picked)
        return pinned

    def _consistent(self, opp_hand: Tuple[int, ...]) -> bool:
        """Hook for opponent inference (guess-leak voids). Uniform over consistent worlds for now."""
        return True

    # --- the MCTS sampling seam --------------------------------------------------------

    def determinize(self, rng: np.random.Generator) -> GameState:
        """Sample a concrete GameState consistent with this information set.

        Distributes the unknown pool into the opponent's hand, their hidden card (if their king is
        unflipped), and muck (their setup-discard + the face-down leftover, which never matter)."""
        unknown = self.unknown_cards()
        pinned = self._pinned_opp_cards() & set(unknown)
        free = [c for c in unknown if c not in pinned]
        free = [int(free[i]) for i in rng.permutation(len(free))]

        # Honor committed cards first, then fill the opponent's hand from the free pool.
        n_free_in_hand = self.opp_hand_count - len(pinned)
        opp_hand = tuple(sorted(pinned | set(free[:n_free_in_hand])))
        rest = free[n_free_in_hand:]

        opp_hidden: Optional[int] = None
        if self.opp_has_hidden:
            opp_hidden, rest = rest[0], rest[1:]

        opp_setup_discard = rest[0] if rest else None
        leftover_facedown = rest[1] if len(rest) > 1 else (rest[0] if rest else self.leftover_faceup)

        observer, opp = self.observer, 1 - self.observer
        hands = [(), ()]
        hands[observer] = self.own_hand
        hands[opp] = opp_hand
        hidden = [None, None]
        hidden[observer] = self.own_hidden
        hidden[opp] = opp_hidden
        setup_discard = [None, None]
        setup_discard[observer] = self.own_setup_discard
        setup_discard[opp] = opp_setup_discard

        return GameState(
            hands=tuple(hands),
            hidden=tuple(hidden),
            kings=self.kings,
            antechambers=self.antechambers,
            stack=self.stack,
            discard=self.discard,
            leftover_faceup=self.leftover_faceup,
            leftover_facedown=leftover_facedown,
            muted_values=self.muted_values,
            turn_player=self.turn_player,
            starting_player=self.starting_player,
            pending=self.pending,
            history=self.history,
            winner=None,
            setup_discard=tuple(setup_discard),
        )
