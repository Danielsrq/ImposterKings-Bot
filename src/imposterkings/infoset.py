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

import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from . import cards
from .actions import Action, StepKind
from .rng import as_search_rng
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
    # Guess-leaked knowledge about the OPPONENT's hand, from the observer's view (by card name).
    opp_hand_lacks: frozenset = field(default_factory=frozenset)   # names the opp hand has 0 of
    opp_hand_has: frozenset = field(default_factory=frozenset)     # names the opp hand has >=1 of

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
            opp_hand_lacks=state.hand_lacks[observer],
            opp_hand_has=state.hand_has[observer],
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
        """True if ``opp_hand`` respects the guess-leaked knowledge: none of the lacked names and at
        least one of every 'has' name. Used to validate the constructive determinization."""
        names = [cards.card_name(c) for c in opp_hand]
        if any(n in self.opp_hand_lacks for n in names):
            return False
        return all(h in names for h in self.opp_hand_has)

    # --- knowledge level (how narrowed the opponent's hand is) --------------------------

    def possible_opp_hands(self, cap: int = 3) -> int:
        """How many DISTINCT opponent hands (as card-name multisets) are still consistent with what
        the observer knows -- guesses + card-counting (``unknown_cards`` already excludes public zones).
        Counts stop early once ``cap`` distinct are found (we only care about 1 vs 2 vs more). The
        hidden card is treated as an irrelevant unknown: we only constrain the HAND."""
        candidates = [c for c in self.unknown_cards() if cards.card_name(c) not in self.opp_hand_lacks]
        if self.opp_hand_count > len(candidates):
            return 0  # over-constrained (shouldn't happen with consistent facts)
        seen = set()
        for combo in itertools.combinations(candidates, self.opp_hand_count):
            names = tuple(sorted(cards.card_name(c) for c in combo))
            if names in seen:
                continue
            if all(h in names for h in self.opp_hand_has):
                seen.add(names)
                if len(seen) >= cap:
                    break
        return len(seen)

    def knowledge_level(self) -> Optional[str]:
        """``"perfect"`` if the observer knows the opponent's exact hand (1 possibility), ``"binary"``
        if it is a 50-50 between 2, else ``None``."""
        n = self.possible_opp_hands(cap=3)
        return "perfect" if n == 1 else "binary" if n == 2 else None

    # --- the MCTS sampling seam --------------------------------------------------------

    def determinize(self, rng, use_knowledge: bool = True) -> GameState:
        """Sample a concrete GameState consistent with this information set.

        Distributes the unknown pool into the opponent's hand, their hidden card (if their king is
        unflipped), and muck (their setup-discard + the face-down leftover, which never matter).
        With ``use_knowledge`` (default), the sampled opponent HAND honors guess leaks: it excludes
        every ``opp_hand_lacks`` name and includes >=1 of every ``opp_hand_has`` name. Cards of a
        lacked name may still be the opponent's hidden card or muck -- the constraint is hand-only.
        ``use_knowledge=False`` reproduces uniform sampling (A/B benchmarking + back-compat)."""
        unknown = self.unknown_cards()
        pinned = self._pinned_opp_cards() & set(unknown)
        lacks = self.opp_hand_lacks if use_knowledge else frozenset()
        has = self.opp_hand_has if use_knowledge else frozenset()

        free = [int(c) for c in unknown if c not in pinned]
        as_search_rng(rng).shuffle(free)          # in place; see rng.py (a raw Generator still works)

        # Build the opponent's hand constructively (no rejection loops on tight constraints):
        hand = set(pinned)                                       # committed cards always in hand
        for nm in has:                                           # (a) force >=1 of each 'has' name
            if any(cards.card_name(c) == nm for c in hand):
                continue
            pick = next((c for c in free if c not in hand and cards.card_name(c) == nm), None)
            if pick is not None:
                hand.add(pick)
        for c in free:                                           # (b) fill from non-lacked cards
            if len(hand) >= self.opp_hand_count:
                break
            if c not in hand and cards.card_name(c) not in lacks:
                hand.add(c)
        if len(hand) < self.opp_hand_count:                      # (c) infeasible-constraint fallback
            for c in free:
                if len(hand) >= self.opp_hand_count:
                    break
                hand.add(c)

        opp_hand = tuple(sorted(hand))
        rest = [c for c in free if c not in hand]                # lacked/leftover cards -> hidden + muck

        opp_hidden: Optional[int] = None
        if self.opp_has_hidden and rest:
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

        # Carry the observer's knowledge into the reconstructed world (opp's beliefs are irrelevant
        # to a search from the observer's seat) so in-tree moves keep it consistent.
        hand_lacks = [frozenset(), frozenset()]
        hand_lacks[observer] = self.opp_hand_lacks
        hand_has = [frozenset(), frozenset()]
        hand_has[observer] = self.opp_hand_has

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
            hand_lacks=tuple(hand_lacks),
            hand_has=tuple(hand_has),
        )
