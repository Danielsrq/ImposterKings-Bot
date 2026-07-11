"""Card semantics: the ``resolve(state, action)`` dispatcher and every ability's effect.

This is the rules arbiter (bigtwo's ``combos.py`` analog). :mod:`state` owns the container and the
turn/stack plumbing and delegates ``GameState.apply`` here. Everything is expressed as copy-on-write
transitions on :class:`~imposterkings.state.GameState` via its ``advance`` / ``replace_top`` /
``with_`` helpers, so an action either pushes more decision steps (ability sub-choices, reaction
windows) or ends the turn when the resolution stack empties.

Key conventions:
- One ``ABILITY_MAY`` decision (and exactly one King's-Hand reaction window) gates each optional
  on-play ability. Inner choices (Soldier's package, Mystic's number, Judge's queue) are NOT
  separately counterable -- one window per played card.
- A "may" ability is countered by discarding BOTH King's Hand and the played card (removing it from
  the stack). Queen's mandatory disgrace and the passive overrides (Elder/Zealot/Warlord/Oathbound)
  are never counterable.
"""
from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Tuple

from . import cards, rules
from .actions import Action, ActionKind, StepKind
from .cards import Ability
from .state import GameState, PendingStep, StackCard

# Instance ids of the unique reaction cards (scanned once from the registry).
KINGSHAND_ID = next(i for i, d in enumerate(cards.CARD_DEFS) if d.ability == Ability.KINGSHAND)
ASSASSIN_ID = next(i for i, d in enumerate(cards.CARD_DEFS) if d.ability == Ability.ASSASSIN)

# Optional ("may") on-play abilities, each gated by an ABILITY_MAY decision. Soldier/Judge are NOT here:
# their guess is mandatory (you must name a card), so they go straight to ABILITY_GUESS on play.
_OPTIONAL_ONPLAY = frozenset({
    Ability.PRINCESS, Ability.SENTRY, Ability.MYSTIC, Ability.INQUISITOR, Ability.FOOL,
})

# Mandatory on-play guesses: no declare/decline -- you must name a card (King's Hand still counters
# after the name, in _resolve_guess). Inquisitor's guess is a "may" but is flattened into ABILITY_MAY
# (offered as {decline} + guesses) so it also opens the window only after the name is chosen.
_MANDATORY_GUESS = frozenset({Ability.SOLDIER, Ability.JUDGE})


# --- tiny immutable-sequence helpers ---------------------------------------------------

def _without(seq: Tuple[int, ...], item: int) -> Tuple[int, ...]:
    i = seq.index(item)
    return seq[:i] + seq[i + 1:]


def _without_index(seq: tuple, i: int) -> tuple:
    return seq[:i] + seq[i + 1:]


def _set_index(seq: tuple, i: int, val) -> tuple:
    return seq[:i] + (val,) + seq[i + 1:]


def _add_to_hand(hand: Tuple[int, ...], card: int) -> Tuple[int, ...]:
    return tuple(sorted(hand + (card,)))


def _know_add(knowledge, seat: int, name: str):
    """Guess-knowledge tuple ``(frozenset, frozenset)`` with ``name`` added at ``seat``."""
    k = list(knowledge)
    k[seat] = k[seat] | {name}
    return tuple(k)


def _know_drop(knowledge, seat: int, name: str):
    k = list(knowledge)
    k[seat] = k[seat] - {name}
    return tuple(k)


# --- legality (shared by generate.py and the win check) --------------------------------

def _can_play(state: GameState, card: int, player: int,
              v_top: Optional[int], lead, lead_royalty: bool) -> bool:
    if v_top is None:
        return True  # empty stack: first/fresh play is unrestricted
    if state.effective_hand_value(card) >= v_top:
        return True
    ability = cards.card_ability(card)
    if ability == Ability.OATHBOUND and v_top > 6 and len(state.hands[player]) >= 2:
        return True  # disgrace the beaten card; Oathbound sits at 6, then play any card (needs a 2nd)
    if ability == Ability.ELDER and lead_royalty:
        return True  # Elder plays over any royalty
    if ability == Ability.ZEALOT and state.kings[player] and lead is not None and not lead_royalty:
        return True  # Zealot plays over any non-royalty once its own king is flipped
    return False


def legal_play_cards(state: GameState, player: int) -> List[int]:
    """Distinct hand cards (one representative id per name) that may legally be played now."""
    v_top = state.leading_value()
    lead = state.leading
    lead_royalty = lead is not None and state.active_royalty(lead)
    out: List[int] = []
    seen = set()
    for c in state.hands[player]:
        name = cards.card_name(c)
        if name in seen:
            continue
        if _can_play(state, c, player, v_top, lead, lead_royalty):
            out.append(c)
            seen.add(name)
    return out


# --- placing a card on the stack + triggering its on-play ability ----------------------

def _land(state: GameState, card: int, actor: int, *, ascended: bool, v_top: Optional[int]):
    """Place ``card`` on the stack and return ``(new_state, substeps)`` for its on-play ability.

    Oathbound override (played FROM HAND over a card > 6): the BEATEN card is disgraced, the Oathbound
    stays live at value 6, and the substep is the free follow-up play. An ASCENDED Oathbound (from the
    antechamber) does NOT trigger this -- ascension already let it beat the card -- so it just lands at
    6 with no ability. Otherwise the card lands (Warlord at 9 if royalty present) and optional abilities
    yield an ABILITY_MAY substep; Queen disgraces beneath immediately.
    """
    ability = cards.card_ability(card)

    if (not ascended) and ability == Ability.OATHBOUND and v_top is not None and v_top > 6:
        new_hand = _without(state.hands[actor], card)
        beaten = replace(state.stack[-1], disgraced=True)         # disgrace the card it beat
        new_stack = state.stack[:-1] + (beaten, StackCard(card))  # Oathbound stays live at value 6
        st = state.with_(hands=_set_index(state.hands, actor, new_hand), stack=new_stack)
        return st, (PendingStep(StepKind.OATHBOUND_SECOND, actor, source=card),)

    override = 9 if (ability == Ability.WARLORD and state.royalty_present()) else None
    new_stack = state.stack + (StackCard(card, value_override=override),)
    if ascended:
        new_hands = state.hands
    else:
        new_hands = _set_index(state.hands, actor, _without(state.hands[actor], card))
    st = state.with_(hands=new_hands, stack=new_stack)

    if ability == Ability.QUEEN:
        beneath = tuple(replace(s, disgraced=True) for s in st.stack[:-1])
        st = st.with_(stack=beneath + (st.stack[-1],))
        return st, ()
    if ability in _MANDATORY_GUESS:
        # Mandatory guess (no decline). The King's-Hand window opens after the name is declared
        # (in _resolve_guess); the ability's effect only applies on a correct, uncountered guess.
        return st, (PendingStep(StepKind.ABILITY_GUESS, actor, source=card),)
    if ability in _OPTIONAL_ONPLAY:
        if ability == Ability.PRINCESS and not (st.hands[actor] and st.hands[1 - actor]):
            return st, ()   # a swap needs a card from EACH side -> no "Use ability?" window at all
        return st, (PendingStep(StepKind.ABILITY_MAY, actor, source=card),)
    return st, ()


def _proceed(state: GameState, substeps: tuple, *, pop_top: bool, end_turn_player: int) -> GameState:
    """Push ``substeps`` either by popping the answered top step (normal play) or onto an empty
    stack (ascension). With no substeps the turn is over and the opponent's turn begins."""
    if pop_top:
        return state.advance(*substeps)
    if not substeps:
        return state._begin_turn(1 - end_turn_player)
    return state.with_(pending=tuple(reversed(substeps)))


def _resolve_ascend(state: GameState, actor: int) -> GameState:
    """Answer a StepKind.ASCEND (forced): dequeue the front card, land it, trigger its ability. Pops the
    ASCEND step (``pop_top=True``) just like MAIN+PLAY_CARD, so with no substeps the opponent's turn
    begins and with substeps the ascended card's ability sub-decisions surface next."""
    ante = state.antechambers[actor]
    card = ante[0]
    st = state.with_(antechambers=_set_index(state.antechambers, actor, ante[1:]))
    st, substeps = _land(st, card, actor, ascended=True, v_top=None)
    return _proceed(st, substeps, pop_top=True, end_turn_player=actor)


# --- stack mutation helpers ------------------------------------------------------------

def _stack_index_of(state: GameState, card: int) -> int:
    for i in range(len(state.stack) - 1, -1, -1):
        if state.stack[i].card == card:
            return i
    raise ValueError(f"card {card} not on stack")


def _disgrace_card(state: GameState, card: int) -> GameState:
    i = _stack_index_of(state, card)
    return state.with_(stack=_set_index(state.stack, i, replace(state.stack[i], disgraced=True)))


def _set_override(state: GameState, card: int, value: int) -> GameState:
    i = _stack_index_of(state, card)
    return state.with_(stack=_set_index(state.stack, i, replace(state.stack[i], value_override=value)))


def _sentry_targets(state: GameState, source: int) -> List[int]:
    return [i for i, sc in enumerate(state.stack)
            if not sc.disgraced and not state.active_royalty(sc) and sc.card != source]


def _fool_targets(state: GameState, source: int) -> List[int]:
    """Fool may take back any non-disgraced stack card except itself."""
    return [i for i, sc in enumerate(state.stack) if not sc.disgraced and sc.card != source]


# --- ability resolution branches -------------------------------------------------------

def _begin_resolution(state: GameState, source: int, owner: int) -> GameState:
    """King's Hand was declined on a bare declaration (Sentry/Princess): push the ability's own swap
    sub-decisions. Mystic/Fool are flattened (their parameter is chosen before the window), so they
    never reach here. The current top is the reaction step, popped by ``advance``."""
    ability = cards.card_ability(source)
    if ability == Ability.PRINCESS:
        if not state.hands[owner] or not state.hands[1 - owner]:
            return state.advance()  # a swap needs a card from each player
        return state.advance(PendingStep(StepKind.ABILITY_HAND_CARD, owner, source=source))
    if ability == Ability.SENTRY:
        st = _disgrace_card(state, source)
        if not _sentry_targets(st, source) or not st.hands[owner]:
            return st.advance()
        return st.advance(PendingStep(StepKind.ABILITY_STACK_TARGET, owner, source=source, limit=1))
    return state.advance()


def _resolve_guess(state: GameState, step: PendingStep, action: Action) -> GameState:
    """A guess is committed and made public. Open a King's-Hand window only if the guess actually
    lands (the named card is held), since otherwise there is no effect to counter."""
    source, owner = step.source, step.actor
    defender = 1 - owner
    held = any(cards.card_name(c) == action.name for c in state.hands[defender])
    if held:
        # The guesser now knows the defender holds >=1 of that name (a King's-Hand counter discards
        # the KH card, not the guessed one, so the fact stands).
        return state.advance(
            PendingStep(StepKind.REACTION_KINGSHAND, defender,
                        source=source, against=source, guess=action.name),
            hand_has=_know_add(state.hand_has, owner, action.name),
            hand_lacks=_know_drop(state.hand_lacks, owner, action.name),
        )
    # Wrong guess -> nothing happens, but the guesser learns the defender's hand lacks that name.
    return state.advance(hand_lacks=_know_add(state.hand_lacks, owner, action.name),
                         hand_has=_know_drop(state.hand_has, owner, action.name))


def _after_guess_kingshand_declined(state: GameState, step: PendingStep) -> GameState:
    """The defender did not counter a landed guess: apply the guess ability's effect."""
    source, name = step.source, step.guess
    owner = 1 - step.actor              # the active player who guessed
    defender = step.actor
    ability = cards.card_ability(source)
    if ability == Ability.INQUISITOR:
        held = tuple(c for c in state.hands[defender] if cards.card_name(c) == name)
        new_def = tuple(c for c in state.hands[defender] if c not in held)
        new_ante = state.antechambers[defender] + held
        # All copies of the name are extracted to the (public) antechamber -> the hand now lacks it.
        return state.advance(
            hands=_set_index(state.hands, defender, new_def),
            antechambers=_set_index(state.antechambers, defender, new_ante),
            hand_lacks=_know_add(state.hand_lacks, owner, name),
        )
    if ability == Ability.JUDGE:
        return state.advance(PendingStep(StepKind.ABILITY_HAND_CARD, owner, source=source, guess=name))
    if ability == Ability.SOLDIER:
        # A correct (uncountered) guess immediately grants +2 -- it is tied to the guess, not a
        # separate choice. The only remaining decision is which 0-3 stack cards to disgrace.
        st = _set_override(state, source, cards.card_value(source) + rules.SOLDIER_BONUS)
        return st.advance(PendingStep(StepKind.ABILITY_STACK_TARGET, owner,
                                      source=source, limit=rules.SOLDIER_DISGRACE_CAP))
    return state.advance()


def _soldier_disgrace(state: GameState, chosen: Tuple[int, ...]) -> GameState:
    new_stack = list(state.stack)
    for pos in set(chosen):
        new_stack[pos] = replace(new_stack[pos], disgraced=True)
    return state.advance(stack=tuple(new_stack))


def _flip_resolve(state: GameState, flipper: int) -> GameState:
    """Complete a king-flip: disgrace the top card, mark the king used, take the hidden card."""
    top = state.stack[-1]
    new_stack = state.stack[:-1] + (replace(top, disgraced=True),)
    new_kings = _set_index(state.kings, flipper, True)
    hid = state.hidden[flipper]
    if hid is not None:
        new_hands = _set_index(state.hands, flipper, _add_to_hand(state.hands[flipper], hid))
        new_hidden = _set_index(state.hidden, flipper, None)
    else:
        new_hands, new_hidden = state.hands, state.hidden
    return state.advance(stack=new_stack, kings=new_kings, hands=new_hands, hidden=new_hidden)


# --- the dispatcher --------------------------------------------------------------------

def resolve(state: GameState, action: Action) -> GameState:
    step = state.pending[-1]
    k = step.kind
    actor = step.actor

    if k == StepKind.SETUP_HIDE:
        return state.advance(
            hands=_set_index(state.hands, actor, _without(state.hands[actor], action.card)),
            hidden=_set_index(state.hidden, actor, action.card),
        )

    if k == StepKind.SETUP_DISCARD:
        return state.advance(
            hands=_set_index(state.hands, actor, _without(state.hands[actor], action.card)),
            setup_discard=_set_index(state.setup_discard, actor, action.card),
        )

    if k == StepKind.ASCEND:
        return _resolve_ascend(state, actor)

    if k == StepKind.MAIN:
        if action.kind == ActionKind.FLIP_KING:
            return state.advance(PendingStep(StepKind.REACTION_ASSASSIN, 1 - actor))
        st, substeps = _land(state, action.card, actor, ascended=False, v_top=state.leading_value())
        return _proceed(st, substeps, pop_top=True, end_turn_player=actor)

    if k == StepKind.OATHBOUND_SECOND:
        st, substeps = _land(state, action.card, actor, ascended=False, v_top=state.leading_value())
        return _proceed(st, substeps, pop_top=True, end_turn_player=actor)

    if k == StepKind.ABILITY_MAY:
        if action.kind == ActionKind.DECLINE_ABILITY:
            return state.advance()
        if action.kind == ActionKind.DECLARE_ABILITY:   # Sentry/Princess: window on the bare declaration
            return state.advance(PendingStep(StepKind.REACTION_KINGSHAND, 1 - actor,
                                             source=step.source, against=step.source))
        # Flattened abilities declare their PARAMETER here; open the King's-Hand window carrying it.
        if action.kind == ActionKind.GUESS_CARD:        # Inquisitor -- window only if the guess lands
            return _resolve_guess(state, step, action)
        if action.kind == ActionKind.CHOOSE_NUMBER:     # Mystic
            return state.advance(PendingStep(StepKind.REACTION_KINGSHAND, 1 - actor,
                                             source=step.source, against=step.source, number=action.number))
        if action.kind == ActionKind.CHOOSE_STACK_TARGET:  # Fool
            return state.advance(PendingStep(StepKind.REACTION_KINGSHAND, 1 - actor,
                                             source=step.source, against=step.source, picked=action.target))

    if k == StepKind.REACTION_KINGSHAND:
        if action.kind == ActionKind.REVEAL_KINGSHAND:
            reactor = step.actor
            i = _stack_index_of(state, step.source)
            # A card counters a card: both are expended and the interaction is undone. The countered
            # card leaves the stack (leading reverts to what's beneath), both go to discard, and the turn
            # RETURNS to the active player, who must play again (must beat the reverted leading). The
            # active player is still ``turn_player`` -- the reaction only moved ``pending[-1].actor``.
            st = state.with_(
                stack=_without_index(state.stack, i),
                discard=state.discard + (step.source, KINGSHAND_ID),
                hands=_set_index(state.hands, reactor, _without(state.hands[reactor], KINGSHAND_ID)),
            )
            return st._begin_turn(st.turn_player, ascend=False)
        # Declined: apply the declared ability. Guesses carry their name; the flattened Mystic/Fool carry
        # their parameter (number / stack index); Sentry/Princess fall through to their own sub-decisions.
        if step.guess is not None:
            return _after_guess_kingshand_declined(state, step)
        ability = cards.card_ability(step.source)
        if ability == Ability.MYSTIC:
            st = _disgrace_card(state, step.source)
            return st.advance(muted_values=st.muted_values | {step.number})
        if ability == Ability.FOOL:
            owner = 1 - step.actor
            grabbed = state.stack[step.picked].card
            return state.advance(
                stack=_without_index(state.stack, step.picked),
                hands=_set_index(state.hands, owner, _add_to_hand(state.hands[owner], grabbed)),
            )
        return _begin_resolution(state, step.source, owner=1 - step.actor)

    if k == StepKind.ABILITY_GUESS:
        return _resolve_guess(state, step, action)

    if k == StepKind.ABILITY_NUMBER:
        return state.advance(muted_values=state.muted_values | {action.number})

    if k == StepKind.ABILITY_HAND_CARD:
        if action.kind == ActionKind.STOP:           # Judge: decline to queue
            return state.advance()
        ability = cards.card_ability(step.source)
        card = action.card
        if ability == Ability.PRINCESS:
            return state.advance(PendingStep(StepKind.ABILITY_SWAP_RESPOND, 1 - actor,
                                             source=step.source, picked=card))
        if ability == Ability.SENTRY:
            pos = step.picked
            grabbed = state.stack[pos].card
            return state.advance(
                stack=_set_index(state.stack, pos, StackCard(card)),
                hands=_set_index(state.hands, actor, _add_to_hand(_without(state.hands[actor], card), grabbed)),
            )
        if ability == Ability.JUDGE:
            return state.advance(
                hands=_set_index(state.hands, actor, _without(state.hands[actor], card)),
                antechambers=_set_index(state.antechambers, actor, state.antechambers[actor] + (card,)),
            )

    if k == StepKind.ABILITY_SWAP_RESPOND:  # Princess: responder picks their give-card
        responder = step.actor
        princess = 1 - responder
        give_p, give_o = step.picked, action.card
        new_hands = list(state.hands)
        new_hands[princess] = _add_to_hand(_without(state.hands[princess], give_p), give_o)
        new_hands[responder] = _add_to_hand(_without(state.hands[responder], give_o), give_p)
        return state.advance(hands=tuple(new_hands))

    if k == StepKind.ABILITY_STACK_TARGET:
        ability = cards.card_ability(step.source)
        if ability == Ability.FOOL:
            pos = action.target
            grabbed = state.stack[pos].card
            return state.advance(
                stack=_without_index(state.stack, pos),
                hands=_set_index(state.hands, actor, _add_to_hand(state.hands[actor], grabbed)),
            )
        if ability == Ability.SENTRY:
            return state.advance(PendingStep(StepKind.ABILITY_HAND_CARD, actor,
                                             source=step.source, picked=action.target))
        if ability == Ability.SOLDIER:
            if action.kind == ActionKind.STOP:
                return _soldier_disgrace(state, step.chosen)
            new_chosen = step.chosen + (action.target,)
            if len(new_chosen) >= rules.SOLDIER_DISGRACE_CAP:
                return _soldier_disgrace(state, new_chosen)
            return state.replace_top(replace(step, chosen=new_chosen, limit=step.limit - 1))

    if k == StepKind.REACTION_ASSASSIN:
        if action.kind == ActionKind.REVEAL_ASSASSIN:
            reactor = step.actor
            # A revealed Assassin is public and spent regardless of outcome -> commit it to the
            # discard now, so it leaves the unknown pool (keeps determinized worlds consistent when
            # the nested King's-Hand window is itself the search root).
            st = state.with_(
                hands=_set_index(state.hands, reactor, _without(state.hands[reactor], ASSASSIN_ID)),
                discard=state.discard + (ASSASSIN_ID,),
            )
            return st.advance(PendingStep(StepKind.REACTION_KH_VS_ASSASSIN, st.turn_player))
        return _flip_resolve(state, state.turn_player)

    if k == StepKind.REACTION_KH_VS_ASSASSIN:
        flipper = step.actor
        assassin_player = 1 - flipper
        if action.kind == ActionKind.REVEAL_KINGSHAND:
            st = state.with_(
                hands=_set_index(state.hands, flipper, _without(state.hands[flipper], KINGSHAND_ID)),
                discard=state.discard + (KINGSHAND_ID,),  # Assassin already discarded on reveal
            )
            return _flip_resolve(st, flipper)
        return state.with_(winner=assassin_player, pending=())  # Assassin resolves -> instant win

    raise ValueError(f"resolve: unhandled step kind {k!r} for action {action!r}")
