"""Featurization v2.2: the token set IS the deck -- 18 fixed card-instance tokens + 2 king tokens.

Design record: `attention_exploration.md` ("Featurization v2.2"). Key properties:
- FIXED sequence `[18 x card(46) | 2 x king(4) | board(4) | phase(15) | action(51)]` -> S = 24 with CLS
  (model-side). No padding, no mask, stable heatmap axes for the whole game.
- **Zone posterior** (12): location as a probability distribution -- a delta for seen cards, a
  slot-proportional hypergeometric spread over the hidden zones for unseen ones, reshaped by
  `hand_has`/`hand_lacks`. Sums to 1. Computed from counts + knowledge only (determinization-invariant);
  this block is the belief-net socket.
- **Mute-as-deletion**: a muted card's mechanic-signature bits are ZEROED (the featurizer mirrors the
  engine rule that muting strips abilities and tags); `is_muted` stays explicit; base value immutable.
- **Phase context on entities**: `pending_source` / `pending_guess_target` / `pending_mute_target` are
  transient per-card flags (chain-scoped, all-zero outside their windows); the phase token shrinks to the
  step-kind one-hot.
- **Canonical duplicate rule**: hidden copies of 2-copy names are exchangeable -- visible placements fill
  the LOWER instance slots in a deterministic order; same infoset -> same encoding.

v1 (`features.py`) is untouched; consumers dispatch on the checkpoint's feature version.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..actions import Action, ActionKind, StepKind
from ..cards import CARD_NAMES, card_ids_for_name, card_name, card_value
from ..infoset import InformationSet
from .features import _ABILITY_CAT, _N_STEP, _TYPE_BASE, _TYPE_IX, _action_vec
from .features import ACTION_DIM  # 51: the prong-B action encoding, reused verbatim

_NAMES: List[str] = list(CARD_NAMES)
_COPIES: Dict[str, int] = {n: len(card_ids_for_name(n)) for n in _NAMES}
N_CARDS = 18

# Instance slots in fixed order: names in CARD_NAMES order, duplicated names get consecutive slots.
INSTANCE_NAMES: List[str] = [n for n in _NAMES for _ in range(_COPIES[n])]
INSTANCE_LABELS: List[str] = [f"{n}#{k}" if _COPIES[n] > 1 else n
                              for n in _NAMES for k in range(_COPIES[n])]
assert len(INSTANCE_NAMES) == N_CARDS

ZONES = ["my_hand", "my_hidden", "my_setup", "their_hand", "their_hidden", "their_setup",
         "my_ante", "their_ante", "stack", "discard", "faceup", "facedown"]
_Z = {z: i for i, z in enumerate(ZONES)}
_NZONE = len(ZONES)                                   # 12
_HIDDEN_ZONES = ("their_hand", "their_hidden", "their_setup", "facedown")

# Card token layout: name(14) | power/9 | base/9 | mechanics(8) | state(10) | zone posterior(12) = 46
_MECH_OFF = 16
_STATE_OFF = _MECH_OFF + 8                            # 24
STATE_FIELDS = ["is_muted", "disgraced", "is_leading", "stack_depth", "is_legal_now",
                "is_candidate_action", "pending_selected", "pending_source",
                "pending_guess_target", "pending_mute_target"]
_ZONE_OFF = _STATE_OFF + len(STATE_FIELDS)            # 34
CARD_DIM = _ZONE_OFF + _NZONE                          # 46
CAND_COL = _STATE_OFF + STATE_FIELDS.index("is_candidate_action")

KING_DIM = 4                                           # owner mine/theirs (2) | flipped | flip_legal_now
BOARD_DIM = 4                                          # hand sizes | is_my_turn | is_reaction_window
PHASE_DIM = _N_STEP                                    # 15: step-kind only (context lives on the cards)


@dataclass
class Tokens2:
    """v2 attention-model input: FIXED-shape token set (no ragged storage, no mask)."""
    cards: np.ndarray          # [18, 46]
    kings: np.ndarray          # [2, 4]   (mine, theirs)
    board: np.ndarray          # [4]
    phase: np.ndarray          # [15]
    action: np.ndarray         # [51]  (all-zero when no candidate action is stamped)
    labels: List[str]          # len 18, == INSTANCE_LABELS (fixed; kept for API symmetry)


@dataclass
class _Placement:
    """One resolved location for a card instance: a zone delta (visible) or None (unseen -> spread)."""
    zone: Optional[str]                                # None = unseen
    engine_id: Optional[int] = None                    # the engine card id, when visible
    disgraced: bool = False
    is_leading: bool = False
    stack_depth: float = 0.0
    power: Optional[float] = None                      # effective value when on-stack (override-aware)


def _visible_placements(view: InformationSet) -> Dict[str, List[_Placement]]:
    """Every card the observer can SEE, grouped by name, in a canonical order (zone index, stack depth)."""
    out: Dict[str, List[_Placement]] = {n: [] for n in _NAMES}

    def put(zone, cid, **kw):
        out[card_name(cid)].append(_Placement(zone, engine_id=cid, **kw))

    for c in view.own_hand:
        put("my_hand", c)
    if view.own_hidden is not None:
        put("my_hidden", view.own_hidden)
    if view.own_setup_discard is not None:
        put("my_setup", view.own_setup_discard)
    n_stack = len(view.stack)
    for i, sc in enumerate(view.stack):
        eff = 0.0 if sc.disgraced else (
            3.0 if card_value(sc.card) in view.muted_values else
            float(sc.value_override if sc.value_override is not None else card_value(sc.card)))
        put("stack", sc.card, disgraced=sc.disgraced, is_leading=(i == n_stack - 1),
            stack_depth=(i + 1) / n_stack, power=eff)
    for c in view.antechambers[view.observer]:
        put("my_ante", c)
    for c in view.antechambers[1 - view.observer]:
        put("their_ante", c)
    for c in view.discard:
        put("discard", c)
    if view.leftover_faceup is not None and view.leftover_faceup >= 0:
        put("faceup", view.leftover_faceup)

    for n in _NAMES:                                   # canonical order: zone index, then stack depth
        out[n].sort(key=lambda p: (_Z[p.zone], p.stack_depth))
    return out


def _hidden_slots(view: InformationSet, n_unseen: int) -> Dict[str, float]:
    """Slot counts per hidden zone. their_hand/their_hidden are known directly; the remainder of the
    unseen pool sits in {their_setup, facedown} (split 1/1 in a real deal; residual-driven so scenario
    states without leftovers stay consistent)."""
    hand = float(view.opp_hand_count)
    hidden = 1.0 if view.opp_has_hidden else 0.0
    rest = max(0.0, n_unseen - hand - hidden)
    facedown = min(1.0, rest)
    setup = rest - facedown
    return {"their_hand": hand, "their_hidden": hidden, "their_setup": setup, "facedown": facedown}


def tokenize_state(view: InformationSet, legal_moves=None) -> Tokens2:
    """The action-independent v2 encoding (compute ONCE per position; stamp candidates via
    :func:`with_action`). ``legal_moves`` skips the internal ``view.legal_moves()`` determinization."""
    obs = view.observer
    muted = view.muted_values

    if legal_moves is None and view.to_play == obs:
        legal_moves = view.legal_moves()
    legal_names = set()
    can_flip = False
    for a in (legal_moves or ()):
        if a.kind == ActionKind.PLAY_CARD and a.card is not None:
            legal_names.add(card_name(a.card))
        elif a.kind == ActionKind.FLIP_KING:
            can_flip = True
    ascend_front = (view.antechambers[obs][0] if view.antechambers[obs] and view.to_play == obs
                    else None)

    placed = _visible_placements(view)
    n_visible = sum(len(v) for v in placed.values())
    n_unseen = N_CARDS - n_visible
    slots = _hidden_slots(view, n_unseen)

    # hand_has: pin ONE canonical unseen copy per has-name into their_hand (documented approximation for
    # has-constrained duplicates); reduce the open hand slots accordingly.
    has_pins: Dict[str, int] = {}
    for nm in view.opp_hand_has:
        unseen_copies = _COPIES[nm] - len(placed[nm])
        if unseen_copies > 0 and slots["their_hand"] >= 1.0:
            has_pins[nm] = 1
            slots["their_hand"] -= 1.0
    open_total = sum(slots.values())

    # pending-chain context (transient flags)
    step = view.pending[-1] if view.pending else None
    pend_src_name = card_name(step.source) if step is not None and step.source is not None else None
    pend_src_slot = 0                                   # which copy of the source name carries the flag
    if pend_src_name is not None:
        vis_ids = [p.engine_id for p in placed[pend_src_name]]
        if step.source in vis_ids:
            pend_src_slot = vis_ids.index(step.source)  # id-matched when the source is visible
    pend_guess = getattr(step, "guess", None) if step is not None else None
    pend_number = getattr(step, "number", None) if step is not None else None
    selected_ids = set()
    if step is not None and getattr(step, "chosen", None):
        for pos in step.chosen:
            if 0 <= pos < len(view.stack):
                selected_ids.add(view.stack[pos].card)

    cards = np.zeros((N_CARDS, CARD_DIM), np.float32)
    slot_i = 0
    for nm in _NAMES:
        tix = _TYPE_IX[nm]
        base = float(_TYPE_BASE[tix])
        is_mut = base in muted
        vis = placed[nm]
        pins = has_pins.get(nm, 0)
        for k in range(_COPIES[nm]):
            v = cards[slot_i]
            v[tix] = 1.0
            v[15] = base / 9.0                                        # base (immutable)
            if is_mut:
                v[_STATE_OFF + 0] = 1.0                               # is_muted; mechanics stay ZERO
            else:
                v[_MECH_OFF:_MECH_OFF + 8] = _ABILITY_CAT[tix]        # mechanic signature
            if pend_src_name == nm and k == pend_src_slot:
                v[_STATE_OFF + 7] = 1.0                               # pending_source (one copy only)
            if pend_guess == nm:
                v[_STATE_OFF + 8] = 1.0                               # pending_guess_target
            if pend_number is not None and int(base) == int(pend_number):
                v[_STATE_OFF + 9] = 1.0                               # pending_mute_target

            if k < len(vis):                                          # a SEEN copy: zone delta + state
                p = vis[k]
                v[_ZONE_OFF + _Z[p.zone]] = 1.0
                v[14] = (p.power if p.power is not None else (3.0 if is_mut else base)) / 9.0
                v[_STATE_OFF + 1] = 1.0 if p.disgraced else 0.0
                v[_STATE_OFF + 2] = 1.0 if p.is_leading else 0.0
                v[_STATE_OFF + 3] = p.stack_depth
                if p.engine_id in selected_ids:
                    v[_STATE_OFF + 6] = 1.0                           # pending_selected
                if p.zone == "my_hand" and nm in legal_names:
                    v[_STATE_OFF + 4] = 1.0                           # is_legal_now
                if p.zone == "my_ante" and p.engine_id == ascend_front:
                    v[_STATE_OFF + 4] = 1.0                           # ASCEND: the queued card is the play
            else:                                                     # an UNSEEN copy: posterior spread
                v[14] = (3.0 if is_mut else base) / 9.0
                if pins > 0:
                    v[_ZONE_OFF + _Z["their_hand"]] = 1.0             # hand_has pin (canonical copy)
                    pins -= 1
                else:
                    zs = dict(slots)
                    if nm in view.opp_hand_lacks:
                        zs["their_hand"] = 0.0                        # hand_lacks: excluded, renormalize
                    tot = sum(zs.values())
                    if tot <= 0:                                      # degenerate (scenario states)
                        v[_ZONE_OFF + _Z["facedown"]] = 1.0
                    else:
                        for z, w in zs.items():
                            v[_ZONE_OFF + _Z[z]] = w / tot
            slot_i += 1

    kings = np.zeros((2, KING_DIM), np.float32)
    kings[0, 0] = 1.0
    kings[0, 2] = 1.0 if view.kings[obs] else 0.0
    kings[0, 3] = 1.0 if can_flip else 0.0
    kings[1, 1] = 1.0
    kings[1, 2] = 1.0 if view.kings[1 - obs] else 0.0

    board = np.array([len(view.own_hand) / 8.0, view.opp_hand_count / 7.0,
                      1.0 if view.to_play == obs else 0.0,
                      1.0 if view.to_play != view.turn_player else 0.0], np.float32)

    phase = np.zeros(PHASE_DIM, np.float32)
    if step is not None:
        phase[list(StepKind).index(step.kind)] = 1.0

    return Tokens2(cards, kings, board, phase, np.zeros(ACTION_DIM, np.float32),
                   list(INSTANCE_LABELS))


def with_action(tok: Tokens2, action: Optional[Action]) -> Tokens2:
    """Stamp a candidate action (copy-on-write). Fixed token set -> no transient '*' token: a guess/claim
    simply flags the named instances (their zone posterior already says how plausible the claim is)."""
    if action is None:
        return tok
    cards = tok.cards.copy()
    my_hand_col = _ZONE_OFF + _Z["my_hand"]
    my_ante_col = _ZONE_OFF + _Z["my_ante"]

    def flag_name(nm: str, require_mine: bool):
        for i, inm in enumerate(INSTANCE_NAMES):
            if inm != nm:
                continue
            if require_mine and cards[i, my_hand_col] < 1.0 and cards[i, my_ante_col] < 1.0:
                continue
            cards[i, CAND_COL] = 1.0
            if require_mine:
                return                                  # first owned copy only (canonical)

    if action.card is not None:
        flag_name(card_name(action.card), require_mine=True)
    if action.name is not None:
        flag_name(action.name, require_mine=False)      # guess/claim: every instance of the name
    return Tokens2(cards, tok.kings, tok.board, tok.phase, _action_vec(action), tok.labels)


def tokenize(view: InformationSet, action: Optional[Action] = None) -> Tokens2:
    """v2 twin of ``features.tokenize`` -- equivalent to ``with_action(tokenize_state(view), action)``."""
    return with_action(tokenize_state(view), action)


def card_token_fields() -> List[str]:
    """One label per dim of the v2 card token (interpretability twin of features.card_token_fields)."""
    from .features import card_token_fields as _v1
    mech = [f"mech:{s.split(':', 1)[1]}" for s in _v1()[14:22]]      # reuse the v1 category names
    return ([f"id:{n}" for n in _NAMES] + ["power", "base"] + mech
            + [f"st:{s}" for s in STATE_FIELDS] + [f"zone:{z}" for z in ZONES])
