"""Featurization v3: the late-fusion token set (Study 11) -- a candidate-INDEPENDENT state + pointer candidates.

Design record: `attention_exploration.md` ("Study 11"). The state is encoded ONCE per decision; every legal
move becomes a candidate that cross-attends to it (``latefusion_model``). So nothing about the move may
leak into the state tokens (v2's ``is_candidate_action`` stamp is gone) and the move refers to the state by
POINTER instead of by card type.

State: a FIXED sequence of S = 23 tokens

    [18 x card(44) | hand:mine | hand:theirs | king:mine | king:theirs | decision(40)]

- **card** = v2's card token minus ``base`` (redundant with the name) and ``is_candidate_action`` (a move
  property), with the scalar ``stack_depth`` replaced by an INTEGER ``card_depth`` (depth from the top of the
  court, capped; -1 = off-stack) that the model embeds, plus a ``picked_target`` role flag. The zone
  posterior, mute-as-deletion and the canonical duplicate rule are v2's, computed by ``features2`` itself.
- **hand** (mine / theirs): size, has-hidden-card, antechamber count. **king** (mine / theirs): flipped,
  flip-legal-now. Owner is a token-type id (model side), not a feature.
- **decision** = the pending step being answered: step kind, whether the decider is the REACTOR (it is not
  the turn player's own step), the carried guess name / Mystic number, the Soldier picks left and picks
  made. ``src_ptr`` points at the source card's slot. v2's ``is_my_turn`` is dropped: rows are always
  featurized from the decider's seat, so it was 1.0 in every training row.

``picked`` (a PendingStep field) is encoded ONLY where it is a PUBLIC stack position -- the Fool's chosen
target carried into the opponent's King's-Hand window and the Sentry's grabbed position carried into its
hand-card step. At ``ABILITY_SWAP_RESPOND`` it is the Princess's still-hidden give-card
(``infoset._pinned_opp_cards``) and must never reach the responder's tokens.

Candidates: integer records ``(kind, ptr_a, ptr_b, name, number)`` -- ``ptr_*`` index the 23 state tokens
(-1 = none), ``name`` a CARD_NAMES index (the model averages that name's instances), ``number`` a Mystic
value 1..8 (-1 = none).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..actions import Action, ActionKind, StepKind
from ..cards import CARD_NAMES, card_name, card_value
from ..infoset import InformationSet
from . import features2 as F2

N_CARDS = F2.N_CARDS                                   # 18
_NAMES: List[str] = list(CARD_NAMES)
_NAME_IX: Dict[str, int] = {n: i for i, n in enumerate(_NAMES)}
INSTANCE_NAMES = F2.INSTANCE_NAMES
INSTANCE_LABELS = F2.INSTANCE_LABELS
FIRST_SLOT: Dict[str, int] = {n: INSTANCE_NAMES.index(n) for n in _NAMES}

# --- state token layout ------------------------------------------------------------------------------
HAND_MINE, HAND_THEIRS, KING_MINE, KING_THEIRS, DECISION = 18, 19, 20, 21, 22
S_STATE = 23
STATE_LABELS: List[str] = list(INSTANCE_LABELS) + ["hand:mine", "hand:theirs", "king:mine", "king:theirs",
                                                   "decision"]

# card = name(14) | power | mechanics(8) | state(8) | picked_target | zone posterior(12) = 44
_V2_KEEP = (list(range(0, 15))                         # name one-hot + effective power
            + list(range(F2._MECH_OFF, F2._MECH_OFF + 8))
            + [F2._STATE_OFF + F2.STATE_FIELDS.index(f) for f in
               ("is_muted", "disgraced", "is_leading", "is_legal_now", "pending_selected", "pending_source",
                "pending_guess_target", "pending_mute_target")])
_PICKED_COL = len(_V2_KEEP)                            # 31
_ZONE_OFF = _PICKED_COL + 1                            # 32
CARD_DIM = _ZONE_OFF + F2._NZONE                       # 44
DEPTH_CAP = 6                                          # depths >= 6 share one embedding; -1 = off-stack

HAND_DIM = 3                                           # size/8 | has_hidden | antechamber count/3
KING_DIM = 2                                           # flipped | flip_legal_now (mine only)
_N_STEP = len(StepKind)                                # 15
_N_NUM = 8                                             # Mystic values 1..8
DECISION_DIM = _N_STEP + 1 + len(_NAMES) + _N_NUM + 2  # 40
N_KINDS = len(ActionKind)                              # 14
MAX_CANDS = 15                                         # GUESS_CARD offers all 14 names + DECLINE

# picked is a public stack position at exactly these steps (see module docstring)
_PICKED_PUBLIC = (StepKind.REACTION_KINGSHAND, StepKind.ABILITY_HAND_CARD)
# moves with no card payload point at the decision token (ptr_a) and its source card (ptr_b)
_STEP_ANSWERS = (ActionKind.DECLARE_ABILITY, ActionKind.DECLINE_ABILITY, ActionKind.DECLINE_REACTION,
                 ActionKind.STOP)
_REVEAL_CARD = {ActionKind.REVEAL_KINGSHAND: "KingsHand", ActionKind.REVEAL_ASSASSIN: "Assassin"}

# static instance maps the model uses for name / number candidates
NAME_MASK = np.zeros((len(_NAMES), N_CARDS), np.float32)      # [14, 18]: instances of each name
VALUE_MASK = np.zeros((_N_NUM + 1, N_CARDS), np.float32)      # [9, 18]: instances of each BASE value
for _i, _n in enumerate(INSTANCE_NAMES):
    NAME_MASK[_NAME_IX[_n], _i] = 1.0
    _v = int(F2._TYPE_BASE[F2._TYPE_IX[_n]])
    if _v <= _N_NUM:                                           # value 9 (Queen/Princess) is never a target
        VALUE_MASK[_v, _i] = 1.0


@dataclass
class State3:
    """v3 state: fixed shape, candidate-independent. ``slot_of`` maps a VISIBLE engine card id -> its slot."""
    cards: np.ndarray          # [18, 44]
    card_depth: np.ndarray     # [18] int8, -1 off-stack, else min(depth from top, DEPTH_CAP)
    hands: np.ndarray          # [2, 3]   (mine, theirs)
    kings: np.ndarray          # [2, 2]   (mine, theirs)
    decision: np.ndarray       # [40]
    src_ptr: int               # slot of the pending step's source card, -1 if none / not visible
    slot_of: Dict[int, int]


@dataclass
class Cands3:
    """Pointer candidates, one row per move (unpadded)."""
    kind: np.ndarray           # [N] int8   ActionKind.value - 1
    ptr_a: np.ndarray          # [N] int8   state-token index or -1
    ptr_b: np.ndarray          # [N] int8
    name: np.ndarray           # [N] int8   CARD_NAMES index or -1
    number: np.ndarray         # [N] int8   1..8 or -1


def slot_map(view: InformationSet) -> Dict[int, int]:
    """Visible engine id -> instance slot, by v2's canonical duplicate rule: the k-th VISIBLE copy of a name
    (in ``_visible_placements`` order: zone index, then stack depth) takes slot ``FIRST_SLOT[name] + k``."""
    placed = F2._visible_placements(view)
    out: Dict[int, int] = {}
    for nm in _NAMES:
        for k, p in enumerate(placed[nm]):
            out[p.engine_id] = FIRST_SLOT[nm] + k
    return out


def tokenize_state3(view: InformationSet, legal_moves=None) -> State3:
    """The v3 state tokens (compute ONCE per decision). ``legal_moves`` skips the determinization that
    ``features2.tokenize_state`` would otherwise run for ``is_legal_now`` / ``flip_legal_now``."""
    obs = view.observer
    t2 = F2.tokenize_state(view, legal_moves=legal_moves)
    slot_of = slot_map(view)

    cards = np.zeros((N_CARDS, CARD_DIM), np.float32)
    cards[:, :_PICKED_COL] = t2.cards[:, _V2_KEEP]
    cards[:, _ZONE_OFF:] = t2.cards[:, F2._ZONE_OFF:]

    depth = np.full(N_CARDS, -1, np.int8)
    n_stack = len(view.stack)
    for i, sc in enumerate(view.stack):
        depth[slot_of[sc.card]] = min(n_stack - 1 - i, DEPTH_CAP)

    step = view.pending[-1] if view.pending else None
    if (step is not None and step.kind in _PICKED_PUBLIC and step.picked is not None
            and 0 <= step.picked < n_stack):
        cards[slot_of[view.stack[step.picked].card], _PICKED_COL] = 1.0

    opp = 1 - obs
    hands = np.array([[len(view.own_hand) / 8.0, 1.0 if view.own_hidden is not None else 0.0,
                       len(view.antechambers[obs]) / 3.0],
                      [view.opp_hand_count / 8.0, 1.0 if view.opp_has_hidden else 0.0,
                       len(view.antechambers[opp]) / 3.0]], np.float32)
    kings = np.array([[t2.kings[0, 2], t2.kings[0, 3]],
                      [t2.kings[1, 2], 0.0]], np.float32)

    decision = np.zeros(DECISION_DIM, np.float32)
    src_ptr = -1
    if step is not None:
        decision[list(StepKind).index(step.kind)] = 1.0
        o = _N_STEP
        decision[o] = 1.0 if view.to_play != view.turn_player else 0.0       # I am the reactor
        o += 1
        if step.guess is not None:
            decision[o + _NAME_IX[step.guess]] = 1.0
        o += len(_NAMES)
        if step.number is not None and 1 <= step.number <= _N_NUM:
            decision[o + step.number - 1] = 1.0
        o += _N_NUM
        decision[o] = step.limit / 3.0                                       # Soldier picks left
        decision[o + 1] = len(step.chosen) / 3.0                             # picks made
        if step.source is not None:
            src_ptr = slot_of.get(step.source, -1)
    return State3(cards, depth, hands, kings, decision, src_ptr, slot_of)


def _instance_of(view: InformationSet, st: State3, name: str) -> int:
    """Slot of the decider's own (visible) copy of ``name`` -- the card a payload-free reveal uses."""
    for cid in view.own_hand:
        if card_name(cid) == name:
            return st.slot_of[cid]
    return -1


def encode_candidates(view: InformationSet, st: State3, moves: Sequence[Action]) -> Cands3:
    """One pointer record per move (see module docstring)."""
    n = len(moves)
    kind = np.zeros(n, np.int8)
    pa = np.full(n, -1, np.int8)
    pb = np.full(n, -1, np.int8)
    name = np.full(n, -1, np.int8)
    number = np.full(n, -1, np.int8)
    for i, m in enumerate(moves):
        kind[i] = m.kind.value - 1
        if m.card is not None:
            pa[i] = st.slot_of[m.card]
        elif m.target is not None:
            pa[i] = st.slot_of[view.stack[m.target].card]
        elif m.kind == ActionKind.FLIP_KING:
            pa[i] = KING_MINE
        elif m.kind in _REVEAL_CARD:
            pa[i] = _instance_of(view, st, _REVEAL_CARD[m.kind])
        elif m.kind in _STEP_ANSWERS:
            pa[i] = DECISION
            pb[i] = st.src_ptr
        if m.name is not None:
            name[i] = _NAME_IX[m.name]
        if m.number is not None:
            number[i] = m.number
    return Cands3(kind, pa, pb, name, number)


def tokenize3(view: InformationSet, moves: Optional[Sequence[Action]] = None):
    """``(State3, Cands3)`` for the decider's legal moves (or the given ``moves``)."""
    if moves is None:
        moves = view.legal_moves()
    st = tokenize_state3(view, legal_moves=list(moves))
    return st, encode_candidates(view, st, moves)


def card_token_fields() -> List[str]:
    """One label per dim of the v3 card token."""
    v2 = F2.card_token_fields()
    return [v2[i] for i in _V2_KEEP] + ["st:picked_target"] + v2[F2._ZONE_OFF:]


def decision_fields() -> List[str]:
    return ([f"step:{k.name}" for k in StepKind] + ["i_am_reactor"] + [f"guess:{n}" for n in _NAMES]
            + [f"number:{v}" for v in range(1, _N_NUM + 1)] + ["picks_left", "picks_made"])
