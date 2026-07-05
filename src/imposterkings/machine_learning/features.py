"""The shared featurizer: (InformationSet, candidate Action) -> a fixed float32 vector (numpy only).

Info-set based -- the model sees only what the acting seat can observe (own hand, public zones, and the
name-set knowledge that guesses reveal), NEVER the omniscient state. The layout is a "bag of located
cards" (per-card-type counts by zone) + game-global scalars + the candidate action, so the same numbers
can later feed the attention model's token adapter (DATASET.md §5). Values are already ~[0,1], so no
global standardization is needed.

Layout (FEATURE_DIM = 216):
  state  (112): per-type counts (14 card types) over 8 zones
  global ( 53): leading value, phase, kings, counts, muted bitmask, reaction flag, leftover
  action ( 51): kind, played-card type, guess name, number, target
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..actions import Action, ActionKind, StepKind
from ..cards import CARD_NAMES, card_name, card_value
from ..infoset import InformationSet

_NAMES: List[str] = list(CARD_NAMES)           # 14 distinct card names (value-desc order)
_NTYPE = len(_NAMES)                            # 14
_TYPE_IX = {n: i for i, n in enumerate(_NAMES)}
_N_KIND = len(ActionKind)                       # 14
_N_STEP = len(StepKind)                         # 15

_STATE_ZONES = ["own_hand", "opp_known_has", "opp_known_lacks", "own_hidden",
                "stack", "discard", "ante_own", "ante_opp"]


def _type_ix(card_id: int) -> int:
    return _TYPE_IX[card_name(card_id)]


def _counts(card_ids) -> np.ndarray:
    v = np.zeros(_NTYPE, np.float32)
    for c in card_ids:
        v[_type_ix(c)] += 1.0
    return v


def _type_onehot(card_id: Optional[int]) -> np.ndarray:
    v = np.zeros(_NTYPE, np.float32)
    if card_id is not None:
        v[_type_ix(card_id)] = 1.0
    return v


def _name_multihot(names) -> np.ndarray:
    v = np.zeros(_NTYPE, np.float32)
    for nm in names:
        v[_TYPE_IX[nm]] = 1.0
    return v


def _eff_leading_value(view: InformationSet) -> Optional[int]:
    """Effective value of the leading (top) stack card, mirroring state.effective_stack_value."""
    if not view.stack:
        return None
    sc = view.stack[-1]
    if sc.disgraced:
        return 0
    if card_value(sc.card) in view.muted_values:
        return 3
    if sc.value_override is not None:
        return sc.value_override
    return card_value(sc.card)


def _state_block(view: InformationSet) -> np.ndarray:
    obs = view.observer
    return np.concatenate([
        _counts(view.own_hand),
        _name_multihot(view.opp_hand_has),            # names known to be IN opp hand (correct guesses)
        _name_multihot(view.opp_hand_lacks),          # names known NOT in opp hand
        _type_onehot(view.own_hidden),
        _counts([s.card for s in view.stack]),
        _counts(view.discard),
        _counts(view.antechambers[obs]),
        _counts(view.antechambers[1 - obs]),
    ])                                                # 8 * 14 = 112


def _global_block(view: InformationSet) -> np.ndarray:
    obs = view.observer
    lead = np.zeros(10, np.float32)                   # leading effective value one-hot 0..9
    ev = _eff_leading_value(view)
    if ev is not None and 0 <= ev <= 9:
        lead[ev] = 1.0
    phase = np.zeros(_N_STEP, np.float32)             # current decision phase
    if view.pending:
        phase[view.pending[-1].kind.value - 1] = 1.0
    muted = np.zeros(8, np.float32)                   # muted base values 1..8
    for mv in view.muted_values:
        if 1 <= mv <= 8:
            muted[mv - 1] = 1.0
    scalars = np.array([                              # order matches feature_names()
        float(view.kings[obs]), float(view.kings[1 - obs]),
        view.opp_hand_count / 7.0, float(view.opp_has_hidden), len(view.own_hand) / 8.0,
        float(view.to_play != view.turn_player),      # is this a reaction window?
    ], np.float32)
    return np.concatenate([lead, phase, scalars, muted,
                           _type_onehot(view.leftover_faceup)])   # 10+15+6+8+14 = 53


def _action_block(action: Action) -> np.ndarray:
    kind = np.zeros(_N_KIND, np.float32)
    kind[action.kind.value - 1] = 1.0
    guess = np.zeros(_NTYPE, np.float32)
    if action.name is not None:
        guess[_TYPE_IX[action.name]] = 1.0
    number = np.zeros(8, np.float32)
    if action.number is not None and 1 <= action.number <= 8:
        number[action.number - 1] = 1.0
    target = np.array([action.target / 8.0 if action.target is not None else 0.0], np.float32)
    return np.concatenate([kind, _type_onehot(action.card), guess, number, target])   # 51


def encode(view: InformationSet, action: Action) -> np.ndarray:
    """Encode a (position, candidate-action) pair into the model input vector."""
    return np.concatenate([_state_block(view), _global_block(view), _action_block(action)])


_STATE_DIM = len(_STATE_ZONES) * _NTYPE                 # 8 * 14  = 112
_GLOBAL_DIM = 10 + _N_STEP + 6 + 8 + _NTYPE             # 10+15+6+8+14 = 53
_ACTION_DIM = _N_KIND + _NTYPE + _NTYPE + 8 + 1         # kind+card+guess+number+target = 51
FEATURE_DIM = _STATE_DIM + _GLOBAL_DIM + _ACTION_DIM    # = 216


def feature_names() -> List[str]:
    """Human-readable name per feature index (for interpretability / debugging)."""
    names: List[str] = []
    for z in _STATE_ZONES:
        names += [f"{z}:{n}" for n in _NAMES]
    names += [f"lead_val:{v}" for v in range(10)]
    names += [f"phase:{s.name}" for s in StepKind]
    names += ["king_own", "king_opp", "opp_hand_count", "opp_has_hidden", "own_hand_size", "is_reaction"]
    names += [f"muted:{v}" for v in range(1, 9)]
    names += [f"leftover:{n}" for n in _NAMES]
    names += [f"act_kind:{k.name}" for k in ActionKind]
    names += [f"act_card:{n}" for n in _NAMES]
    names += [f"guess:{n}" for n in _NAMES]
    names += [f"number:{v}" for v in range(1, 9)]
    names += ["target"]
    return names
