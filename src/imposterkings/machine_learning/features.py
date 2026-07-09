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

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..abilities import _MANDATORY_GUESS, _OPTIONAL_ONPLAY
from ..actions import Action, ActionKind, StepKind
from ..budget import _HEAVY_ABILITIES
from ..cards import (CARD_NAMES, Ability, card_ability, card_ids_for_name, card_name,
                     card_value, is_reaction, is_royalty)
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


# ============================================================================================
# Token adapter for the attention model (DATASET.md §5). "One featurizer, two adapters": the MLP
# `encode` above flattens everything to a fixed vector; `tokenize` below emits a variable-length set
# of small, single-purpose tokens -- card / board / phase / action -- plus per-card labels for the
# CLS->card importance readout. The learned CLS token is added model-side (no input features here).
# Per-type native widths feed per-type input projections (W_card/W_board/W_phase/W_action -> d_model).
# ============================================================================================

# Ability-category sets, built from the engine's existing groupings (identity fully determines these).
_GUESS_ABILITIES = _MANDATORY_GUESS | {Ability.INQUISITOR}          # Soldier, Judge, Inquisitor
_STACK_TARGET_ABILITIES = frozenset({Ability.SOLDIER, Ability.SENTRY, Ability.FOOL})
_MUTE_ABILITIES = frozenset({Ability.MYSTIC})
_FOLLOWUP_ABILITIES = frozenset({Ability.OATHBOUND})

_ZONES = ["my_hand", "opp_known", "my_hidden", "my_setup_discard", "leading", "stack_below",
          "ante_mine", "ante_opp", "discard", "leftover", "opp_unknown"]
_NZONE = len(_ZONES)                                                # 11
_Z = {z: i for i, z in enumerate(_ZONES)}

# Card-token block offsets (identity | ability_cat | value | zone | state | belief).
_CAT_OFF = _NTYPE                                                   # 14
_VAL_OFF = _CAT_OFF + 8                                             # 22  (base, effective)
_ZONE_OFF = _VAL_OFF + 2                                            # 24
_STATE_OFF = _ZONE_OFF + _NZONE                                    # 34
_BELIEF_OFF = _STATE_OFF + 7                                        # 41
CARD_DIM = _BELIEF_OFF + 2                                          # 43
_CAND_IX = _STATE_OFF + 4                                           # is_candidate_action -> 38
_PSEL_IX = _STATE_OFF + 6                                           # pending_selected     -> 40

BOARD_DIM = 2 + 2 + 2 + 8                                           # kings, sizes, turn, muted = 14
PHASE_DIM = _N_STEP + _NTYPE + _NTYPE + 8 + 1 + 1                   # 15+14+14+8+1+1 = 53
ACTION_DIM = _N_KIND + 8 + 1                                        # kind + number + target = 23

# Per-type tables (a card name fully determines its ability/value), precomputed once.
_TYPE_ID0 = [card_ids_for_name(n)[0] for n in _NAMES]               # a representative id per type


def _ability_cat(card_id: int) -> np.ndarray:
    ab = card_ability(card_id)
    return np.array([is_royalty(card_id), is_reaction(card_id),
                     ab in _OPTIONAL_ONPLAY, ab in _GUESS_ABILITIES, ab in _HEAVY_ABILITIES,
                     ab in _STACK_TARGET_ABILITIES, ab in _MUTE_ABILITIES,
                     ab in _FOLLOWUP_ABILITIES], np.float32)


_TYPE_BASE = np.array([card_value(i) for i in _TYPE_ID0], np.float32)          # [14]
_ABILITY_CAT = np.stack([_ability_cat(i) for i in _TYPE_ID0])                  # [14, 8]
_KING_RELATED = np.array([float(is_royalty(i) or is_reaction(i)) for i in _TYPE_ID0], np.float32)


@dataclass
class Tokens:
    """Attention-model input: a variable-length set of card tokens + singleton board/phase/action
    tokens + per-card labels (for the CLS->card importance readout). The learned CLS is model-side."""
    cards: np.ndarray          # [N, CARD_DIM]
    board: np.ndarray          # [BOARD_DIM]
    phase: np.ndarray          # [PHASE_DIM]
    action: np.ndarray         # [ACTION_DIM]  (all-zero when no candidate action is marked)
    labels: List[str]          # len N


def _type_counts(card_ids) -> dict:
    d: dict = {}
    for c in card_ids:
        t = _type_ix(c)
        d[t] = d.get(t, 0) + 1
    return d


def _eff_by_type(tix: int, muted) -> float:
    """Effective value of a hand/ante/discard card of this type (muted base value reads as 3)."""
    base = int(_TYPE_BASE[tix])
    return 3.0 if base in muted else float(base)


def _eff_stack(sc, muted) -> float:
    """Effective value of a stack card, mirroring state.effective_stack_value."""
    if sc.disgraced:
        return 0.0
    if card_value(sc.card) in muted:
        return 3.0
    if sc.value_override is not None:
        return float(sc.value_override)
    return float(card_value(sc.card))


def _card_token(tix: Optional[int], zone: str, eff: float, *, disgraced=0.0, is_muted=0.0,
                is_leading=0.0, is_legal=0.0, is_candidate=0.0, pending_selected=0.0,
                identity_known=1.0, count=1.0) -> np.ndarray:
    v = np.zeros(CARD_DIM, np.float32)
    king = 0.0
    if tix is not None:
        v[tix] = 1.0                                               # identity one-hot
        v[_CAT_OFF:_CAT_OFF + 8] = _ABILITY_CAT[tix]               # ability category multihot
        v[_VAL_OFF] = _TYPE_BASE[tix] / 9.0                        # base value
        king = _KING_RELATED[tix]
    v[_VAL_OFF + 1] = eff / 9.0                                    # effective value
    v[_ZONE_OFF + _Z[zone]] = 1.0                                  # zone one-hot
    v[_STATE_OFF:_STATE_OFF + 7] = (disgraced, is_muted, is_leading, is_legal, is_candidate,
                                    king, pending_selected)
    v[_BELIEF_OFF] = identity_known
    v[_BELIEF_OFF + 1] = min(float(count), 7.0) / 7.0
    return v


def _board_token(view: InformationSet) -> np.ndarray:
    obs = view.observer
    v = np.zeros(BOARD_DIM, np.float32)
    v[0], v[1] = float(view.kings[obs]), float(view.kings[1 - obs])
    v[2], v[3] = len(view.own_hand) / 8.0, view.opp_hand_count / 7.0
    v[4], v[5] = float(view.to_play == obs), float(view.to_play != view.turn_player)
    for mv in view.muted_values:
        if 1 <= mv <= 8:
            v[6 + mv - 1] = 1.0
    return v


def _phase_token(view: InformationSet) -> np.ndarray:
    v = np.zeros(PHASE_DIM, np.float32)
    if not view.pending:
        return v
    step = view.pending[-1]
    v[step.kind.value - 1] = 1.0                                   # phase one-hot           [0:15]
    o = _N_STEP
    if step.source is not None and step.source >= 0:
        v[o + _type_ix(step.source)] = 1.0                        # resolving source type   [15:29]
    o += _NTYPE
    if step.guess is not None:
        v[o + _TYPE_IX[step.guess]] = 1.0                        # carried guess name      [29:43]
    o += _NTYPE
    if step.number is not None and 1 <= step.number <= 8:
        v[o + step.number - 1] = 1.0                             # carried Mystic number   [43:51]
    o += 8
    v[o] = min(step.limit, 3) / 3.0                               # multi-select remaining  [51]
    v[o + 1] = min(len(step.chosen), 3) / 3.0                     # multi-select picks made [52]
    return v


def _action_vec(action: Optional[Action]) -> np.ndarray:
    v = np.zeros(ACTION_DIM, np.float32)
    if action is None:
        return v
    v[action.kind.value - 1] = 1.0                                # ActionKind one-hot  [0:14]
    if action.number is not None and 1 <= action.number <= 8:
        v[_N_KIND + action.number - 1] = 1.0                     # number              [14:22]
    v[_N_KIND + 8] = action.target / 8.0 if action.target is not None else 0.0   # target [22]
    return v


def tokenize(view: InformationSet, action: Optional[Action] = None) -> Tokens:
    """The attention-model adapter: the acting seat's InfoSet -> card tokens + singleton board / phase
    / action tokens + per-card labels. ``action`` (a candidate move) marks ``is_candidate_action`` and
    fills the action token for "action-in" scoring ``f(state, action) -> q``."""
    obs = view.observer
    muted = view.muted_values
    toks: List[np.ndarray] = []
    labels: List[str] = []

    def add(tix, zone, eff, **kw):
        toks.append(_card_token(tix, zone, eff, **kw))
        labels.append(f"{zone}:{_NAMES[tix] if tix is not None else '?'}")

    legal_types = set()
    if view.to_play == obs:                                       # legal_moves() needs the actor's seat
        for a in view.legal_moves():
            if a.kind == ActionKind.PLAY_CARD and a.card is not None:
                legal_types.add(_type_ix(a.card))

    # stack positions already committed to a pending disgrace (not yet flagged disgraced) -> by type
    selected_types = set()
    if view.pending and view.pending[-1].chosen:
        for pos in view.pending[-1].chosen:
            if 0 <= pos < len(view.stack):
                selected_types.add(_type_ix(view.stack[pos].card))

    def muted_of(tix):
        return 1.0 if int(_TYPE_BASE[tix]) in muted else 0.0

    for tix, cnt in _type_counts(view.own_hand).items():
        add(tix, "my_hand", _eff_by_type(tix, muted), is_muted=muted_of(tix),
            is_legal=1.0 if tix in legal_types else 0.0, count=cnt)
    if view.own_hidden is not None:
        t = _type_ix(view.own_hidden)
        add(t, "my_hidden", _eff_by_type(t, muted), is_muted=muted_of(t))
    if view.own_setup_discard is not None:                          # own setup-discard: known, out of play
        t = _type_ix(view.own_setup_discard)
        add(t, "my_setup_discard", _eff_by_type(t, muted), is_muted=muted_of(t))
    for nm in sorted(view.opp_hand_has):
        t = _TYPE_IX[nm]
        add(t, "opp_known", _eff_by_type(t, muted), is_muted=muted_of(t))
    if view.opp_hand_count > 0:
        add(None, "opp_unknown", 0.0, identity_known=0.0, count=view.opp_hand_count)
    if view.stack:
        top = view.stack[-1]
        tt = _type_ix(top.card)
        add(tt, "leading", _eff_stack(top, muted), disgraced=1.0 if top.disgraced else 0.0,
            is_muted=1.0 if card_value(top.card) in muted else 0.0, is_leading=1.0,
            pending_selected=1.0 if tt in selected_types else 0.0)
        below: dict = {}
        for sc in view.stack[:-1]:
            below.setdefault(_type_ix(sc.card), []).append(sc)
        for t, group in below.items():
            add(t, "stack_below", _eff_by_type(t, muted), is_muted=muted_of(t),
                disgraced=1.0 if any(s.disgraced for s in group) else 0.0,
                pending_selected=1.0 if t in selected_types else 0.0, count=len(group))
    for tix, cnt in _type_counts(view.antechambers[obs]).items():
        add(tix, "ante_mine", _eff_by_type(tix, muted), is_muted=muted_of(tix),
            is_legal=1.0 if tix in legal_types else 0.0, count=cnt)   # ASCEND: the queued card is the play
    for tix, cnt in _type_counts(view.antechambers[1 - obs]).items():
        add(tix, "ante_opp", _eff_by_type(tix, muted), is_muted=muted_of(tix), count=cnt)
    for tix, cnt in _type_counts(view.discard).items():
        add(tix, "discard", _eff_by_type(tix, muted), is_muted=muted_of(tix), count=cnt)
    if view.leftover_faceup is not None and view.leftover_faceup >= 0:
        t = _type_ix(view.leftover_faceup)
        add(t, "leftover", _eff_by_type(t, muted), is_muted=muted_of(t))

    if action is not None:
        _mark_candidate(toks, labels, action)

    cards_arr = np.stack(toks) if toks else np.zeros((0, CARD_DIM), np.float32)
    return Tokens(cards_arr, _board_token(view), _phase_token(view), _action_vec(action), labels)


def _mark_candidate(toks: List[np.ndarray], labels: List[str], action: Action) -> None:
    """Point ``is_candidate_action`` at the involved token; append a transient ``*`` choice token if
    none matches (a guess claims the opponent holds ``name``; an unmatched card play)."""
    def flag(label: str, tix: int, zone: str):
        for i, lb in enumerate(labels):
            if lb == label:
                toks[i][_CAND_IX] = 1.0
                return
        toks.append(_card_token(tix, zone, float(_TYPE_BASE[tix]), is_candidate=1.0))
        labels.append(label + "*")

    if action.card is not None:
        tix = _type_ix(action.card)
        flag(f"my_hand:{_NAMES[tix]}", tix, "my_hand")
    if action.name is not None:
        tix = _TYPE_IX[action.name]
        flag(f"opp_known:{_NAMES[tix]}", tix, "opp_known")


# --- interpretability: one label per dim of each token kind (mirrors feature_names()) --------------

def card_token_fields() -> List[str]:
    f = [f"id:{n}" for n in _NAMES]
    f += ["cat:royalty", "cat:reaction", "cat:optional", "cat:guess", "cat:heavy",
          "cat:stack_target", "cat:mute", "cat:followup"]
    f += ["val:base", "val:eff"]
    f += [f"zone:{z}" for z in _ZONES]
    f += ["st:disgraced", "st:muted", "st:leading", "st:legal_now", "st:candidate",
          "st:king_related", "st:pending_selected"]
    f += ["bel:identity_known", "bel:count"]
    return f


def board_fields() -> List[str]:
    return (["king_own", "king_opp", "own_hand_size", "opp_hand_size", "is_my_turn", "is_reaction"]
            + [f"muted:{v}" for v in range(1, 9)])


def phase_fields() -> List[str]:
    return ([f"phase:{s.name}" for s in StepKind] + [f"src:{n}" for n in _NAMES]
            + [f"guess:{n}" for n in _NAMES] + [f"number:{v}" for v in range(1, 9)]
            + ["limit", "chosen"])


def action_fields() -> List[str]:
    return [f"act:{k.name}" for k in ActionKind] + [f"number:{v}" for v in range(1, 9)] + ["target"]
