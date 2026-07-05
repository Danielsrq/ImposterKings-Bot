"""The shared featurizer: dimension consistency, determinism, and correct zone/action encoding (no torch)."""
from __future__ import annotations

import numpy as np

from imposterkings import scenario as sb
from imposterkings.actions import ActionKind
from imposterkings.machine_learning import features as F


def _view_and_moves():
    st = sb.build(hand0=["Judge", "Queen", "Fool"], hand1=["Warlord", "Elder"],
                  stack=["Zealot"], turn_player=0)
    return st.information_set(0), st.legal_moves()


def test_feature_dim_matches_encode_and_names():
    view, moves = _view_and_moves()
    x = F.encode(view, moves[0])
    assert x.shape == (F.FEATURE_DIM,) == (216,)
    assert len(F.feature_names()) == F.FEATURE_DIM
    assert x.dtype == np.float32 and np.isfinite(x).all()


def test_encode_is_deterministic():
    view, moves = _view_and_moves()
    assert np.array_equal(F.encode(view, moves[0]), F.encode(view, moves[0]))


def test_own_hand_and_action_slots():
    view, moves = _view_and_moves()
    names = F.feature_names()
    x = F.encode(view, moves[0])
    for card in ("Judge", "Queen", "Fool"):                 # own hand encoded as counts
        assert x[names.index(f"own_hand:{card}")] == 1.0
    assert x[names.index("own_hand:Warlord")] == 0.0        # opponent's card, not in own hand
    # the action's kind slot is set, and exactly one action-kind slot is hot
    kind_slots = [x[names.index(f"act_kind:{k.name}")] for k in ActionKind]
    assert sum(kind_slots) == 1.0 and x[names.index(f"act_kind:{moves[0].kind.name}")] == 1.0


def test_leading_effective_value_encoded():
    # Zealot(3) leading -> lead_val:3 hot; muting value 3 would make it play as 3 anyway (still 3).
    view, _ = _view_and_moves()
    names = F.feature_names()
    x = F.encode(view, view.legal_moves()[0])
    assert x[names.index("lead_val:3")] == 1.0
