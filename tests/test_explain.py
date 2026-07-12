"""Headless tests for machine_learning.explain (torch only, no display): payload shape, the row-sum
(axis-orientation) invariant, candidate-index resolution, the synthetic "*" guess-claim path, the sprite
map, and the non-breaking multi-layer forward. Uses a tiny UNTRAINED checkpoint -- plumbing, not quality."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from imposterkings.actions import Action, ActionKind
from imposterkings.cards import CARD_NAMES, card_name
from imposterkings.machine_learning.attention_model import AttentionModel, AttnConfig, collate
from imposterkings.machine_learning.explain import explain
from imposterkings.machine_learning.features import tokenize
from imposterkings.state import GameState


def _model(d=32, layers=1):
    torch.manual_seed(0)
    return AttentionModel(AttnConfig(d_model=d, n_layers=layers)).eval()


def _view(seed=0):
    s = GameState.deal(np.random.default_rng(seed))
    return s.information_set(s.to_play)


def test_payload_shape_and_axis_orientation():
    v = _view()
    p = explain(v, v.legal_moves()[0], _model())
    S = len(p.seq_labels)
    assert p.attn.shape == (p.n_heads, S, S)
    assert p.seq_labels[0] == "CLS"
    assert p.seq_labels[-3:] == ["board", "phase", "action"]
    # every query row is a softmax over keys -> sums to 1 (validates that the LAST axis is the key axis)
    assert np.allclose(p.attn.sum(axis=-1), 1.0, atol=1e-4)
    assert p.q == p.q and -1.0 <= p.q <= 1.0                        # finite + Tanh-bounded


def test_candidate_index_card_action():
    # Any action carrying a hand card (a play, or a setup hide/discard) flags that card's my_hand token.
    v = _view()
    card_act = next(a for a in v.legal_moves() if a.card is not None)
    p = explain(v, card_act, _model())
    ci = p.candidate_seq_index
    assert ci is not None and ci in p.candidate_seq_indices
    assert p.seq_labels[ci].startswith("my_hand:")                 # the flagged token is the hand card
    assert p.display_names[ci] == card_name(card_act.card)         # "*"-stripped name matches the card


def test_guess_claim_star_path():
    # A guess for a card the opponent is not KNOWN to hold appends a synthetic "opp_known:Name*" token.
    v = _view()
    guess = Action(ActionKind.GUESS_CARD, name="Soldier")
    p = explain(v, guess, _model())
    assert p.candidate_seq_index is not None
    lbl = p.seq_labels[p.candidate_seq_index]
    assert lbl.endswith("*") and lbl.startswith("opp_known:")      # the appended claim token
    assert p.display_names[p.candidate_seq_index] == "Soldier"     # "*" stripped for the sprite lookup


def test_sprite_map_covers_all_names():
    v = _view()
    p = explain(v, v.legal_moves()[0], _model())
    assert set(p.name_to_asset) == set(CARD_NAMES)
    assert all(isinstance(fn, str) and fn.endswith(".jpg") for fn in p.name_to_asset.values())
    # non-card seq positions carry no display name
    assert p.display_names[0] is None                              # CLS
    assert p.display_names[-3:] == [None, None, None]              # board / phase / action


def test_multi_layer_non_breaking():
    v = _view()
    a = v.legal_moves()[0]
    m2 = _model(layers=2)
    p = explain(v, a, m2, all_layers=True)
    assert p.per_layer is not None and len(p.per_layer) == 2
    assert np.allclose(p.per_layer[-1], p.attn, atol=1e-6)         # last layer == the readout attn

    # forward() still returns the (q, last-layer tensor) 2-tuple; forward_layers gives the list.
    b = collate([tokenize(v, a)])
    args = (b["cards"], b["board"], b["phase"], b["action"], b["card_mask"])
    q, attn = m2(*args)
    assert attn.dim() == 4 and attn.shape[0] == 1
    ql, attns = m2.forward_layers(*args)
    assert isinstance(attns, list) and len(attns) == 2
    assert torch.allclose(attns[-1], attn) and torch.allclose(ql, q)


def test_single_layer_no_per_layer_by_default():
    v = _view()
    p = explain(v, v.legal_moves()[0], _model(layers=1))
    assert p.per_layer is None and p.n_layers == 1
    assert p.row0_signed is None and p.attribution is None    # attribution off by default


def test_signed_attribution_shapes_and_reference():
    v = _view()
    a = v.legal_moves()[0]
    m = _model(layers=1)
    p = explain(v, a, m, attribution=True)
    S = len(p.seq_labels)
    assert p.row0_signed.shape == (p.n_heads, S) and p.attribution.shape == (S,)
    assert np.isfinite(p.row0_signed).all() and np.isfinite(p.attribution).all()
    assert np.allclose(p.attribution, p.row0_signed.sum(0), atol=1e-5)   # head-sum consistency

    # Reference: sum of signed contributions == head.weight @ (W_o @ out_concat_cls), i.e. the readout
    # layer's ATTENTION-PATH contribution to the q-logit, computed independently.
    b = collate([tokenize(v, a)])
    args = (b["cards"], b["board"], b["phase"], b["action"], b["card_mask"])
    with torch.no_grad():
        q, attns, values = m.forward_layers(*args, need_values=True)
        attn_last, v_last = attns[-1][0], values[-1][0]                  # [heads,S,S], [heads,S,dh]
        o = (attn_last[:, 0, :].unsqueeze(-1) * v_last).sum(1).reshape(-1)   # concat_h sum_j A[h,0,j] v[h,j]
        ref = float(m.head.weight[0] @ (m.layers[-1].attn.wo.weight @ o))
    assert abs(float(p.row0_signed.sum()) - ref) < 1e-4


# --- featurization v2: fixed S=24 axes, kings as tokens, zone posteriors -------------------------------

from imposterkings.machine_learning import features2 as F2      # noqa: E402


def _model2(d=32, layers=1):
    torch.manual_seed(0)
    return AttentionModel(AttnConfig(d_model=d, n_layers=layers, feat="v2")).eval()


def test_v2_payload_shape_and_fixed_axes():
    v = _view()
    p = explain(v, v.legal_moves()[0], _model2())
    assert p.feat == "v2" and len(p.seq_labels) == 24                # [CLS|18 cards|2 kings|brd|ph|act]
    assert p.attn.shape == (p.n_heads, 24, 24)
    assert p.seq_labels[0] == "CLS"
    assert p.seq_labels[1:19] == F2.INSTANCE_LABELS                  # the deck IS the axis, in fixed order
    assert p.seq_labels[19:21] == ["king:mine", "king:theirs"]
    assert p.seq_labels[-3:] == ["board", "phase", "action"]         # same contract as v1
    assert np.allclose(p.attn.sum(axis=-1), 1.0, atol=1e-4)          # rows are softmaxes over keys
    assert p.card_seq_range == (1, 19)


def test_v2_every_card_token_has_art_and_kings_do_not():
    v = _view()
    p = explain(v, v.legal_moves()[0], _model2())
    assert all(p.display_names[i] in CARD_NAMES for i in range(1, 19))   # all 18 render as cards
    assert p.display_names[1] == "Princess" and p.display_names[-4] is None   # kings carry no card name
    assert p.display_names[0] is None and p.display_names[-3:] == [None, None, None]


def test_v2_zone_posterior_rows_sum_to_one_and_seen_are_deltas():
    v = _view()
    p = explain(v, v.legal_moves()[0], _model2())
    assert p.zone_posterior.shape == (18, 12) and p.zone_names == F2.ZONES
    assert np.allclose(p.zone_posterior.sum(axis=1), 1.0, atol=1e-5)
    assert len(p.card_seen) == 18
    for i, seen in enumerate(p.card_seen):                            # seen <=> the posterior is a delta
        assert seen == bool(p.zone_posterior[i].max() >= 1.0 - 1e-6)
    assert any(p.card_seen) and not all(p.card_seen)                  # my hand is seen; the rest is belief
    # my own hand cards are located exactly, on my_hand
    for c in v.own_hand:
        i = F2.INSTANCE_NAMES.index(card_name(c))
        assert p.card_seen[i] and p.zone_posterior[i][F2._Z["my_hand"]] == 1.0


def test_v2_candidate_is_the_owned_copy():
    v = _view()
    act = next(a for a in v.legal_moves() if a.card is not None)
    p = explain(v, act, _model2())
    ci = p.candidate_seq_index
    assert ci is not None and ci in p.candidate_seq_indices
    assert p.display_names[ci] == card_name(act.card)
    assert p.card_seen[ci - 1]                                        # the played copy is one I can see
    assert p.zone_posterior[ci - 1][F2._Z["my_hand"]] == 1.0 or \
           p.zone_posterior[ci - 1][F2._Z["my_ante"]] == 1.0


def test_v2_guess_flags_every_instance_of_the_name():
    # No synthetic "*" token in v2: a claim lights up BOTH Soldiers; their posteriors say how plausible.
    v = _view()
    p = explain(v, Action(ActionKind.GUESS_CARD, name="Soldier"), _model2())
    flagged = {p.display_names[i] for i in p.candidate_seq_indices}
    assert flagged == {"Soldier"} and len(p.candidate_seq_indices) == 2
    assert [p.seq_labels[i] for i in p.candidate_seq_indices] == ["Soldier#0", "Soldier#1"]


def test_v2_attribution_and_layers():
    v = _view()
    a = v.legal_moves()[0]
    p = explain(v, a, _model2(layers=2), all_layers=True, attribution=True)
    assert p.row0_signed.shape == (p.n_heads, 24) and p.attribution.shape == (24,)
    assert np.allclose(p.attribution, p.row0_signed.sum(0), atol=1e-5)
    assert p.per_layer is not None and len(p.per_layer) == 2
    assert np.allclose(p.per_layer[-1], p.attn, atol=1e-6)
