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
