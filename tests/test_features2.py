"""Featurization v2.2 (features2): fixed-18 instance tokens, zone posteriors, mute-as-deletion,
pending flags, canonical duplicates. Headless, engine-only (no torch/pygame)."""
import numpy as np

from imposterkings.actions import Action, ActionKind, DECLARE, DECLINE_REACTION, StepKind
from imposterkings.cards import card_name, card_value
from imposterkings.infoset import InformationSet
from imposterkings.machine_learning import features2 as F2
from imposterkings.state import GameState

from .helpers import mainstate, run, cid, sc


def _view(seed=0, seat=None):
    s = GameState.deal(np.random.default_rng(seed))
    return InformationSet.from_state(s, s.to_play if seat is None else seat)


def _tok(view):
    return F2.tokenize_state(view)


def _zone(v, z):
    return float(v[F2._ZONE_OFF + F2._Z[z]])


def _slot(name, k=0):
    idx = [i for i, n in enumerate(F2.INSTANCE_NAMES) if n == name]
    return idx[k]


# --- shapes + posterior invariants ------------------------------------------------------------------

def test_fixed_shapes_and_posterior_sums():
    t = _tok(_view())
    assert t.cards.shape == (18, F2.CARD_DIM) and t.kings.shape == (2, 4)
    assert t.board.shape == (4,) and t.phase.shape == (15,) and t.action.shape == (51,)
    assert t.labels == F2.INSTANCE_LABELS
    sums = t.cards[:, F2._ZONE_OFF:].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5)                          # every posterior sums to 1


def test_seen_cards_are_deltas_unseen_are_spreads():
    view = _view()
    t = _tok(view)
    zone_block = t.cards[:, F2._ZONE_OFF:]
    for i, nm in enumerate(F2.INSTANCE_NAMES):
        mx = zone_block[i].max()
        assert mx <= 1.0 + 1e-6
        # a delta iff max == 1; my own hand cards must be deltas on my_hand
    my_names = {card_name(c) for c in view.own_hand}
    for nm in my_names:
        i = _slot(nm)                                                 # visible copies fill lower slots
        assert _zone(t.cards[i], "my_hand") == 1.0


def test_posteriors_are_determinization_invariant():
    rng = np.random.default_rng(5)
    for g in range(4):
        s = GameState.deal(np.random.default_rng(200 + g))
        steps = 0
        while not s.is_terminal() and steps < 25:
            view = InformationSet.from_state(s, s.to_play)
            a = F2.tokenize_state(view, legal_moves=s.legal_moves())
            b = F2.tokenize_state(view, legal_moves=view.legal_moves())   # fresh determinization inside
            c = F2.tokenize_state(view)                                   # another one
            assert np.array_equal(a.cards, b.cards) and np.array_equal(a.cards, c.cards)
            assert np.array_equal(a.kings, b.kings) and np.array_equal(a.board, c.board)
            legal = s.legal_moves()
            s = s.apply(legal[rng.integers(len(legal))])
            steps += 1


def test_hand_lacks_zeroes_their_hand_and_renormalizes():
    # A wrong guess teaches hand_lacks; the lacked name's unseen copy must carry 0 mass on their_hand.
    st = mainstate(hand0=(cid("Soldier"), cid("Fool")), hand1=(cid("Warlord"), cid("Elder")),
                   stack=(sc("Zealot"),))
    st = run(st, Action(ActionKind.PLAY_CARD, card=cid("Soldier")),
             Action(ActionKind.GUESS_CARD, name="Queen"))            # wrong: P1 lacks Queen
    view = InformationSet.from_state(st, 0)
    t = _tok(view)
    q = t.cards[_slot("Queen")]
    assert _zone(q, "their_hand") == 0.0
    assert abs(q[F2._ZONE_OFF:].sum() - 1.0) < 1e-5                  # renormalized over the rest


def test_mute_as_deletion_and_base_immutable():
    st = mainstate(hand0=(cid("Oathbound"), cid("Fool")), hand1=(cid("Soldier"),),
                   stack=(sc("Warlord"),), muted={6})
    t = _tok(InformationSet.from_state(st, 0))
    ob = t.cards[_slot("Oathbound")]
    assert ob[F2._MECH_OFF:F2._MECH_OFF + 8].sum() == 0.0            # mechanics DELETED
    assert ob[F2._STATE_OFF + 0] == 1.0                              # is_muted explicit
    assert abs(ob[14] - 3.0 / 9.0) < 1e-6                            # power -> 3/9
    assert abs(ob[15] - 6.0 / 9.0) < 1e-6                            # base immutable
    so = t.cards[_slot("Soldier")]                                    # unmuted control
    assert so[F2._MECH_OFF:F2._MECH_OFF + 8].sum() > 0


def test_canonical_duplicates_visible_fills_lower_slot():
    # One Soldier visible on the stack, one unseen: slot 0 = stack delta, slot 1 = spread.
    st = mainstate(hand0=(cid("Fool"),), hand1=(cid("Queen"),), stack=(sc("Soldier"),))
    t = _tok(InformationSet.from_state(st, 0))
    s0, s1 = t.cards[_slot("Soldier", 0)], t.cards[_slot("Soldier", 1)]
    assert _zone(s0, "stack") == 1.0 and s0[F2._STATE_OFF + 2] == 1.0     # visible, leading
    assert _zone(s1, "stack") == 0.0 and s1[F2._ZONE_OFF:].max() < 1.0   # unseen spread


def test_pending_flags_lifecycle_soldier_guess():
    st = mainstate(hand0=(cid("Soldier"), cid("Fool")), hand1=(cid("Warlord"),), stack=(sc("Zealot"),))
    t0 = _tok(InformationSet.from_state(st, 0))
    assert t0.cards[:, F2._STATE_OFF + 7].sum() == 0                 # no pending_source at MAIN
    st = run(st, Action(ActionKind.PLAY_CARD, card=cid("Soldier")))
    t1 = _tok(InformationSet.from_state(st, 0))                       # ABILITY_GUESS pending
    assert t1.cards[:, F2._STATE_OFF + 7].sum() == 1                 # exactly ONE pending_source
    assert t1.cards[_slot("Soldier", 0), F2._STATE_OFF + 7] == 1.0   # ...on the played Soldier
    st = run(st, Action(ActionKind.GUESS_CARD, name="Warlord"))       # landed -> KH window
    t2 = _tok(InformationSet.from_state(st, 1))                       # DEFENDER's view
    w = t2.cards[_slot("Warlord")]
    assert w[F2._STATE_OFF + 8] == 1.0                                # pending_guess_target on Warlord
    assert _zone(w, "my_hand") == 1.0                                 # (their view: it's in MY hand)
    st = run(st, DECLINE_REACTION)
    st = run(st, Action(ActionKind.STOP))                             # decline the disgrace select
    t3 = _tok(InformationSet.from_state(st, 0))
    assert t3.cards[:, F2._STATE_OFF + 7:F2._STATE_OFF + 10].sum() == 0   # chain over -> all clear


def test_pending_mute_target_marks_every_base_value_instance():
    st = mainstate(hand0=(cid("Mystic"), cid("Fool")), hand1=(cid("KingsHand"),), stack=(sc("Elder"),))
    st = run(st, Action(ActionKind.PLAY_CARD, card=cid("Mystic")),
             Action(ActionKind.CHOOSE_NUMBER, number=7))              # declare mute 7 -> KH window
    t = _tok(InformationSet.from_state(st, 1))
    marked = {F2.INSTANCE_NAMES[i] for i in range(18)
              if t.cards[i, F2._STATE_OFF + 9] == 1.0}
    assert marked == {"Warlord", "Mystic"}                            # every base-7 instance


def test_kings_and_board():
    st = mainstate(hand0=(cid("Fool"),), hand1=(cid("Queen"),), stack=(sc("Elder"),),
                   kings=(True, False))
    t = _tok(InformationSet.from_state(st, 0))
    assert t.kings[0].tolist()[:3] == [1.0, 0.0, 1.0]                # mine: owner=mine, flipped
    assert t.kings[1].tolist()[:3] == [0.0, 1.0, 0.0]                # theirs: unflipped
    assert t.board[2] == 1.0                                          # my turn


def test_with_action_stamps_candidates_copy_on_write():
    view = _view()
    st = F2.tokenize_state(view)
    base = st.cards.copy()
    play = next(a for a in view.legal_moves() if a.card is not None)
    t = F2.with_action(st, play)
    assert (t.cards[:, F2.CAND_COL] == 1.0).sum() == 1               # one owned copy flagged
    i = int(np.argmax(t.cards[:, F2.CAND_COL]))
    assert F2.INSTANCE_NAMES[i] == card_name(play.card)
    assert t.action.sum() > 0
    assert np.array_equal(st.cards, base)                             # base untouched
    # a guess flags EVERY instance of the name (no transient token in the fixed set)
    g = F2.with_action(st, Action(ActionKind.GUESS_CARD, name="Soldier"))
    assert (g.cards[:, F2.CAND_COL] == 1.0).sum() == 2               # both Soldiers


# --- model + dataset plumbing (torch; skipped cleanly without it) ------------------------------------

def test_model_forward_save_load_v2(tmp_path):
    import pytest
    torch = pytest.importorskip("torch")
    from imposterkings.machine_learning.attention_model import (
        AttentionModel, AttnConfig, collate2, evaluator_from_model, load, save)

    torch.manual_seed(0)
    m = AttentionModel(AttnConfig(d_model=32, feat="v2")).eval()
    view = _view()
    toks = [F2.tokenize(view, a) for a in view.legal_moves()[:3]]
    b = collate2(toks)
    with torch.no_grad():
        q, attn = m(b["cards"], b["board"], b["phase"], b["action"], kings=b["kings"])
    assert q.shape == (len(toks),) and attn.shape[-1] == 24          # S = 1+18+2+3
    p = str(tmp_path / "m2.pt")
    save(p, m)
    m2, _ = load(p)
    assert m2.cfg.feat == "v2"
    with torch.no_grad():
        q2, _ = m2(b["cards"], b["board"], b["phase"], b["action"], kings=b["kings"])
    assert torch.allclose(q, q2)
    # the evaluator dispatch drives a real state end-to-end
    s = GameState.deal(np.random.default_rng(1))
    value, priors = evaluator_from_model(m2)(s)
    assert abs(sum(priors.values()) - 1.0) < 1e-5 and value[0] == -value[1]


def test_dataset_v2_build_load_roundtrip(tmp_path):
    import glob
    import os
    import pytest
    pytest.importorskip("torch")
    from imposterkings.machine_learning import token_dataset as TD

    DATA = os.path.join("datasets", "selfplay_k20l3")
    if not glob.glob(os.path.join(DATA, "*.jsonl")):
        pytest.skip("self-play corpus not present")
    out = str(tmp_path / "v2.npz")
    stats = TD.build(DATA, out, limit=4, feat="v2")
    rows = TD.load(out)
    assert rows.feat == "v2" and len(rows) == stats["n_rows"] and len(rows) > 0
    c, k, b, p, a = rows.tokens(0)
    assert c.shape == (18, F2.CARD_DIM) and k.shape == (2, 4)
    assert np.allclose(c[:, F2._ZONE_OFF:].sum(axis=1), 1.0, atol=1e-4)   # posteriors survive the npz
