"""Belief net (Study 7): the true-zone label builder, the belief head + masked cross-entropy, and the
sanity that the head can memorise a position (seen cards -> their known zone).

The label builder (`features2.true_zones`) is engine-only (no torch); the head/loss/checkpoint tests
`importorskip('torch')` so the suite still runs on a torch-free checkout.
"""
from __future__ import annotations

import numpy as np
import pytest

from imposterkings.cards import DECK_SIZE, card_ids_for_name, card_name
from imposterkings.infoset import InformationSet
from imposterkings.machine_learning import features2 as F2
from imposterkings.machine_learning.features2 import _omniscient_zones, true_zones
from imposterkings.state import GameState

from .helpers import mainstate


def _walk(seed, plies):
    """Deal and advance ``plies`` legal moves (deterministic), yielding (state, observer) at each step."""
    s = GameState.deal(np.random.default_rng(seed))
    rng = np.random.default_rng(1000 + seed)
    for _ in range(plies):
        if s.is_terminal():
            break
        yield s, s.to_play
        legal = s.legal_moves()
        s = s.apply(legal[rng.integers(len(legal))])


# --- label builder: engine-only ---------------------------------------------------------------------

def test_true_zones_multiset_matches_the_omniscient_truth():
    """The (name, zone) multiset the labels encode must equal the omniscient state's -- across many
    positions and plies, which also exercises duplicate copies landing in different zones."""
    for seed in range(12):
        for s, obs in _walk(seed, plies=6):
            view = InformationSet.from_state(s, obs)
            zl, sm = true_zones(view, s)
            got = sorted((F2.INSTANCE_NAMES[i], F2.ZONES[int(zl[i])]) for i in range(F2.N_CARDS))
            zmap = _omniscient_zones(s, obs)
            want = sorted((card_name(c), zmap[c]) for c in range(DECK_SIZE))
            assert got == want


def test_seen_mask_at_deal_is_nine_and_nine():
    """At the opening decision the observer knows its 8 dealt cards + the face-up leftover (9 seen); the
    other 9 (opponent's 8 + the face-down leftover) are the belief target, all in hidden zones."""
    s = GameState.deal(np.random.default_rng(7))
    view = InformationSet.from_state(s, s.to_play)
    zl, sm = true_zones(view, s)
    assert sm.sum() == 9 and (~sm).sum() == 9
    hidden_ix = {F2._Z[z] for z in F2._HIDDEN_ZONES}
    assert all(int(zl[i]) in hidden_ix for i in range(F2.N_CARDS) if not sm[i])   # unseen => hidden zone


def test_seen_slots_carry_their_visible_zone():
    """A seen slot's label is its real (observed) zone, and own-hand cards are marked seen on my_hand."""
    s = GameState.deal(np.random.default_rng(2))
    obs = s.to_play
    view = InformationSet.from_state(s, obs)
    zl, sm = true_zones(view, s)
    my_hand_names = [card_name(c) for c in view.own_hand]
    for c in view.own_hand:
        slot = next(i for i, nm in enumerate(F2.INSTANCE_NAMES)
                    if nm == card_name(c) and sm[i] and F2.ZONES[int(zl[i])] == "my_hand")
        assert slot is not None
    assert sm[[i for i, nm in enumerate(F2.INSTANCE_NAMES) if nm in my_hand_names and sm[i]]].all()


def test_duplicate_split_one_seen_one_unseen():
    """Find a real deal where a 2-copy name has one copy in the observer's hand and one hidden from it;
    the seen copy fills a lower slot with its true zone, the unseen copy carries the omniscient zone."""
    dup = next(n for n in F2._NAMES if len(card_ids_for_name(n)) == 2)      # e.g. Elder / Oathbound
    for seed in range(200):
        s = GameState.deal(np.random.default_rng(seed))
        obs = s.to_play
        ids = card_ids_for_name(dup)
        seen_here = [c for c in ids if c in s.hands[obs]]
        unseen_here = [c for c in ids if c in s.hands[1 - obs]]
        if len(seen_here) == 1 and len(unseen_here) == 1:
            view = InformationSet.from_state(s, obs)
            zl, sm = true_zones(view, s)
            slots = [i for i, nm in enumerate(F2.INSTANCE_NAMES) if nm == dup]
            zones = {(bool(sm[i]), F2.ZONES[int(zl[i])]) for i in slots}
            assert (True, "my_hand") in zones and (False, "their_hand") in zones
            return
    pytest.skip("no deal in range split a duplicate across the two hands")


# --- belief head + masked cross-entropy: torch ------------------------------------------------------

def test_ce_per_card_matches_manual():
    torch = pytest.importorskip("torch")
    from imposterkings.machine_learning.train_belief import _ce_per_card
    logits = torch.randn(4, 18, 12)
    zl = torch.randint(0, 12, (4, 18))
    ce = _ce_per_card(logits, zl)
    logp = torch.log_softmax(logits, dim=-1)
    manual = -logp.gather(-1, zl.unsqueeze(-1)).squeeze(-1)
    assert ce.shape == (4, 18) and torch.allclose(ce, manual)


def test_belief_head_shapes():
    torch = pytest.importorskip("torch")
    from imposterkings.machine_learning.attention_model import AttentionModel, AttnConfig
    m = AttentionModel(AttnConfig(d_model=32, n_layers=2, n_heads=4, ffn_hidden=64, feat="v2", head="belief"))
    logits, attn = m.forward_belief(torch.randn(3, 18, 46), torch.randn(3, 4), torch.randn(3, 15),
                                    torch.zeros(3, 51), kings=torch.randn(3, 2, 4))
    assert logits.shape == (3, 18, 12) and attn.shape[0] == 3


def test_belief_head_requires_v2():
    pytest.importorskip("torch")
    from imposterkings.machine_learning.attention_model import AttnConfig
    with pytest.raises(ValueError):
        AttnConfig(feat="v1", head="belief")


def _one_position_tensors(seed=3):
    torch = pytest.importorskip("torch")
    s = GameState.deal(np.random.default_rng(seed))
    for _ in range(4):                                     # advance to a mid-game state (mix of zones)
        s = s.apply(s.legal_moves()[0])
    view = InformationSet.from_state(s, s.to_play)
    t = F2.tokenize_state(view)
    zl, sm = true_zones(view, s)
    T = lambda a: torch.tensor(a).unsqueeze(0)
    return (T(t.cards), T(t.board), T(t.phase), T(t.action), T(t.kings),
            torch.tensor(zl).unsqueeze(0), torch.tensor(sm).unsqueeze(0))


def test_overfit_single_position_recovers_all_zones():
    """Sanity: on one position the head can memorise every card's zone -- including the SEEN cards
    (the check the 12-way head exists to enable) -- and the unseen-CE falls."""
    torch = pytest.importorskip("torch")
    from imposterkings.machine_learning.attention_model import AttentionModel, AttnConfig
    from imposterkings.machine_learning.train_belief import _ce_per_card
    cards, board, phase, action, kings, zl, sm = _one_position_tensors()
    torch.manual_seed(0)
    m = AttentionModel(AttnConfig(d_model=32, n_layers=2, n_heads=4, ffn_hidden=64, feat="v2", head="belief"))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    first = None
    for i in range(200):
        logits, _ = m.forward_belief(cards, board, phase, action, kings=kings)
        loss = _ce_per_card(logits, zl).mean()
        if i == 0:
            first = float(loss.detach())
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        logits, _ = m.forward_belief(cards, board, phase, action, kings=kings)
    assert float(_ce_per_card(logits, zl).mean()) < first                 # loss dropped
    assert (logits.argmax(-1) == zl).all()                                # memorised all 18 slots
    assert (logits.argmax(-1)[sm] == zl[sm]).all()                        # seen-card sanity specifically


def test_checkpoint_roundtrip_preserves_belief_logits():
    torch = pytest.importorskip("torch")
    from imposterkings.machine_learning.attention_model import AttentionModel, AttnConfig, load, save
    cfg = AttnConfig(d_model=32, n_layers=2, n_heads=4, ffn_hidden=64, feat="v2", head="belief")
    m = AttentionModel(cfg)
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "belief.pt")
        save(path, m, meta={"target": "belief"})
        m2, meta = load(path)
        assert m2.cfg.head == "belief" and meta["target"] == "belief"
        x = (torch.randn(2, 18, 46), torch.randn(2, 4), torch.randn(2, 15), torch.zeros(2, 51))
        kings = torch.randn(2, 2, 4)
        m.eval(); m2.eval()
        with torch.no_grad():
            a, _ = m.forward_belief(*x, kings=kings)
            b, _ = m2.forward_belief(*x, kings=kings)
        assert torch.allclose(a, b)
