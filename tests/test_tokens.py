"""Token adapter (features.tokenize): the attention model's card/board/phase/action tokens.

Covers the contract (shapes + field-name helpers), the located-card zones + opponent belief tokens,
action-in marking, and -- the load-bearing case -- that multi-step "subturn" plies (Soldier
guess vs disgrace-multi-select) produce distinct phase tokens, since that progress lives in the
PendingStep (`chosen`/`limit`), not on the board.
"""
import dataclasses

import numpy as np
import pytest

from imposterkings.actions import Action, ActionKind, StepKind
from imposterkings.cards import card_ids_for_name, card_name, card_value
from imposterkings.infoset import InformationSet
from imposterkings.machine_learning import features as F
from imposterkings.state import GameState, PendingStep, StackCard


def _view(seed: int = 0) -> InformationSet:
    s = GameState.deal(np.random.default_rng(seed))
    return InformationSet.from_state(s, s.to_play)


def _cf(tok, name):                                   # a card-token field value by label
    return tok[F.card_token_fields().index(name)]


def test_contract_shapes_and_field_lengths():
    t = F.tokenize(_view())
    assert F.CARD_DIM == 44 and t.cards.shape[1] == F.CARD_DIM
    assert t.board.shape == (F.BOARD_DIM,)
    assert t.phase.shape == (F.PHASE_DIM,)
    assert t.action.shape == (F.ACTION_DIM,)
    assert len(t.labels) == t.cards.shape[0]
    assert not t.action.any()                          # no candidate action -> all zero
    assert len(F.card_token_fields()) == F.CARD_DIM
    assert len(F.board_fields()) == F.BOARD_DIM
    assert len(F.phase_fields()) == F.PHASE_DIM
    assert len(F.action_fields()) == F.ACTION_DIM


def test_zones_leftover_and_opp_unknown():
    v = _view()
    t = F.tokenize(v)
    assert any(l.startswith("my_hand:") for l in t.labels)
    assert any(l.startswith("leftover:") for l in t.labels)
    unk = [i for i, l in enumerate(t.labels) if l == "opp_unknown:?"]
    assert len(unk) == 1
    tok = t.cards[unk[0]]
    assert tok[:F._NTYPE].sum() == 0.0                 # identity zeroed for the unknown mass
    assert _cf(tok, "bel:identity_known") == 0.0
    assert _cf(tok, "bel:count") == pytest.approx(min(v.opp_hand_count, 7) / 7)


def test_opp_known_from_landed_guess():
    v = dataclasses.replace(_view(), opp_hand_has=frozenset({"Queen"}))
    t = F.tokenize(v)
    idx = t.labels.index("opp_known:Queen")
    tok = t.cards[idx]
    assert tok[F._TYPE_IX["Queen"]] == 1.0             # identity filled
    assert _cf(tok, "bel:identity_known") == 1.0


def test_is_muted_and_effective_value():
    v0 = _view()
    val = card_value(v0.own_hand[0])
    v = dataclasses.replace(v0, muted_values=frozenset({val}))
    t = F.tokenize(v)
    idx = t.labels.index(f"my_hand:{card_name(v.own_hand[0])}")
    tok = t.cards[idx]
    assert _cf(tok, "st:muted") == 1.0
    assert _cf(tok, "val:eff") == pytest.approx(3 / 9)


def test_action_in_marks_played_card():
    v = _view()
    card = v.own_hand[0]
    t = F.tokenize(v, Action(kind=ActionKind.PLAY_CARD, card=card))
    idx = t.labels.index(f"my_hand:{card_name(card)}")
    assert _cf(t.cards[idx], "st:candidate") == 1.0
    assert t.action[F.action_fields().index("act:PLAY_CARD")] == 1.0
    assert t.action[F.action_fields().index(f"card:{card_name(card)}")] == 1.0   # explicit card one-hot


def test_action_in_guess_adds_choice_token():
    t = F.tokenize(_view(), Action(kind=ActionKind.GUESS_CARD, name="Queen"))
    assert "opp_known:Queen*" in t.labels             # transient choice token
    idx = t.labels.index("opp_known:Queen*")
    assert _cf(t.cards[idx], "st:candidate") == 1.0
    assert t.action[F.action_fields().index("act:GUESS_CARD")] == 1.0
    assert t.action[F.action_fields().index("guess:Queen")] == 1.0              # explicit guess one-hot


def test_setup_phase_and_setup_discard_token():
    # (a) the dealt state is a SETUP_HIDE ply -> phase + HIDE_CARD action are representable
    v0 = _view(3)
    assert v0.pending[-1].kind == StepKind.SETUP_HIDE
    t0 = F.tokenize(v0, Action(kind=ActionKind.HIDE_CARD, card=v0.own_hand[0]))
    pf = F.phase_fields()
    assert t0.phase[pf.index("phase:SETUP_HIDE")] == 1.0
    assert t0.action[F.action_fields().index("act:HIDE_CARD")] == 1.0
    # (b) play through setup to a MAIN ply -> the own setup-discard becomes a known token
    s = GameState.deal(np.random.default_rng(3))
    rng = np.random.default_rng(0)
    while True:
        v = InformationSet.from_state(s, s.to_play)
        if v.pending and v.pending[-1].kind == StepKind.MAIN:
            break
        lm = v.legal_moves(); s = s.apply(lm[int(rng.integers(len(lm)))])
    assert v.own_setup_discard is not None
    t = F.tokenize(v)
    idx = t.labels.index(f"my_setup_discard:{card_name(v.own_setup_discard)}")
    assert _cf(t.cards[idx], "bel:identity_known") == 1.0


def test_subturn_phase_tokens_distinguish_guess_vs_disgrace():
    base = _view()
    obs = base.observer
    sol = card_ids_for_name("Soldier")[0]
    pf = F.phase_fields()
    guess = dataclasses.replace(
        base, pending=(PendingStep(StepKind.ABILITY_GUESS, actor=obs, source=sol),))
    disg0 = dataclasses.replace(
        base, pending=(PendingStep(StepKind.ABILITY_STACK_TARGET, actor=obs, source=sol,
                                   limit=3, chosen=()),))
    disg1 = dataclasses.replace(
        base, pending=(PendingStep(StepKind.ABILITY_STACK_TARGET, actor=obs, source=sol,
                                   limit=2, chosen=(0,)),))
    pg, p0, p1 = F.tokenize(guess).phase, F.tokenize(disg0).phase, F.tokenize(disg1).phase
    assert pg[pf.index("phase:ABILITY_GUESS")] == 1.0
    assert p0[pf.index("phase:ABILITY_STACK_TARGET")] == 1.0
    assert p0[pf.index("src:Soldier")] == 1.0
    assert not np.array_equal(pg, p0)                  # different phase
    assert not np.array_equal(p0, p1)                  # same phase, different multi-select progress
    assert p1[pf.index("limit")] == pytest.approx(2 / 3)
    assert p1[pf.index("chosen")] == pytest.approx(1 / 3)


def test_pending_selected_flags_committed_disgrace_target():
    base = _view()
    obs = base.observer
    sol, queen = card_ids_for_name("Soldier")[0], card_ids_for_name("Queen")[0]
    # stack = [Queen @ pos0 (a target), Soldier @ pos1 (leading)]; disgrace has already picked pos0
    v = dataclasses.replace(
        base, stack=(StackCard(queen, False, None), StackCard(sol, False, None)),
        pending=(PendingStep(StepKind.ABILITY_STACK_TARGET, actor=obs, source=sol,
                             limit=2, chosen=(0,)),))
    t = F.tokenize(v)
    tok = t.cards[t.labels.index("stack_below:Queen")]
    assert _cf(tok, "st:pending_selected") == 1.0      # committed but not yet flagged disgraced
    assert _cf(tok, "st:disgraced") == 0.0


def test_determinism():
    v = _view()
    a, b = F.tokenize(v, Action(kind=ActionKind.PLAY_CARD, card=v.own_hand[0])), \
        F.tokenize(v, Action(kind=ActionKind.PLAY_CARD, card=v.own_hand[0]))
    assert np.array_equal(a.cards, b.cards)
    assert np.array_equal(a.board, b.board) and np.array_equal(a.phase, b.phase)


# ---- fuzz: tokenize every ply of real random games and assert invariants --------------------
# This is the strongest "does it function correctly" check -- random legal play naturally exercises
# every ability, reaction and multi-select subturn, and the invariants would catch a dropped card,
# a wrong zone, a stale legal-flag, a NaN, or a phase mismatch anywhere along the way.

from collections import Counter                                            # noqa: E402

from imposterkings.cards import DECK_SIZE, card_name                       # noqa: E402
from imposterkings.actions import ActionKind as _AK                        # noqa: E402

_ZONE_IX = [F.card_token_fields().index(f"zone:{z}") for z in F._ZONES]
_ID_SLICE = slice(0, F._NTYPE)


def _recon_counts(t, zone):
    """Reconstruct {card_name: count} for a zone from its tokens (count is stored as count/7)."""
    c = Counter()
    cnt_ix = F.card_token_fields().index("bel:count")
    for tok, lb in zip(t.cards, t.labels):
        z, _, nm = lb.partition(":")
        if z == zone and not lb.endswith("*"):
            c[nm] += round(float(tok[cnt_ix]) * 7)
    return c


def _assert_ply_invariants(view, t):
    for arr in (t.cards, t.board, t.phase, t.action):
        assert np.isfinite(arr).all()
        assert (arr >= -1e-6).all() and (arr <= 1.2).all()               # 1.2: setup opp_size 8/7
    for tok, lb in zip(t.cards, t.labels):
        assert tok[_ZONE_IX].sum() == pytest.approx(1.0)                  # exactly one zone
        idsum = tok[_ID_SLICE].sum()
        assert idsum == pytest.approx(0.0 if lb == "opp_unknown:?" else 1.0)
    # phase one-hot matches the pending step being decided
    ph = t.phase[: F._N_STEP]
    if view.pending:
        assert ph.sum() == pytest.approx(1.0)
        assert ph[view.pending[-1].kind.value - 1] == 1.0
    else:
        assert ph.sum() == 0.0
    # card conservation: the tokens' per-zone type multisets == the info-set's zones
    assert _recon_counts(t, "my_hand") == Counter(card_name(c) for c in view.own_hand)
    assert _recon_counts(t, "my_setup_discard") == Counter(
        [card_name(view.own_setup_discard)] if view.own_setup_discard is not None else [])
    assert _recon_counts(t, "discard") == Counter(card_name(c) for c in view.discard)
    assert _recon_counts(t, "ante_mine") == Counter(card_name(c) for c in view.antechambers[view.observer])
    assert _recon_counts(t, "ante_opp") == Counter(card_name(c) for c in view.antechambers[1 - view.observer])
    stack_types = Counter(card_name(sc.card) for sc in view.stack)
    assert _recon_counts(t, "leading") + _recon_counts(t, "stack_below") == stack_types
    # is_legal_now flags exactly the card types with a legal PLAY_CARD right now
    flagged = {lb.split(":")[1] for tok, lb in zip(t.cards, t.labels)
               if (lb.startswith("my_hand:") or lb.startswith("ante_mine:"))
               and tok[F.card_token_fields().index("st:legal_now")] == 1.0}
    legal_types = {card_name(a.card) for a in view.legal_moves()
                   if a.kind == _AK.PLAY_CARD and a.card is not None}
    assert flagged == legal_types


def test_fuzz_invariants_over_random_games():
    rng = np.random.default_rng(20260709)
    plies = 0
    for g in range(40):
        s = GameState.deal(np.random.default_rng(g))
        steps = 0
        while not s.is_terminal() and steps < 250:
            view = InformationSet.from_state(s, s.to_play)
            legal = view.legal_moves()
            _assert_ply_invariants(view, F.tokenize(view))
            for a in legal:                                              # action-in marks correctly
                t = F.tokenize(view, a)
                assert t.action[a.kind.value - 1] == 1.0
                if a.card is not None:
                    assert (t.cards[:, F._CAND_IX] == 1.0).any()
                if a.name is not None:
                    assert f"opp_known:{a.name}*" in t.labels or \
                        any(l.startswith("opp_known:") for l in t.labels)
            s = s.apply(legal[rng.integers(len(legal))])
            steps += 1
            plies += 1
    assert plies > 500                                                    # actually exercised many states


# --- tokenize_state / with_action split (the evaluator's tokenize-once hot path) --------------------

def test_split_equals_monolithic_tokenize():
    # with_action(tokenize_state(view), a) must equal tokenize(view, a) for EVERY legal move, and the
    # shared state tokens must never be mutated by the stamping (copy-on-write).
    rng = np.random.default_rng(7)
    for g in range(6):
        s = GameState.deal(np.random.default_rng(g))
        steps = 0
        while not s.is_terminal() and steps < 40:
            view = InformationSet.from_state(s, s.to_play)
            legal = view.legal_moves()
            st = F.tokenize_state(view, legal_moves=legal)
            base = st.cards.copy()
            for a in legal:
                got, want = F.with_action(st, a), F.tokenize(view, a)
                assert np.array_equal(got.cards, want.cards)
                assert np.array_equal(got.action, want.action)
                assert got.labels == want.labels
                assert np.array_equal(got.board, want.board) and np.array_equal(got.phase, want.phase)
            assert np.array_equal(st.cards, base)                     # base tokens untouched
            assert (st.cards[:, F._CAND_IX] == 0.0).all()             # no candidate leaked into the base
            s = s.apply(legal[rng.integers(len(legal))])
            steps += 1


def test_evaluator_determinizes_once_per_leaf(tmp_path):
    # The attention evaluator must not re-run view.legal_moves() (-> determinize) per candidate move.
    torch = pytest.importorskip("torch")
    from imposterkings.machine_learning.attention_model import AttentionModel, AttnConfig
    from imposterkings.machine_learning.attention_model import evaluator_from_model, save

    torch.manual_seed(0)
    ev = evaluator_from_model(AttentionModel(AttnConfig(d_model=32)).eval())
    calls = {"n": 0}
    orig = InformationSet.legal_moves

    def counting(self, *a, **kw):
        calls["n"] += 1
        return orig(self, *a, **kw)

    InformationSet.legal_moves = counting
    try:
        s = GameState.deal(np.random.default_rng(3))
        while len(s.legal_moves()) < 3:                                # a leaf with several candidates
            s = s.apply(s.legal_moves()[0])
        calls["n"] = 0
        ev(s)
        assert calls["n"] == 0                                         # legal moves passed in, never recomputed
    finally:
        InformationSet.legal_moves = orig


def test_state_legal_moves_equal_view_legal_moves_for_tokenize():
    # The evaluator passes state.legal_moves() (TRUE state) where tokenize used view.legal_moves()
    # (a determinized sample). The featurizer only extracts PLAY_CARD legality, which depends solely on
    # observer-known info -- so the tokens must be IDENTICAL under either source, for any determinization.
    rng = np.random.default_rng(11)
    checked = 0
    for g in range(8):
        s = GameState.deal(np.random.default_rng(100 + g))
        steps = 0
        while not s.is_terminal() and steps < 40:
            view = InformationSet.from_state(s, s.to_play)
            via_state = F.tokenize_state(view, legal_moves=s.legal_moves())
            via_view = F.tokenize_state(view, legal_moves=view.legal_moves())   # fresh determinization
            internal = F.tokenize_state(view)                                   # another determinization
            assert np.array_equal(via_state.cards, via_view.cards)
            assert np.array_equal(via_state.cards, internal.cards)
            assert via_state.labels == internal.labels
            checked += 1
            legal = s.legal_moves()
            s = s.apply(legal[rng.integers(len(legal))])
            steps += 1
    assert checked > 200
