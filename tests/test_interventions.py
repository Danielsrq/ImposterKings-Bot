"""machine_learning.interventions: the exact logit decomposition and the causal hooks (Study 10).

Tiny random v2 nets on real positions -- these pin ARITHMETIC (closure, identities, hook hygiene), not
model quality. The identities matter: a last-layer head ablation has a closed-form effect, so if the hooks
disagree with the algebra, every downstream faithfulness number is wrong."""
import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from imposterkings.actions import StepKind
from imposterkings.machine_learning import attention_model as AM
from imposterkings.machine_learning import features2 as F2
from imposterkings.machine_learning import interventions as I
from imposterkings.machine_learning.explain import explain
from imposterkings.state import GameState


def _model(layers=2, d=32, heads=4, seed=0):
    torch.manual_seed(seed)
    return AM.AttentionModel(AM.AttnConfig(d_model=d, n_layers=layers, n_heads=heads, ffn_hidden=64,
                                           dropout=0.1, feat="v2")).eval()


def _positions(n):
    out = []
    for seed in range(n):
        st = GameState.deal(np.random.default_rng(seed), starting_player=seed % 2)
        while st.phase in (StepKind.SETUP_HIDE, StepKind.SETUP_DISCARD):
            st = st.apply(st.legal_moves()[0])
        out.append(st)
    return out


def _batch(n=12):
    toks, views = [], []
    for st in _positions(n):
        v = st.information_set(st.to_play)
        for mv in st.legal_moves()[:3]:
            toks.append(F2.tokenize(v, mv))
            views.append((v, mv))
    b = AM.collate2(toks)
    return {k: b[k] for k in ("cards", "kings", "board", "phase", "action")}, views


def _q(m, b):
    with torch.no_grad():
        return m(b["cards"], b["board"], b["phase"], b["action"], kings=b["kings"])


def _no_hooks(m):
    return all(not mod._forward_pre_hooks and not mod._forward_hooks for mod in m.modules())


@pytest.mark.parametrize("layers", [1, 2])
def test_noop_spec_is_bit_identical_and_leaves_no_hooks(layers):
    m, (b, _) = _model(layers), _batch()
    q0, a0 = _q(m, b)
    with I.intervene(m, I.InterventionSpec()):
        q1, a1 = _q(m, b)
    assert torch.equal(q0, q1) and torch.equal(a0, a1)
    I.decompose(m, b)
    assert _no_hooks(m)
    q2, _ = _q(m, b)
    assert torch.equal(q0, q2)


@pytest.mark.parametrize("layers", [1, 2])
def test_decomposition_closes_and_matches_q(layers):
    m, (b, _) = _model(layers), _batch()
    d = I.decompose(m, b)
    assert np.abs(d.resid).max() < 1e-4
    assert np.allclose(np.tanh(d.logit), d.q, atol=1e-6)
    assert np.allclose(d.attn.sum(-1), d.attn_head, atol=1e-5)          # no ablation: per-token sums agree
    total = sum(d.components.values())
    assert np.allclose(total, d.logit, atol=1e-4)


def test_last_layer_slice_equals_the_drawer():
    m, (b, views) = _model(2), _batch(4)
    d = I.decompose(m, b)
    for i in (0, 3, len(views) - 1):
        p = explain(views[i][0], views[i][1], m, attribution=True)
        assert np.allclose(d.attn[i, -1], p.row0_signed, atol=1e-5)


def test_head_ablation_edits_exactly_the_head_columns():
    m, (b, _) = _model(2), _batch()
    dh = m.cfg.d_model // m.cfg.n_heads
    with I.record(m) as r0:
        _q(m, b)
    spec = I.InterventionSpec(ablations=(I.HeadAblation(layer=1, head=2, mode="zero"),))
    with I.intervene(m, spec), I.record(m) as r1:
        _q(m, b)
    c0, c1 = r0["concat"][1], r1["concat"][1]
    assert torch.all(c1[..., 2 * dh:3 * dh] == 0)
    assert torch.equal(c0[..., :2 * dh], c1[..., :2 * dh]) and torch.equal(c0[..., 3 * dh:], c1[..., 3 * dh:])


@pytest.mark.parametrize("mode", ["zero", "mean"])
def test_last_layer_head_ablation_matches_the_identity(mode):
    m, (b, _) = _model(2), _batch()
    means = I.head_means(m, _RowsShim(b), np.arange(b["cards"].shape[0]))
    base = I.decompose(m, b)
    h, dh = 1, m.cfg.d_model // m.cfg.n_heads
    spec = I.InterventionSpec(ablations=(I.HeadAblation(layer=1, head=h, mode=mode),))
    da = I.decompose(m, b, spec, means=means)
    assert np.abs(da.resid).max() < 1e-4                                 # closure holds under ablation
    with torch.no_grad():
        u = (m.head.weight[0] @ m.layers[1].attn.wo.weight).view(m.cfg.n_heads, dh)[h]
        repl = 0.0 if mode == "zero" else float(u @ means[1, 0, h * dh:(h + 1) * dh])
    # the DIRECT term is the identity: the head's slot now carries exactly the replacement
    assert np.allclose(da.attn_head[:, 1, h], repl, atol=1e-5)
    keep = [x for x in range(m.cfg.n_heads) if x != h]
    assert np.allclose(da.attn_head[:, :, keep], base.attn_head[:, :, keep], atol=1e-6)
    assert np.allclose(da.ffn[:, 0], base.ffn[:, 0], atol=1e-6)           # upstream untouched
    # ...but the TOTAL is not: the last layer's FFN reads CLS after attention (an indirect path)
    direct = repl - base.attn_head[:, 1, h]
    via_ffn = da.ffn[:, 1] - base.ffn[:, 1]
    assert np.allclose(da.logit - base.logit, direct + via_ffn, atol=1e-4)
    assert np.allclose(I.logits(m, b, spec, means=means), da.logit, atol=1e-6)


def test_mean_ablation_requires_means():
    m, (b, _) = _model(2), _batch(2)
    with pytest.raises(ValueError):
        I.logits(m, b, I.InterventionSpec(ablations=(I.HeadAblation(0, 0, "mean"),)))
    assert _no_hooks(m)


def test_knockout_zeroes_and_renormalises():
    m, (b, _) = _model(2), _batch()
    j = I.ASSASSIN_SEQ
    with I.record(m) as r0:
        _q(m, b)
    spec = I.InterventionSpec(knockouts=(I.AttnKnockout(layer=1, keys=(j,), head=0),))
    with I.intervene(m, spec), I.record(m) as r1:
        _q(m, b)
    a0, a1 = r0["attn"][1][:, 0, 0], r1["attn"][1][:, 0, 0]               # head 0, CLS row
    assert torch.all(a1[:, j] == 0)
    expect = a0 / (1 - a0[:, j:j + 1])
    keep = [k for k in range(a0.shape[-1]) if k != j]
    assert torch.allclose(a1[:, keep], expect[:, keep], atol=1e-5)
    assert torch.equal(r0["attn"][1][:, 1:], r1["attn"][1][:, 1:])         # other heads untouched


def test_knockout_all_but_one_and_closed_form():
    m, (b, _) = _model(2), _batch()
    S, h, k = 24, 1, 5
    others = tuple(x for x in range(S) if x != k)
    d = I.decompose(m, b, I.InterventionSpec(knockouts=(I.AttnKnockout(1, others, head=h),)))
    with I.intervene(m, I.InterventionSpec(knockouts=(I.AttnKnockout(1, others, head=h),))), \
            I.record(m) as r:
        _q(m, b)
    assert torch.allclose(r["attn"][1][:, h, 0, k], torch.ones(1), atol=1e-6)
    base = I.decompose(m, b)
    c_k = base.attn[:, 1, h, k] / np.maximum(_attn(m, b)[:, h, 0, k], 1e-12)  # u.v_k
    assert np.allclose(d.attn_head[:, 1, h], c_k, atol=1e-4)
    # closed form for knocking out ONE key j at the last layer, DIRECT term:
    # d(attn_head) = A_j/(1-A_j) * (T - c_j)   (the final FFN then adds an indirect change on top)
    j = 7
    A = _attn(m, b)[:, h, 0, :]
    c = base.attn[:, 1, h, :] / np.maximum(A, 1e-12)
    T = (A * c).sum(-1)
    pred = A[:, j] / (1 - A[:, j]) * (T - c[:, j])
    dk = I.decompose(m, b, I.InterventionSpec(knockouts=(I.AttnKnockout(1, (j,), head=h),)))
    assert np.allclose(dk.attn_head[:, 1, h] - base.attn_head[:, 1, h], pred, atol=1e-5)
    assert np.allclose(dk.logit - base.logit,
                       (dk.attn_head[:, 1, h] - base.attn_head[:, 1, h]) + (dk.ffn[:, 1] - base.ffn[:, 1]),
                       atol=1e-4)


def test_knocking_out_every_key_raises():
    m, (b, _) = _model(1), _batch(2)
    with pytest.raises(ValueError):
        I.logits(m, b, I.InterventionSpec(knockouts=(I.AttnKnockout(0, tuple(range(24))),)))
    assert _no_hooks(m)


def test_residual_patch_noop_and_full_donor():
    m, (b, _) = _model(2), _batch()
    base = I.logits(m, b)
    with I.record(m) as r:
        _q(m, b)
    same = I.logits(m, b, patch=(I.ResidualPatch(1, (3, 7)), r["resid_in"][1][:, [3, 7]]))
    assert np.allclose(same, base, atol=1e-6)
    donor = {k: v.roll(1, 0) for k, v in b.items()}                       # row i gets row i-1's input
    with I.record(m) as rd:
        _q(m, donor)
    full = I.logits(m, b, patch=(I.ResidualPatch(0, tuple(range(24))), rd["resid_in"][0]))
    assert np.allclose(full, np.roll(base, 1), atol=1e-5)


def test_cf_lacks_equals_tokenizing_the_counterfactual_view():
    checked = 0
    for st in _positions(30):
        v = st.information_set(st.to_play)
        mv = st.legal_moves()[0]
        tok = F2.tokenize(v, mv)
        cf, ok = I.cf_assassin(tok.cards[None], "lacks")
        if not ok[0]:
            continue
        cv = dataclasses.replace(v, opp_hand_lacks=v.opp_hand_lacks | {"Assassin"})
        ref = F2.tokenize(cv, mv).cards
        assert np.allclose(cf[0], ref, atol=1e-6)
        diff = np.abs(cf[0] - tok.cards).sum(-1) > 0
        assert diff.sum() == 1 and diff[I.ASSASSIN_SLOT]
        assert np.isclose(cf[0, I.ASSASSIN_SLOT, I.ZONE_SL].sum(), 1.0)
        assert set(cv.legal_moves()) == set(v.legal_moves())
        checked += 1
    assert checked >= 5


def test_cf_variants_and_not_applicable_rows():
    st = _positions(1)[0]
    v = st.information_set(st.to_play)
    cards = np.stack([F2.tokenize(v, None).cards] * 2)
    cards[1, I.ASSASSIN_SLOT, I.ZONE_SL] = 0.0                              # make row 1 "seen" (a delta)
    cards[1, I.ASSASSIN_SLOT, I.DISCARD_COL] = 1.0
    dead, ok = I.cf_assassin(cards, "dead")
    if ok[0]:
        assert dead[0, I.ASSASSIN_SLOT, I.DISCARD_COL] == 1.0
    assert not ok[1] and np.array_equal(dead[1], cards[1])
    cold, _ = I.cf_assassin(cards, "cold")
    if ok[0]:
        assert cold[0, I.ASSASSIN_SLOT, I.REACTION_COL] == 0.0
    with pytest.raises(ValueError):
        I.cf_assassin(cards, "nope")


def test_ablated_model_with_empty_spec_plays_like_the_plain_model():
    from imposterkings.arena import play_game
    from imposterkings.machine_learning.agent import NNAgent, NNPolicy
    m = _model(2)
    wrapped = I.AblatedModel(m, I.InterventionSpec())
    for seed in range(3):
        moves = {}
        for tag, net in (("plain", m), ("wrapped", wrapped)):
            seen = []
            ag = NNAgent(NNPolicy(net))
            play_game([ag, ag], np.random.default_rng(seed), on_decision=lambda s, vw, mv, a, st: seen.append(mv))
            moves[tag] = seen
        assert moves["plain"] == moves["wrapped"]
    import pickle
    pickle.loads(pickle.dumps(I.AblatedModel(m, I.InterventionSpec(ablations=(I.HeadAblation(0, 1, "zero"),)))))


# --- helpers -----------------------------------------------------------------------------------

class _RowsShim:
    """Minimal TokenRows stand-in over an in-memory batch (for head_means)."""

    def __init__(self, b):
        for k, v in b.items():
            setattr(self, k, v.numpy())


def _attn(m, b):
    with I.record(m) as r:
        _q(m, b)
    return r["attn"][len(m.layers) - 1].numpy().astype(np.float64)
