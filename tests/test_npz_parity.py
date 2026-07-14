"""The gate that lets the release drop torch: numpy inference must reproduce torch EXACTLY ENOUGH.

torch is 4.24 GB installed and serves a 108,737-parameter net (435 KB). The shipped game therefore reads
weights from a plain ``.npz`` and runs the forward pass in numpy (``npz_infer``). That is only legitimate
if the numbers do not move -- so this pins q, the per-head attention AND the per-head values to 1e-5
against torch, on real positions.

Attention and values are pinned, not just q, because the drawer RENDERS them: a numpy port that got q right
and the attention map subtly wrong would ship a correct bot with a lying explanation, and no q-only test
would catch it.

The known risk is GELU. torch's default is the exact ``erf`` form; numpy ships no ``erf``, so npz_infer
uses Abramowitz & Stegun 7.1.26 (|err| <= 1.5e-7 on erf). If this test fails, UPGRADE THE ERF -- do not
loosen the tolerance.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from imposterkings.actions import StepKind
from imposterkings.machine_learning import attention_model as AM
from imposterkings.machine_learning import features, features2 as F2, npz_infer
from imposterkings.state import GameState

GATE = 1e-5


def _positions(n):
    """Real mid-game positions (past setup), each with its legal moves."""
    out = []
    for seed in range(n):
        st = GameState.deal(np.random.default_rng(seed), starting_player=seed % 2)
        while st.phase in (StepKind.SETUP_HIDE, StepKind.SETUP_DISCARD):
            st = st.apply(st.legal_moves()[0])
        out.append(st)
    return out


@pytest.fixture(scope="module")
def attn_pair(tmp_path_factory):
    """A small trained-shaped attention net, exported to .npz and loaded back through the numpy path.

    Random weights, not a checkpoint: the point is that the ARITHMETIC agrees, and random weights exercise
    the same code paths without depending on a gitignored models/ file (so this runs on a clean clone)."""
    torch.manual_seed(0)
    tm = AM.AttentionModel(AM.AttnConfig(d_model=64, n_layers=2, n_heads=4, ffn_hidden=256,
                                         dropout=0.1, feat="v2")).eval()
    p = tmp_path_factory.mktemp("npz") / "attn.pt"
    AM.save(str(p), tm)
    from imposterkings.machine_learning.export_npz import export
    nm = npz_infer.load(export(str(p)))
    return tm, nm


def test_attention_q_attention_and_values_all_match_torch(attn_pair):
    tm, nm = attn_pair
    dq, da, dv = [], [], []
    for st in _positions(40):
        view = st.information_set(st.to_play)
        for mv in st.legal_moves()[:3]:
            tok = F2.tokenize(view, mv)
            b = AM.collate2([tok])
            with torch.no_grad():
                q_t, attns_t, vals_t = tm.forward_layers(
                    b["cards"], b["board"], b["phase"], b["action"], b["card_mask"],
                    need_values=True, kings=b["kings"])
            q_n, attns_n, vals_n = nm._encode(**npz_infer.batch_of(tok, "v2"))
            dq.append(abs(float(q_t[0]) - float(q_n[0])))
            da += [np.abs(a[0].numpy() - c[0]).max() for a, c in zip(attns_t, attns_n)]
            dv += [np.abs(a[0].numpy() - c[0]).max() for a, c in zip(vals_t, vals_n)]

    assert dq and da and dv                                  # the loop actually ran
    assert max(dq) < GATE, f"q drifted: {max(dq):.2e}"
    assert max(da) < GATE, f"attention map drifted: {max(da):.2e} -- the DRAWER would lie"
    assert max(dv) < GATE, f"values drifted: {max(dv):.2e} -- the attribution would lie"


def test_the_gelu_approximation_is_the_thing_at_risk():
    """Pin the GELU directly, so a failure points at the erf rather than at 'something in the net'."""
    x = (np.random.default_rng(0).standard_normal(20000) * 4).astype(np.float32)
    with torch.no_grad():
        ref = torch.nn.functional.gelu(torch.from_numpy(x)).numpy()      # exact erf form (approximate='none')
    assert np.abs(npz_infer._gelu(x) - ref).max() < 1e-5

    # ...and it must NOT be the tanh approximation, which is a DIFFERENT function (~1e-3 off) that would
    # silently pass a loose q-only test while shifting every attention weight.
    tanh_approx = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    assert np.abs(tanh_approx - ref).max() > 1e-4, "tanh GELU is meant to be measurably wrong here"


def test_layernorm_matches_torch_including_the_biased_variance():
    """torch's LayerNorm normalizes by the POPULATION variance. Using the sample variance (ddof=1) is an
    easy, silent off-by-one that shifts every token by ~2% at d=64."""
    x = np.random.default_rng(1).standard_normal((3, 24, 64)).astype(np.float32)
    w = np.random.default_rng(2).standard_normal(64).astype(np.float32)
    b = np.random.default_rng(3).standard_normal(64).astype(np.float32)
    with torch.no_grad():
        ref = torch.nn.functional.layer_norm(torch.from_numpy(x), (64,), torch.from_numpy(w),
                                             torch.from_numpy(b)).numpy()
    assert np.abs(npz_infer._layer_norm(x, w, b) - ref).max() < 1e-5


def test_mlp_matches_the_existing_numpy_evaluator(tmp_path):
    """The MLP forward was ALREADY numpy (evaluator.build_evaluator) -- torch was only unpickling the file.
    So the .npz path must reproduce it bit for bit; any drift means the export mangled the weights."""
    from imposterkings.machine_learning import checkpoint
    from imposterkings.machine_learning.export_npz import export
    from imposterkings.machine_learning.mlp import MLP

    torch.manual_seed(0)
    dim = features.FEATURE_DIM
    m = MLP(dim, [32, 16]).eval()
    p = tmp_path / "mlp.pt"
    checkpoint.save(str(p), m)
    nm = npz_infer.load(export(str(p)))

    for st in _positions(20):
        view = st.information_set(st.to_play)
        x = np.stack([features.encode(view, mv) for mv in st.legal_moves()]).astype(np.float32)
        with torch.no_grad():
            ref = m(torch.from_numpy(x))[:, 0].numpy()
        assert np.abs(nm.q(x) - ref).max() < GATE


def test_explain_gives_the_same_answer_through_either_backend(attn_pair):
    """The payoff: `explain()` no longer imports torch, and the numpy model drives the DRAWER identically.
    Compares the whole payload -- q, the attention map, per-layer maps, and the signed attribution -- since
    those are exactly the numbers the UI paints."""
    from imposterkings.machine_learning.explain import explain

    tm, nm = attn_pair
    for st in _positions(12):
        view = st.information_set(st.to_play)
        mv = st.legal_moves()[0]
        a = explain(view, mv, tm, all_layers=True, attribution=True, ckpt_id="x")
        b = explain(view, mv, nm, all_layers=True, attribution=True, ckpt_id="x")

        assert abs(a.q - b.q) < GATE
        assert np.abs(a.attn - b.attn).max() < GATE
        assert np.abs(a.attribution - b.attribution).max() < GATE
        assert np.abs(a.row0_signed - b.row0_signed).max() < GATE
        for pa, pb in zip(a.per_layer, b.per_layer):
            assert np.abs(pa - pb).max() < GATE
        # the non-numeric payload (labels, candidate indices, the belief block) must be identical too
        assert a.seq_labels == b.seq_labels
        assert a.candidate_seq_index == b.candidate_seq_index
        assert a.candidate_seq_indices == b.candidate_seq_indices
        assert a.card_seen == b.card_seen and a.n_heads == b.n_heads and a.n_layers == b.n_layers
        assert np.abs(a.zone_posterior - b.zone_posterior).max() < GATE


def test_batched_q_equals_one_at_a_time():
    """q_batch is the MCTS leaf path (every legal move scored in ONE forward). It must agree with scoring
    the moves individually -- a broken collate would be invisible to a single-row test."""
    torch.manual_seed(0)
    tm = AM.AttentionModel(AM.AttnConfig(d_model=32, n_layers=2, n_heads=4, feat="v2")).eval()
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.pt")
        AM.save(p, tm)
        from imposterkings.machine_learning.export_npz import export
        nm = npz_infer.load(export(p))

    st = _positions(1)[0]
    view = st.information_set(st.to_play)
    toks = [F2.tokenize(view, mv) for mv in st.legal_moves()]
    batched = nm.q_batch(toks)
    singly = np.array([nm._encode(**npz_infer.batch_of(t, "v2"))[0][0] for t in toks])
    assert np.abs(batched - singly).max() < 1e-6
