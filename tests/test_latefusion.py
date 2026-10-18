"""Late-fusion model (Study 11): shapes, candidate independence / padding invariance, exact decomposition,
checkpoint round-trip and the evaluator / greedy-agent dispatch. Tiny random nets on real positions."""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from imposterkings.machine_learning import features3 as F3
from imposterkings.machine_learning import latefusion_model as LF
from imposterkings.machine_learning.agent import NNPolicy
from imposterkings.machine_learning.benchmark import _evaluator_for
from imposterkings.state import GameState


def _positions(n=30, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for g in range(n):
        st = GameState.deal(np.random.default_rng(500 + g), starting_player=g % 2)
        for _ in range(int(rng.integers(0, 25))):
            if st.is_terminal():
                break
            mv = st.legal_moves()
            st = st.apply(mv[rng.integers(len(mv))])
        if not st.is_terminal():
            out.append(st)
    return out


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return LF.LateFusionModel(LF.LFConfig(d_model=32, n_self=2, n_cross=2, n_heads=4, ffn_hidden=64,
                                          dropout=0.0)).eval()


@pytest.fixture(scope="module")
def batch():
    sts, cds = [], []
    for st in _positions():
        v = st.information_set(st.to_play)
        s, c = F3.tokenize3(v, st.legal_moves())
        sts.append(s)
        cds.append(c)
    return LF.collate(sts, cds), [len(c.kind) for c in cds]


def test_forward_shapes(model, batch):
    b, ns = batch
    out = model(b)
    B, N = b["cand_kind"].shape
    assert out["q"].shape == out["logit"].shape == (B, N)
    assert len(out["attn_self"]) == 2 and out["attn_self"][0].shape == (B, 4, 23, 23)
    assert len(out["attn_cross"]) == 2 and out["attn_cross"][0].shape == (B, 4, N, 23)
    for i, n in enumerate(ns):
        assert torch.isinf(out["logit"][i, n:]).all() and torch.isfinite(out["logit"][i, :n]).all()
    torch.testing.assert_close(out["attn_cross"][1].sum(-1), torch.ones(B, 4, N))


def test_candidates_independent_and_padding_invariant(model):
    st = _positions(5, seed=3)[2]
    v = st.information_set(st.to_play)
    moves = st.legal_moves()
    assert len(moves) >= 2
    s, c = F3.tokenize3(v, moves)
    full = model(LF.collate([s], [c]))["q"][0, :len(moves)]
    # each move alone (a 1-candidate set) and the set padded to 15 give the same q
    for i, m in enumerate(moves):
        one = model(LF.collate([s], [F3.encode_candidates(v, s, [m])]))["q"][0, 0]
        torch.testing.assert_close(one, full[i], atol=1e-6, rtol=0)
    big = F3.Cands3(*(np.pad(getattr(c, f), (0, 15 - len(moves)), constant_values=-1)
                      for f in ("kind", "ptr_a", "ptr_b", "name", "number")))
    torch.testing.assert_close(model(LF.collate([s], [big]))["q"][0, :len(moves)], full, atol=1e-6, rtol=0)


def test_decomposition_closes(model, batch):
    b, ns = batch
    d = LF.decompose(model, b)
    assert d.resid.abs().max() < 1e-4
    assert d.attn.shape[:4] == (b["cand_kind"].shape[0], 2, 4, b["cand_kind"].shape[1])


def test_knockout_bias_moves_attention(model, batch):
    b, _ = batch
    S = 23
    kb = torch.zeros(1, 1, 1, S)
    kb[..., 5] = float("-inf")
    out = model(b, cross_bias=[kb, kb])
    assert out["attn_cross"][0][..., 5].abs().max() == 0
    assert LF.decompose(model, b, cross_bias=[kb, kb]).resid.abs().max() < 1e-4


def test_checkpoint_roundtrip_and_dispatch(model, tmp_path):
    p = str(tmp_path / "lf.pt")
    LF.save(p, model, {"note": "t"})
    m2, meta = LF.load(p)
    assert meta["note"] == "t" and m2.cfg == model.cfg
    st = _positions(4, seed=7)[1]
    v = st.information_set(st.to_play)
    moves = st.legal_moves()
    np.testing.assert_allclose(m2.q_moves(v, moves), model.q_moves(v, moves), atol=1e-6)
    pol = NNPolicy.from_checkpoint(p)
    assert pol.best_move(v) in moves
    ev = _evaluator_for(p)
    value, priors = ev(st)
    q = model.q_moves(v, moves)
    assert value[st.to_play] == pytest.approx(float(q.max()), abs=1e-6)
    assert value[1 - st.to_play] == pytest.approx(-float(q.max()), abs=1e-6)
    assert sum(priors.values()) == pytest.approx(1.0) and set(priors) == set(moves)


def test_batch_from_arrays_matches_collate(model):
    """The training path (npz rows) and the live path (tokenize3 + collate) feed the model identically."""
    sts, cds = [], []
    for st in _positions(8, seed=11):
        v = st.information_set(st.to_play)
        s, c = F3.tokenize3(v, st.legal_moves())
        sts.append(s)
        cds.append(c)
    live = LF.collate(sts, cds)
    K = F3.MAX_CANDS
    arrs = {k: live[k].numpy() for k in ("cards", "card_depth", "hands", "kings", "decision", "src_ptr")}
    for k in LF.CAND_KEYS:
        a = np.full((len(sts), K), -1, np.int8)
        a[:, :live[k].shape[1]] = live[k].numpy()
        arrs[k] = a
    b = LF.batch_from_arrays(arrs, np.arange(len(sts)))
    torch.testing.assert_close(model(b)["q"], model(live)["q"])


def _tiny_arrays(n_games=24):
    """A synthetic v3 per-decision array dict from real positions (labels are random but fixed)."""
    rng = np.random.default_rng(0)
    K = F3.MAX_CANDS
    cols = {k: [] for k in ("cards", "card_depth", "hands", "kings", "decision", "src_ptr") + LF.CAND_KEYS
            + ("y", "w", "chosen", "game_id")}
    for g in range(n_games):
        st = GameState.deal(np.random.default_rng(900 + g), starting_player=g % 2)
        for _ in range(6):
            if st.is_terminal():
                break
            mv = st.legal_moves()
            if len(mv) >= 2:
                v = st.information_set(st.to_play)
                s, c = F3.tokenize3(v, mv)
                n = len(mv)
                for k, f in (("cards", s.cards), ("card_depth", s.card_depth), ("hands", s.hands),
                             ("kings", s.kings), ("decision", s.decision), ("src_ptr", s.src_ptr)):
                    cols[k].append(f)
                for k, fld in zip(LF.CAND_KEYS, ("kind", "ptr_a", "ptr_b", "name", "number")):
                    a = np.full(K, -1, np.int8)
                    a[:n] = getattr(c, fld)
                    cols[k].append(a)
                y = np.zeros(K, np.float32); y[:n] = rng.uniform(-1, 1, n)
                w = np.zeros(K, np.float32); w[:n] = rng.dirichlet(np.ones(n))
                cols["y"].append(y); cols["w"].append(w)
                cols["chosen"].append(int(rng.integers(n))); cols["game_id"].append(g)
            st = st.apply(mv[rng.integers(len(mv))])
    arrs = {k: np.asarray(v) for k, v in cols.items()}
    arrs["src_ptr"] = arrs["src_ptr"].astype(np.int8)
    arrs["card_depth"] = arrs["card_depth"].astype(np.int8)
    arrs["chosen"] = arrs["chosen"].astype(np.int8)
    arrs["decision_id"] = np.arange(len(arrs["game_id"]))
    return arrs


def test_trainer_resume_equals_uninterrupted(tmp_path):
    from imposterkings.machine_learning import train_latefusion as TL
    arrs = _tiny_arrays()
    cfg = LF.LFConfig(d_model=32, n_self=1, n_cross=1, n_heads=4, ffn_hidden=64, dropout=0.1)
    hp = {"epochs": 3, "batch": 16, "lr": 1e-3, "warmup_frac": 0.1, "val_frac": 0.2, "seed": 0,
          "device": "cpu", "beta": 1.0}
    full = TL.run(arrs, cfg, hp, last_path=str(tmp_path / "a.last.pt"))
    part = TL.run(arrs, cfg, hp, last_path=str(tmp_path / "b.last.pt"), max_minutes=1e-9)
    assert not part["done"] and part["epoch"] == 1
    res = TL.run(arrs, cfg, hp, last_path=str(tmp_path / "b.last.pt"), resume=True)
    assert res["done"]
    for (k, a), (_, b) in zip(full["model"].state_dict().items(), res["model"].state_dict().items()):
        torch.testing.assert_close(a, b, msg=k)
    assert full["history"]["val_mse"] == res["history"]["val_mse"]
    with pytest.raises(ValueError):
        TL.run(arrs, cfg, {**hp, "lr": 2e-3}, last_path=str(tmp_path / "b.last.pt"), resume=True)
