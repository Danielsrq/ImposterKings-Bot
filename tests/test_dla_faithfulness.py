"""Study 10 plumbing: the resumable step cache, the pre-registration guard, the Assassin-death logger and
the resumable games driver. Tiny random nets + a tiny synthetic token file -- plumbing, not results."""
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from imposterkings import scenario as sb
from imposterkings.actions import ActionKind, StepKind
from imposterkings.machine_learning import attention_model as AM
from imposterkings.machine_learning import dla_ablation_games as G
from imposterkings.machine_learning import dla_faithfulness as DF
from imposterkings.machine_learning import features2 as F2
from imposterkings.state import GameState


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A random d32/L2 v2 checkpoint + a token npz built from real positions (2 decisions per game)."""
    d = tmp_path_factory.mktemp("df")
    torch.manual_seed(0)
    m = AM.AttentionModel(AM.AttnConfig(d_model=32, n_layers=2, n_heads=4, ffn_hidden=64, feat="v2")).eval()
    ck = str(d / "m" / "attn.pt")
    os.makedirs(os.path.dirname(ck))
    AM.save(ck, m, {"npz": str(d / "t.npz")})
    cols = {k: [] for k in ("cards", "kings", "board", "phase", "action", "y", "w", "z", "game_id",
                            "decision_id", "is_chosen")}
    dec = 0
    for seed in range(60):
        st = GameState.deal(np.random.default_rng(seed), starting_player=seed % 2)
        for _ in range(2):
            while st.phase in (StepKind.SETUP_HIDE, StepKind.SETUP_DISCARD) or len(st.legal_moves()) < 2:
                st = st.apply(st.legal_moves()[0])
            v = st.information_set(st.to_play)
            mv = st.legal_moves()
            for i, a in enumerate(mv):
                t = F2.tokenize(v, a)
                for k in ("cards", "kings", "board", "phase", "action"):
                    cols[k].append(getattr(t, k))
                cols["y"].append(np.tanh(np.random.default_rng(dec + i).normal()))
                cols["w"].append(1.0 / len(mv)); cols["z"].append(0.0)
                cols["game_id"].append(seed); cols["decision_id"].append(dec); cols["is_chosen"].append(int(i == 0))
            dec += 1
            st = st.apply(mv[0])
            if st.is_terminal():
                break
    np.savez(str(d / "t.npz"), **{k: np.asarray(v) for k, v in cols.items()})
    return ck, str(d / "t.npz"), str(d / "out")


def test_step_cache_skips_matching_and_refuses_stale(tiny):
    ck, npz, out = tiny
    ctx = DF.Ctx(ck, npz, out_root=out)
    calls = []
    r1 = ctx.step("demo", {"p": 1}, lambda: calls.append(1) or {"x": 1})
    r2 = ctx.step("demo", {"p": 1}, lambda: calls.append(2) or {"x": 2})
    assert r1 == r2 == {"x": 1} and calls == [1]
    with pytest.raises(DF.StaleStep):
        ctx.step("demo", {"p": 2}, lambda: {"x": 3})
    forced = DF.Ctx(ck, npz, out_root=out, force=["demo"])
    assert forced.step("demo", {"p": 2}, lambda: {"x": 3}) == {"x": 3}


def test_coverage_runs_and_closes(tiny):
    ck, npz, out = tiny
    ctx = DF.Ctx(ck, npz, out_root=out)
    r = DF.step_coverage(ctx, "A")
    assert r["max_abs_resid"] < 1e-4
    shares = sum(v["abs_share"] for v in r["contrastive"].values())
    assert abs(shares - 1.0) < 1e-6


def test_partial_units_survive_and_reset_on_config_change(tiny):
    ck, npz, out = tiny
    ctx = DF.Ctx(ck, npz, out_root=out)
    path, units = ctx.partial("unitdemo", {"a": 1})
    assert units == {}
    ctx.save_partial(path, "unitdemo", {"a": 1}, {"u1": 1})
    assert ctx.partial("unitdemo", {"a": 1})[1] == {"u1": 1}
    assert ctx.partial("unitdemo", {"a": 2})[1] == {}


def test_prereg_guard_refuses_a_mismatch(tiny):
    ck, npz, out = tiny
    ctx = DF.Ctx(ck, npz, out_root=out)
    DF.atomic_json(os.path.join(ctx.out, "prereg.json"),
                   {"ckpt_sha256": "nope", "npz_sha256": ctx.npz_sha, "means_sha256": ctx.means_sha()})
    with pytest.raises(SystemExit):
        DF.load_prereg(ctx)
    DF.atomic_json(os.path.join(ctx.out, "prereg.json"),
                   {"ckpt_sha256": ctx.ckpt_sha, "npz_sha256": ctx.npz_sha, "means_sha256": ctx.means_sha()})
    assert DF.load_prereg(ctx)["ckpt_sha256"] == ctx.ckpt_sha


class _Scripted:
    """Flip whenever possible; reveal Assassin / King's Hand whenever possible; else the first move."""

    def select_move(self, view, rng):
        moves = view.legal_moves()
        for k in (ActionKind.REVEAL_KINGSHAND, ActionKind.REVEAL_ASSASSIN, ActionKind.FLIP_KING):
            for m in moves:
                if m.kind == k:
                    return m
        return moves[0]


def _flip_position(with_kh: bool):
    hand0 = ["Soldier", "Elder"] + (["KingsHand"] if with_kh else [])
    return sb.build(hand0=hand0, hand1=["Assassin", "Fool", "Zealot"], stack=["Warlord"], turn_player=0)


def test_logger_counts_an_assassin_death():
    g = G.play_logged([_Scripted(), _Scripted()], 0, initial_state=_flip_position(False))
    s0 = g["seats"][0]
    assert g["winner"] == 1 and s0["assassin_death"] == 1 and s0["flips"] == 1
    assert s0["unsafe_flips"] == 1 and s0["opp_reveal_opps"] == 1 and s0["opp_reveals"] == 1


def test_logger_king_hand_counter_is_not_a_death():
    g = G.play_logged([_Scripted(), _Scripted()], 0, initial_state=_flip_position(True))
    s0 = g["seats"][0]
    assert s0["assassin_death"] == 0 and s0["kh_counters"] == 1


def test_games_resume_plays_only_missing_and_matches_uninterrupted(tiny, tmp_path):
    ck, npz, out = tiny
    ctx = DF.Ctx(ck, npz, out_root=str(tmp_path))
    ctx.means()
    seeds = 6
    full = G.run_arms(ck, ctx.out, ["base", "L1h0"], seeds, 100, workers=1, chunk=2)
    assert full == {"base": seeds, "L1h0": seeds}
    ref = {a: G.read_rows(os.path.join(ctx.out, "games", f"{a}.jsonl")) for a in ("base", "L1h0")}
    # simulate a kill mid-arm: keep the first 2 rows + a torn line
    p = os.path.join(ctx.out, "games", "L1h0.jsonl")
    lines = open(p).read().splitlines()
    with open(p, "w") as f:
        f.write("\n".join(lines[:2]) + "\n" + lines[2][: len(lines[2]) // 2])
    calls = []
    orig = G._chunk
    G._chunk = lambda *a: calls.append(a[3]) or orig(*a)
    try:
        G.run_arms(ck, ctx.out, ["base", "L1h0"], seeds, 100, workers=1, chunk=2)
    finally:
        G._chunk = orig
    played = sorted(s for ch in calls for s in ch)
    have_before = {json.loads(l)["seed"] for l in lines[:2]}
    assert played == sorted(set(range(100, 100 + seeds)) - have_before)
    again = G.read_rows(p)
    assert len(again) == seeds and again == ref["L1h0"]
