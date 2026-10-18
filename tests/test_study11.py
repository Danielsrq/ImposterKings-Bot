"""Study 11 plumbing: step cache / stale guard, the pre-registered arm-selection rule, the v3 'lacks'
counterfactual, the pre-registration hash guard and the resumable games driver."""
import dataclasses
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from imposterkings.machine_learning import features3 as F3
from imposterkings.machine_learning import study11 as S
from imposterkings.machine_learning.dla_faithfulness import StaleStep
from imposterkings.state import GameState


def test_step_cache_and_stale_guard(tmp_path):
    inp = tmp_path / "in.bin"
    inp.write_bytes(b"a")
    c = S.Cache(out=str(tmp_path / "o"))
    calls = []
    r1 = c.step("x", [str(inp)], {"p": 1}, lambda: calls.append(1) or {"v": 1})
    r2 = S.Cache(out=str(tmp_path / "o")).step("x", [str(inp)], {"p": 1}, lambda: calls.append(1) or {"v": 2})
    assert r1 == r2 == {"v": 1} and len(calls) == 1
    inp.write_bytes(b"b")
    with pytest.raises(StaleStep):
        S.Cache(out=str(tmp_path / "o")).step("x", [str(inp)], {"p": 1}, lambda: {"v": 3})
    assert S.Cache(out=str(tmp_path / "o"), force=["x"]).step("x", [str(inp)], {"p": 1}, lambda: {"v": 3}) == {"v": 3}


def test_select_arm_rule():
    arms = {"A": {"top1_bestq": 0.640, "params": 280, "beta": 1.0},
            "B": {"top1_bestq": 0.643, "params": 390, "beta": 1.0},
            "C": {"top1_bestq": 0.630, "params": 500, "beta": 1.0},
            "Bb0": {"top1_bestq": 0.700, "params": 390, "beta": 0.0}}
    assert S.select_arm(arms) == "A"                       # within 0.5pp of the best -> fewer params
    arms["B"]["top1_bestq"] = 0.650
    assert S.select_arm(arms) == "B"                       # beta=0 arms never compete


def _assassin_unknown_view():
    rng = np.random.default_rng(0)
    for g in range(200):
        st = GameState.deal(np.random.default_rng(2000 + g), starting_player=g % 2)
        while not st.is_terminal():
            v = st.information_set(st.to_play)
            if "Assassin" not in v.opp_hand_lacks:
                s = F3.tokenize_state3(v, st.legal_moves())
                post = s.cards[S.ASSASSIN, S.ZONE_SL]
                if post.max() < 1 - 1e-6 and s.cards[S.ASSASSIN, S.THEIR_HAND] > 0.05:
                    return st, v
            mv = st.legal_moves()
            st = st.apply(mv[rng.integers(len(mv))])
    raise AssertionError("no position with an unknown Assassin")


def test_cf_lacks_equals_tokenized_counterfactual():
    st, v = _assassin_unknown_view()
    moves = st.legal_moves()
    s = F3.tokenize_state3(v, moves)
    cf = S.cf_lacks(s.cards[None], np.array([0]))[0]
    v2 = dataclasses.replace(v, opp_hand_lacks=v.opp_hand_lacks | {"Assassin"})
    np.testing.assert_allclose(cf, F3.tokenize_state3(v2, moves).cards, atol=1e-6)


def test_prereg_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "OUT", str(tmp_path))
    with pytest.raises(SystemExit):
        S.load_prereg()
    a, b, c = (tmp_path / n for n in ("lf.pt", "d96.pt", "v3.npz"))
    for f in (a, b, c):
        f.write_bytes(os.urandom(16))
    monkeypatch.setattr(S, "NPZ3", str(c))
    with open(S.prereg_path(), "w") as f:
        json.dump({"selected_ckpt": str(a), "selected_sha256": S.sha256(str(a)), "d96": str(b),
                   "d96_sha256": S.sha256(str(b)), "npz3_sha256": S.sha256(str(c))}, f)
    assert S.load_prereg()["d96"] == str(b)
    a.write_bytes(b"changed")
    with pytest.raises(SystemExit):
        S.load_prereg()


def test_games_resume_plays_only_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "OUT", str(tmp_path))
    played = []

    def fake_chunk(kind, lf, seeds):
        played.extend(seeds)
        return {"kind": kind, "rows": [{"seed": s, "wins": s % 3 if s % 3 < 3 else 0, "score": (s % 3) / 2,
                                        "plies": [1, 1], "seconds": 0.0} for s in seeds], "seconds": 0.0}

    monkeypatch.setattr(S, "_chunk", fake_chunk)
    base = S.SEED_BASE["G1"]
    path = os.path.join(str(tmp_path), "games", "G1.jsonl")
    os.makedirs(os.path.dirname(path))
    with open(path, "w") as f:                               # 3 finished deals + a torn line (a kill)
        for s in range(3):
            f.write(json.dumps({"seed": base + s, "wins": 1, "score": 0.5, "plies": [1, 1], "seconds": 0}) + "\n")
        f.write('{"seed": %d, "wins": 1' % (base + 3))
    n = S.run_games("G1", "x", deals=8, workers=1, chunk=2, max_minutes=None)
    assert n == 8 and sorted(played) == [base + i for i in range(3, 8)]
    rows = S.read_rows(path)
    assert sorted(rows) == [base + i for i in range(8)]
    assert S.summarize("G1", 8)["deals"] == 8
