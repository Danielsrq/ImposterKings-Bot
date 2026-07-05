"""Self-play datagen -> replayable JSONL -> review replay: round-trip fidelity."""
from __future__ import annotations

import os

import numpy as np

from imposterkings import record
from imposterkings.actions import Action
from imposterkings.analysis import datagen
from imposterkings.state import GameState

_SPEC = ("hybrid", 3, 3)   # tiny budget -> fast


def test_collect_game_is_replayable_and_targets_backfilled():
    rec = datagen.collect_game(_SPEC, seed=7, temp_plies=0, base_seed=0)
    assert rec.deal_seed == 7 and rec.gen["spec"] == "hybrid-k3-l3" and rec.schema_version == 1
    assert rec.winner in (0, 1) and rec.decisions and rec.starting_player in (0, 1)
    for d in rec.decisions:                                   # z back-filled from the mover's reward
        assert d.z == rec.rewards[d.seat]
    # deal_seed + the ordered action log reconstruct the exact game (apply is deterministic)
    st = GameState.deal(np.random.default_rng(rec.deal_seed))
    for d in rec.decisions:
        a = record.dict_to_action(d.chosen)
        assert a in st.legal_moves()                         # recorded action legal at its ply
        st = st.apply(a)
    assert st.winner == rec.winner


def test_action_dict_round_trips():
    rec = datagen.collect_game(_SPEC, seed=11, temp_plies=0, base_seed=0)
    for d in rec.decisions:
        a = record.dict_to_action(d.chosen)
        assert isinstance(a, Action)
        assert record.action_to_dict(a) == d.chosen          # exact JSON round-trip


def test_jsonl_write_read_and_scripted_replay(tmp_path):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pytest
    pytest.importorskip("pygame")
    from imposterkings.ui.review import scripted_trajectory

    rec = datagen.collect_game(_SPEC, seed=5, temp_plies=0, base_seed=0)
    path = str(tmp_path / "games.jsonl")
    record.write_jsonl(path, [rec])
    loaded = record.read_jsonl(path)
    assert len(loaded) == 1 and loaded[0]["deal_seed"] == 5

    r = loaded[0]
    st = GameState.deal(np.random.default_rng(r["deal_seed"]))
    moves = [record.dict_to_action(d["chosen"]) for d in r["decisions"]]
    traj = scripted_trajectory(st, moves, search=False)       # fast path: no re-search
    assert len(traj) == len(moves)
