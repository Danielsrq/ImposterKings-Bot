"""Headless tests for the review screen's attention-entry builder (attn_entries_for): mover gets the
logged played move (+ the search best when different), the opponent gets its cross-search top-2, and the
no-state / no-model / no-read cases return []. Uses a tiny untrained model -- plumbing, not quality."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pytest.importorskip("pygame")                                # ui.review imports ui.render -> pygame

from imposterkings.machine_learning.attention_model import AttentionModel, AttnConfig
from imposterkings.state import GameState
from imposterkings.ui.review import PlyRecord, attn_entries_for


def _model():
    torch.manual_seed(0)
    return AttentionModel(AttnConfig(d_model=32)).eval()


def _rec(seed=0):
    s = GameState.deal(np.random.default_rng(seed))
    moves = s.legal_moves()
    return s, moves, PlyRecord(seat=s.to_play, move=moves[0], view=s.information_set(s.to_play),
                               result=None, state=s)


def _stub_result(stats_moves, best=None):
    stats = [SimpleNamespace(move=m) for m in stats_moves]
    return SimpleNamespace(stats=stats, best_move=best if best is not None else stats_moves[0], root=None)


def test_mover_without_search_gets_played_move_only():
    s, moves, rec = _rec()
    entries = attn_entries_for(rec, owner=rec.seat, seat=rec.seat, model=_model(), ckpt_id="ck")
    assert len(entries) == 1 and entries[0][0] == rec.move
    assert entries[0][1].attribution is not None and entries[0][1].ckpt_id == "ck"


def test_opponent_without_search_gets_nothing():
    s, moves, rec = _rec()
    assert attn_entries_for(rec, owner=rec.seat, seat=1 - rec.seat, model=_model(), ckpt_id="ck") == []


def test_mover_with_search_adds_best_alternative():
    s, moves, rec = _rec()
    assert len(moves) >= 2
    # stats order = [moves[1], moves[0], ...]: search prefers a DIFFERENT move -> it is the alternative
    rec.result_by_seat = (_stub_result([moves[1]] + [m for m in moves if m != moves[1]]),) * 2
    entries = attn_entries_for(rec, owner=rec.seat, seat=rec.seat, model=_model(), ckpt_id="ck")
    assert [m for m, _ in entries] == [rec.move, moves[1]]           # played first, then the search best
    # search AGREES with the played move -> pill 2 is the best ALTERNATIVE (not deduped to one pill)
    rec.result_by_seat = (_stub_result(moves),) * 2                  # stats[0] == played (moves[0])
    entries = attn_entries_for(rec, owner=rec.seat, seat=rec.seat, model=_model(), ckpt_id="ck")
    assert [m for m, _ in entries] == [rec.move, moves[1]]           # played + top alternative


def test_opponent_with_search_gets_top2():
    s, moves, rec = _rec()
    opp = 1 - rec.seat
    rec.result_by_seat = (_stub_result(moves),) * 2
    entries = attn_entries_for(rec, owner=rec.seat, seat=opp, model=_model(), ckpt_id="ck")
    assert [m for m, _ in entries] == list(moves[:2])
    # payloads are the OPPONENT's view: q finite, attribution present
    assert all(p.attribution is not None and -1.0 <= p.q <= 1.0 for _, p in entries)


def test_guards_return_empty():
    s, moves, rec = _rec()
    assert attn_entries_for(rec, rec.seat, rec.seat, model=None, ckpt_id="ck") == []
    rec.state = None
    assert attn_entries_for(rec, rec.seat, rec.seat, model=_model(), ckpt_id="ck") == []


# --- the CLI wiring -------------------------------------------------------------------------------
# run_review() gates the A key on `attn_loader is not None`, so a main() that forgets to pass one builds
# the whole drawer and then silently never opens it. That is exactly the bug these pin.

def _write_ckpt(tmp_path):
    from imposterkings.machine_learning.attention_model import save
    p = tmp_path / "attn_test.pt"
    save(str(p), _model())
    return str(p)


def test_make_attn_loader_is_none_without_a_checkpoint():
    """None is the signal that DISABLES the drawer -- so no checkpoint must yield exactly None."""
    from imposterkings.ui.review import make_attn_loader
    assert make_attn_loader(None) is None


def test_make_attn_loader_defers_the_load_until_called(tmp_path):
    """Lazy: building the loader must not import torch or read the file -- a review whose drawer is never
    opened pays nothing. The model only materializes on the first press of A."""
    from imposterkings.ui.review import attn_ckpt_id, make_attn_loader
    ck = _write_ckpt(tmp_path)
    loader = make_attn_loader(ck)
    assert callable(loader)
    os.remove(ck)                                    # not read yet -> building it clearly did not load
    with pytest.raises(Exception):
        loader()
    from imposterkings.machine_learning.attention_model import save
    save(ck, _model())
    model, ckpt_id = loader()
    assert model.cfg.d_model == 32                   # the real model came back...
    assert ckpt_id == attn_ckpt_id(ck)               # ...with the fingerprint that keys the memo cache


def test_app_and_review_share_one_fingerprint_and_ckpt_list():
    """The fingerprint keys the explain memo cache; if app and review computed it differently, one screen
    would serve the other's stale explanations. Same objects, not merely equal ones."""
    from imposterkings.ui import app, review
    assert app.attn_ckpt_id is review.attn_ckpt_id
    assert app.DEFAULT_ATTN_CKPTS is review.DEFAULT_ATTN_CKPTS


def test_review_main_hands_run_review_a_loader(tmp_path, monkeypatch):
    """End to end through main(): the standalone `python -m imposterkings.ui.review --attn ...` must reach
    run_review WITH a loader (the drawer is dead otherwise), and --no-attn must reach it with None."""
    import pygame
    from imposterkings.ui import review as R

    seen = {}
    monkeypatch.setattr(R, "run_review", lambda s, f, t, attn_loader=None: seen.update(loader=attn_loader))
    monkeypatch.setattr(R, "build_trajectory", lambda *a, **k: [_rec()[2]])
    monkeypatch.setattr(pygame, "init", lambda: None)
    monkeypatch.setattr(pygame.display, "set_mode", lambda *a, **k: pygame.Surface((8, 8)))
    monkeypatch.setattr(pygame.display, "set_caption", lambda *a, **k: None)
    monkeypatch.setattr(pygame, "quit", lambda: None)
    ck = _write_ckpt(tmp_path)

    R.main(["--attn", ck, "--iters", "8"])
    assert callable(seen["loader"]), "--attn did not arm the drawer"
    assert seen["loader"]()[1] == R.attn_ckpt_id(ck)          # and it loads THAT checkpoint

    seen.clear()
    R.main(["--no-attn", "--iters", "8"])
    assert seen["loader"] is None, "--no-attn must leave the drawer disabled"


def test_review_nn_head_accepts_an_attention_checkpoint(tmp_path):
    """--nn used the MLP-only loader, which dies on an attention ckpt with KeyError('feature_dim'). It must
    dispatch on checkpoint type, exactly as the app does, so an attention net can drive the search too."""
    from imposterkings.machine_learning.benchmark import _evaluator_for
    ev = _evaluator_for(_write_ckpt(tmp_path))                # the call review.main() now makes
    s = GameState.deal(np.random.default_rng(0))
    value, priors = ev(s)
    assert len(value) == 2 and abs(value[0] + value[1]) < 1e-6          # zero-sum leaf
    assert set(priors) == set(s.legal_moves()) and abs(sum(priors.values()) - 1.0) < 1e-5


# --- the drawer follows the PLY, not the turn root -------------------------------------------------

def test_every_ply_is_explainable_for_its_own_mover_not_just_turn_roots():
    """A turn is a SEQUENCE of micro-decisions (play, then a guess, then a reaction), and each ply carries
    its own search. The drawer used to key on the turn ROOT, so stepping through a turn left the heatmap
    frozen on that turn's first decision -- about half of all plies were unreachable."""
    from imposterkings.ui.review import attn_entries_for, build_trajectory, turns_of

    traj = build_trajectory(iters=30, seed=2, cross_eval=True)
    roots = {s for s, _e, _o in turns_of(traj)}
    assert len(roots) < len(traj), "this game has no mid-turn plies -- pick another seed"

    model = _model()
    for i, rec in enumerate(traj):
        mover = rec.seat                                    # THIS ply's decider (not the turn's owner:
        entries = attn_entries_for(rec, mover, mover, model, "ck")   # a reaction flips the mover mid-turn)
        assert entries, f"ply {i} has no explanation for its own mover"
        assert entries[0][0] == rec.move                    # the move actually played comes first


def test_the_opponents_read_exists_at_turn_roots_and_is_reported_absent_elsewhere():
    """Only turn-START plies get the dual cross-search, so the opponent's read genuinely does not exist
    mid-turn. That must come back as [] (the drawer then says so) rather than a wrong or invented answer."""
    from imposterkings.ui.review import attn_entries_for, build_trajectory, turns_of

    traj = build_trajectory(iters=30, seed=2, cross_eval=True)
    roots = {s for s, _e, _o in turns_of(traj)}
    model = _model()
    at_root = [i for i in roots if attn_entries_for(traj[i], traj[i].seat, 1 - traj[i].seat, model, "ck")]
    mid = [i for i in range(len(traj)) if i not in roots
           and attn_entries_for(traj[i], traj[i].seat, 1 - traj[i].seat, model, "ck")]
    assert at_root, "turn roots must carry both seats' reads"
    assert not mid, "a mid-turn ply has no opponent cross-search -- it must report empty, not guess"
