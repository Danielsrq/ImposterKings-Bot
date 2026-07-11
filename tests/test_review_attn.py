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
