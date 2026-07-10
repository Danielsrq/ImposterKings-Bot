"""Headless tests for ui.attention_view (pygame only, no display, no torch): the heatmap draws into an
off-screen surface, produces one hitbox per cell, and the geometry round-trips (cell center -> that cell).
Uses a duck-typed payload (SimpleNamespace) so it needs neither torch nor a real checkpoint."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame = pytest.importorskip("pygame")

from imposterkings.ui import attention_view as av


@pytest.fixture(scope="module", autouse=True)
def _pg():
    pygame.init()
    pygame.font.init()
    yield
    pygame.quit()


def _fonts():
    return {"small": pygame.font.SysFont("consolas,arial", 16)}


def _payload(heads=4):
    seq = ["CLS", "my_hand:Soldier", "opp_unknown:?", "board", "phase", "action"]
    names = [None, "Soldier", None, None, None, None]
    s = len(seq)
    a = np.random.RandomState(0).rand(heads, s, s).astype(np.float32)
    a /= a.sum(axis=-1, keepdims=True)                                # rows sum to 1 like a softmax
    return SimpleNamespace(attn=a, seq_labels=seq, display_names=names, n_heads=heads)


def test_one_hit_per_cell_and_roundtrip():
    surf = pygame.Surface((1000, 900))
    p = _payload()
    hits = av.draw_attention(surf, _fonts(), p, (20, 20, 900, 820), candidate_index=1)
    heads, s, _ = p.attn.shape
    assert len(hits) == heads * s * s
    # geometry round-trips: the cell under a hit's own center is that same cell
    h = hits[len(hits) // 2]
    got = av.attn_cell_at(hits, h.rect.center)
    assert got is not None and (got.i, got.j, got.head) == (h.i, h.j, h.head)


def test_row_norm_and_hover_and_tooltip_dont_raise():
    surf = pygame.Surface((1000, 900))
    p = _payload()
    hits = av.draw_attention(surf, _fonts(), p, (20, 20, 900, 820),
                             mode="row_norm", candidate_index=1, hover=(0, 1, 0))
    assert hits
    av.draw_tooltip(surf, _fonts(), p, hits[7], (500, 400))          # 2dp tooltip near mouse


def test_miss_returns_none():
    surf = pygame.Surface((1000, 900))
    p = _payload()
    hits = av.draw_attention(surf, _fonts(), p, (20, 20, 900, 820))
    assert av.attn_cell_at(hits, (5, 5)) is None                     # outside the grid
