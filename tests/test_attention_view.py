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


def test_exclude_indices_renormalizes_and_keeps_original_indices():
    surf = pygame.Surface((1000, 900))
    p = _payload()
    s = len(p.seq_labels)
    board = p.seq_labels.index("board")
    hits = av.draw_attention(surf, _fonts(), p, (20, 20, 900, 820),
                             exclude_indices=(board,), candidate_index=1)
    assert len(hits) == p.n_heads * (s - 1) ** 2                     # board row+col dropped
    assert all(h.i != board and h.j != board for h in hits)          # hits keep ORIGINAL indices
    # hover on the excluded token is a no-op (no crash); hover on a kept token draws
    av.draw_attention(surf, _fonts(), p, (20, 20, 900, 820),
                      exclude_indices=(board,), hover=(0, board, 0))
    av.draw_attention(surf, _fonts(), p, (20, 20, 900, 820),
                      exclude_indices=(board,), hover=(0, 1, 0))


def test_tooltip_with_attribution_shows_total():
    surf = pygame.Surface((1000, 900))
    p = _payload()
    s = len(p.seq_labels)
    p.row0_signed = np.zeros((p.n_heads, s), np.float32)
    p.attribution = np.arange(s, dtype=np.float32)                   # head-summed totals present
    hits = av.draw_attention(surf, _fonts(), p, (20, 20, 900, 820), mode="signed")
    av.draw_tooltip(surf, _fonts(), p, hits[0], (500, 400))          # renders the Σheads Δq line


def test_signed_mode_diverging():
    surf = pygame.Surface((1000, 900))
    p = _payload()
    s = len(p.seq_labels)
    rs = np.zeros((p.n_heads, s), np.float32)
    rs[0, 1], rs[0, 2] = 0.5, -0.5                                   # token1 raises q, token2 lowers q
    p.row0_signed = rs
    hits = av.draw_attention(surf, _fonts(), p, (20, 20, 900, 820), mode="signed", candidate_index=1)
    assert len(hits) == p.n_heads * s * s
    pos = next(h for h in hits if h.head == 0 and h.i == 0 and h.j == 1)
    neg = next(h for h in hits if h.head == 0 and h.i == 0 and h.j == 2)
    cp, cn = surf.get_at(pos.rect.center), surf.get_at(neg.rect.center)
    assert cp[:3] != cn[:3]                                          # opposite signs -> distinct colors
    assert cp[1] > cp[0] and cn[0] > cn[1]                           # positive greener, negative redder


def _payload_l2(heads=4):
    # L2 payload: last layer (attn) uniform; layer 1 carries a distinctive hot card->card cell.
    p = _payload(heads)
    s = len(p.seq_labels)
    l1 = np.full((heads, s, s), 1.0 / s, np.float32)
    l1[0, 2, 1] = 0.9                                                # hot L1 card-row cell
    p.per_layer = [l1, p.attn.copy()]
    return p


def test_routed_attention_composites_layers():
    p = _payload_l2()
    m, routed, dead = av.routed_attention(p)
    assert routed and not dead                                       # composite: every row causal
    assert np.array_equal(m[:, 0, :], p.attn[:, 0, :])               # row 0 = last layer (readout)
    assert np.array_equal(m[:, 1:, :], p.per_layer[0][:, 1:, :])     # card rows = layer 1 (causal)
    assert m[0, 2, 1] == np.float32(0.9)
    # L1 payload: no routing, card rows are the dead ones
    p1 = _payload()
    m1, routed1, dead1 = av.routed_attention(p1)
    assert not routed1 and dead1 and np.array_equal(m1, p1.attn)


def test_l2_render_and_tooltip_use_routed_rows():
    surf = pygame.Surface((1000, 900))
    p = _payload_l2()
    hits = av.draw_attention(surf, _fonts(), p, (20, 20, 900, 820), candidate_index=1)
    s = len(p.seq_labels)
    assert len(hits) == p.n_heads * s * s
    # the hot L1 cell must render brighter than its uniform neighbor (card rows come from layer 1)
    hot = next(h for h in hits if h.head == 0 and h.i == 2 and h.j == 1)
    cold = next(h for h in hits if h.head == 0 and h.i == 2 and h.j == 2)
    assert sum(surf.get_at(hot.rect.center)[:3]) > sum(surf.get_at(cold.rect.center)[:3])
    av.draw_tooltip(surf, _fonts(), p, hot, (500, 400))              # layer-tagged tooltip renders
    # signed mode at L2: row 0 diverging, card rows still the (viridis) L1 view -- no crash
    p.row0_signed = np.zeros((p.n_heads, s), np.float32)
    p.attribution = np.zeros(s, np.float32)
    av.draw_attention(surf, _fonts(), p, (20, 20, 900, 820), mode="signed")


def test_layer_view_pills_select_matrices():
    p = _payload_l2()
    m_causal, r1, d1 = av.routed_attention(p, "causal")
    m_l1, r2, d2 = av.routed_attention(p, "l1")
    m_l2, r3, d3 = av.routed_attention(p, "l2")
    assert r1 and r2 and r3
    assert not d1 and not d2 and d3                                  # only the L2 view has dead card rows
    assert np.array_equal(m_l1, p.per_layer[0])
    assert np.array_equal(m_l2, p.attn)
    assert np.array_equal(m_causal[:, 0, :], p.attn[:, 0, :])
    assert np.array_equal(m_causal[:, 1:, :], p.per_layer[0][:, 1:, :])
    # all three views render + tooltip; signed coloring is suppressed in the l1 view (row 0 isn't readout)
    surf = pygame.Surface((1000, 900))
    p.row0_signed = np.zeros((p.n_heads, len(p.seq_labels)), np.float32)
    p.attribution = np.zeros(len(p.seq_labels), np.float32)
    for view in ("causal", "l1", "l2"):
        hits = av.draw_attention(surf, _fonts(), p, (20, 20, 900, 820), mode="signed", layer_view=view)
        av.draw_tooltip(surf, _fonts(), p, hits[0], (500, 400), layer_view=view)
