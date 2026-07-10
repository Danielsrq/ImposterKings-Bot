"""Sprite-axis attention heatmap for the explainability drawer (post-`explain` payload -> pygame).

Draws the per-head ``S x S`` CLS/token attention as a small-multiples grid (2x2 for the default 4 heads)
with **shared sprite axes** (one vertical strip per grid row, one horizontal strip per grid column -- the
tokens are identical on both axes, so a strip labels every box in its row/column). Row i = query, col j =
key; row 0 is the CLS readout. At L=1 only row 0 (and the candidate row/col) is load-bearing, so other rows
are drawn de-emphasized. Pure drawing + geometry (no torch); consumes an ``AttentionExplanation``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pygame

from ..cards import card_ids_for_name
from . import assets
from .render import BG, GOLD, INK, MUTE, PANEL

RGB = Tuple[int, int, int]

AX = 22            # axis-gutter thickness (one tiny sprite)
GAP = 20           # gap between head boxes
PAD = 8
BOX_MAX = 320      # cap on a single head box (keeps cells from getting huge on small-S turns)
MIN_CELL = 6       # floor so big-S turns still tile (sprites just get tiny)

# Perceptual, colorblind-safe "viridis" ramp (purple -> blue -> green -> yellow). Applied to sqrt(weight);
# unlike a navy->white ramp it keeps the low end visible (purple, not near-black), so faint cells still read.
_STOPS: List[Tuple[float, RGB]] = [
    (0.0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 145, 140)),
    (0.75, (94, 201, 98)), (1.0, (253, 231, 37))]

_TILE_FONT = None


def _tile_font():
    global _TILE_FONT
    if _TILE_FONT is None:
        _TILE_FONT = pygame.font.SysFont("consolas,arial", 10)
    return _TILE_FONT


@dataclass
class AttnHit:
    rect: "pygame.Rect"
    i: int          # query (row) seq index
    j: int          # key (col) seq index
    head: int


@dataclass
class AttnGeom:
    """Placement of one head box, so hover geometry is O(heads) not a per-cell scan."""
    head: int
    bx: int
    by: int
    cell: int
    s: int


def _heat(t: float) -> RGB:
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    for (t0, c0), (t1, c1) in zip(_STOPS, _STOPS[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return tuple(int(round(c0[k] + (c1[k] - c0[k]) * f)) for k in range(3))
    return _STOPS[-1][1]


def _blend(c: RGB, bg: RGB, a: float) -> RGB:
    return tuple(int(round(c[k] * a + bg[k] * (1.0 - a))) for k in range(3))


def _grid_shape(n_heads: int) -> Tuple[int, int]:
    cols = int(math.ceil(math.sqrt(n_heads)))
    rows = int(math.ceil(n_heads / cols))
    return rows, cols


def _abbrev(label: str) -> str:
    return {"CLS": "CLS", "board": "brd", "phase": "ph", "action": "act"}.get(label, label[:3])


def _axis_tile(surface, payload, k: int, x: int, y: int, size: int) -> None:
    """Draw the token-``k`` axis sprite in a ``size``-square slot at (x,y): the card art for a card token,
    the card back for ``opp_unknown``, else a small text tile (CLS/board/phase/action)."""
    name = payload.display_names[k]
    label = payload.seq_labels[k]
    slot = pygame.Rect(x, y, size, size)
    if name is not None:
        try:
            img = assets.card_surface(card_ids_for_name(name)[0], (size, size))
            surface.blit(img, (x, y))
        except Exception:
            pygame.draw.rect(surface, MUTE, slot)
        if label.endswith("*") and size >= 10:            # synthetic "claim" token -> asterisk badge
            surface.blit(_tile_font().render("*", True, GOLD), (x + size - 6, y - 1))
        return
    if label.startswith("opp_unknown"):
        try:
            surface.blit(assets.back_surface((size, size)), (x, y))
        except Exception:
            pygame.draw.rect(surface, MUTE, slot)
        return
    pygame.draw.rect(surface, (44, 48, 58), slot)         # context token -> text tile
    pygame.draw.rect(surface, MUTE, slot, 1)
    if size >= 12:
        t = _tile_font().render(_abbrev(label), True, INK)
        surface.blit(t, (x + (size - t.get_width()) // 2, y + (size - t.get_height()) // 2))


def draw_attention(surface, fonts, payload, rect: Tuple[int, int, int, int], *,
                   mode: str = "absolute", emphasize_rows: Tuple[int, ...] = (0,),
                   candidate_index: Optional[int] = None,
                   hover: Optional[Tuple[int, int, int]] = None) -> List[AttnHit]:
    """Draw the per-head heatmap grid into ``rect`` and return per-cell hitboxes.

    ``mode``: "absolute" (global max over shown heads) or "row_norm" (each query row by its own max).
    ``emphasize_rows`` + ``candidate_index``: full-color/framed rows & the candidate row+column; other
    rows are blended toward the background (they are computed-but-discarded at L=1). ``hover`` = (i,j,head)
    draws the crosshair. Returns ``List[AttnHit]`` for :func:`attn_cell_at`."""
    rx, ry, rw, rh = rect
    attn = payload.attn                                   # [heads, S, S], readout layer
    heads, s, _ = attn.shape
    rows, cols = _grid_shape(heads)

    avail_w = (rw - AX - 2 * PAD - (cols - 1) * GAP) // cols
    avail_h = (rh - AX - 2 * PAD - (rows - 1) * GAP) // rows
    box = max(MIN_CELL * s, min(avail_w, avail_h, BOX_MAX))
    cell = max(MIN_CELL, box // s)
    box = cell * s
    grid_w = AX + cols * box + (cols - 1) * GAP
    grid_h = AX + rows * box + (rows - 1) * GAP
    ox = rx + max(0, (rw - grid_w) // 2)
    oy = ry + max(0, (rh - grid_h) // 2)

    gmax = float(attn.max()) or 1.0
    hits: List[AttnHit] = []
    geoms: List[AttnGeom] = []
    for h in range(heads):
        gr, gc = divmod(h, cols)
        bx = ox + AX + gc * (box + GAP)
        by = oy + AX + gr * (box + GAP)
        geoms.append(AttnGeom(h, bx, by, cell, s))
        row_max = attn[h].max(axis=1) if mode == "row_norm" else None
        for i in range(s):
            denom = (float(row_max[i]) or 1.0) if row_max is not None else gmax
            primary_row = (i in emphasize_rows) or (i == candidate_index)
            for j in range(s):
                col = _heat(math.sqrt(max(0.0, float(attn[h, i, j])) / denom))
                if not (primary_row or j == candidate_index):
                    col = _blend(col, BG, 0.45)            # de-emphasize non-load-bearing cells
                r = pygame.Rect(bx + j * cell, by + i * cell, cell, cell)
                pygame.draw.rect(surface, col, r)
                hits.append(AttnHit(r, i, j, h))
        # gold frames on the load-bearing bands
        for i in emphasize_rows:
            if i < s:
                pygame.draw.rect(surface, GOLD, (bx, by + i * cell, box, cell), 1)
        if candidate_index is not None and candidate_index < s:
            ci = candidate_index
            pygame.draw.rect(surface, GOLD, (bx, by + ci * cell, box, cell), 1)
            pygame.draw.rect(surface, GOLD, (bx + ci * cell, by, cell, box), 1)
        # shared axes: vertical strip once per grid row (gc==0), horizontal strip once per grid column (gr==0)
        if gc == 0:
            for k in range(s):
                _axis_tile(surface, payload, k, ox, by + k * cell, min(AX, cell))
        if gr == 0:
            for k in range(s):
                _axis_tile(surface, payload, k, bx + k * cell, oy, min(AX, cell))

    if hover is not None:
        _draw_crosshair(surface, payload, geoms, hover)
    return hits


def _draw_crosshair(surface, payload, geoms: List[AttnGeom], hover: Tuple[int, int, int]) -> None:
    hi, hj, hh = hover
    g = next((g for g in geoms if g.head == hh), None)
    if g is None or hi >= g.s or hj >= g.s:
        return
    band = pygame.Surface((g.cell * g.s, g.cell), pygame.SRCALPHA)
    band.fill((*GOLD, 60))
    surface.blit(band, (g.bx, g.by + hi * g.cell))                    # row highlight
    colband = pygame.Surface((g.cell, g.cell * g.s), pygame.SRCALPHA)
    colband.fill((*GOLD, 60))
    surface.blit(colband, (g.bx + hj * g.cell, g.by))                 # column highlight
    pygame.draw.rect(surface, INK, (g.bx + hj * g.cell, g.by + hi * g.cell, g.cell, g.cell), 2)
    # enlarge the two involved axis sprites (pop-out beside the box)
    big = min(48, max(24, g.cell * 2))
    _axis_tile(surface, payload, hi, g.bx - big - 4, g.by + hi * g.cell, big)
    _axis_tile(surface, payload, hj, g.bx + hj * g.cell, g.by - big - 4, big)


def attn_cell_at(hits: List[AttnHit], pos) -> Optional[AttnHit]:
    """The cell under ``pos`` (cells don't overlap, first containing rect wins)."""
    for hit in hits:
        if hit.rect.collidepoint(pos):
            return hit
    return None


def draw_tooltip(surface, fonts, payload, hit: AttnHit, pos) -> None:
    """Floating 2dp tooltip: ``<label i> -> <label j> = 0.24`` near the mouse."""
    small = fonts["small"]
    v = float(payload.attn[hit.head, hit.i, hit.j])
    lines = [f"{payload.seq_labels[hit.i]} -> {payload.seq_labels[hit.j]}",
             f"head {hit.head}   weight {v:.2f}"]
    lh = small.get_linesize()
    w = max(small.size(t)[0] for t in lines) + 12
    h = len(lines) * lh + 8
    tx = min(pos[0] + 14, surface.get_width() - w - 4)
    ty = min(pos[1] + 14, surface.get_height() - h - 4)
    pygame.draw.rect(surface, PANEL, (tx, ty, w, h))
    pygame.draw.rect(surface, MUTE, (tx, ty, w, h), 1)
    for i, t in enumerate(lines):
        surface.blit(small.render(t, True, INK), (tx + 6, ty + 4 + i * lh))
