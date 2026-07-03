"""Icicle + outline renderers for the MCTS search tree (post-game review screen).

The icicle is **ply-banded**: a player's whole compound turn (play -> declare -> select ...) is grouped
into one vertically-aligned band, so short branches (e.g. a bare card play) pad blank until the next
player's band. Block **width = visits** (recursive icicle partition) and **color = the card's identity**
(Queen red, ...), NOT the eval -- the eval is printed as text on the block. Perspective: values are
shown from the searching seat's point of view (``+`` = good for that panel's player).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import pygame

from ..actions import Action, ActionKind
from ..cards import CARD_DEFS, card_name
from .render import CARD_COLORS, DIVIDER, GOLD, INK, MUTE, NEUTRAL, _compact_action, _text

RGB = Tuple[int, int, int]

_CARD_KINDS = {ActionKind.PLAY_CARD, ActionKind.HIDE_CARD,
               ActionKind.DISCARD_CARD, ActionKind.CHOOSE_HAND_CARD}

# A representative card color per value (for Mystic's "mute N" -- a value maps to 1-2 cards; pick one).
_VALUE_COLOR = {}
for _d in CARD_DEFS:
    _VALUE_COLOR.setdefault(_d.value, CARD_COLORS.get(_d.name, NEUTRAL))


def move_color(move: Optional[Action]) -> RGB:
    """Color a move by the card it concerns: the played/hidden/given card, the GUESSED card's color,
    or (for a Mystic mute) a representative card of that value. Neutral grey for abilities/reactions."""
    if move is None:
        return NEUTRAL
    if move.kind in _CARD_KINDS and move.card is not None:
        return CARD_COLORS.get(card_name(move.card), NEUTRAL)
    if move.kind == ActionKind.GUESS_CARD and move.name:
        return CARD_COLORS.get(move.name, NEUTRAL)
    if move.kind == ActionKind.CHOOSE_NUMBER and move.number is not None:
        return _VALUE_COLOR.get(move.number, NEUTRAL)
    return NEUTRAL


def _ink_for(bg: RGB) -> RGB:
    """Readable text color for a filled block (dark ink on light fills, light on dark)."""
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    return (20, 20, 20) if lum > 140 else (240, 240, 240)


def _truncate(font, s: str, max_w: int) -> str:
    if max_w <= 0:
        return ""
    if font.size(s)[0] <= max_w:
        return s
    while s and font.size(s + "…")[0] > max_w:
        s = s[:-1]
    return (s + "…") if s else ""


@dataclass
class Block:
    x: float
    y: float
    w: float
    h: float
    label: str
    persp_eval: float
    color: RGB
    on_path: bool          # lies on the actually-played line through this turn
    move: Action
    node: object           # the MCTS Node (for click-to-zoom)
    visits: int
    visit_pct: float       # node.n / (layout root).n * 100


def path_node_ids(root, played_path) -> Set[int]:
    """Ids of the nodes on the actually-played line: walk ``root`` following ``played_path`` moves."""
    ids: Set[int] = set()
    cur = root
    for m in (played_path or []):
        nxt = next((c for c in cur.children.values() if c.incoming_move == m), None)
        if nxt is None:
            break
        ids.add(id(nxt))
        cur = nxt
    return ids


def layout_icicle(root, rect: Tuple[float, float, float, float], observer: int, *,
                  top_k: int = 6, max_turns: int = 6,
                  on_path_ids: Set[int] = frozenset()) -> List[Block]:
    """Ply-banded icicle layout for ``root``'s subtree within ``rect`` = (x, y, w, h).

    x: recursive visit partition (child width = parent width * child.n / parent.n, top-``top_k`` kids).
    y: turn bands -- ``turn_index`` increments each time the mover changes; each band's height is the
    deepest micro-turn within it, so shorter branches pad blank until the next player's band.
    """
    x0, y0, W, H = rect
    raw: List[list] = []                  # [node, x, w, turn_index, local_depth]
    band_maxlocal: Dict[int, int] = {}

    def walk(node, x, w, turn_index, local_depth):
        band_maxlocal[turn_index] = max(band_maxlocal.get(turn_index, 0), local_depth)
        raw.append([node, x, w, turn_index, local_depth])
        total = node.n
        if not total:
            return
        cx = x
        for c in sorted(node.children.values(), key=lambda c: c.n, reverse=True)[:top_k]:
            cw = w * (c.n / total)
            if c.player_just_moved == node.player_just_moved:
                ct, cl = turn_index, local_depth + 1
            else:
                ct, cl = turn_index + 1, 0
            if ct < max_turns:
                walk(c, cx, cw, ct, cl)
            cx += cw

    root_n = root.n or 1
    cx = x0
    for c in sorted(root.children.values(), key=lambda c: c.n, reverse=True)[:top_k]:
        walk(c, cx, W * (c.n / root_n), 0, 0)
        cx += W * (c.n / root_n)

    band_top: Dict[int, int] = {}
    acc = 0
    for b in range(max_turns):
        if b not in band_maxlocal:
            break
        band_top[b] = acc
        acc += band_maxlocal[b] + 1
    row_h = H / max(1, acc)

    blocks: List[Block] = []
    for node, x, w, ti, ld in raw:
        y = y0 + (band_top[ti] + ld) * row_h
        mq = node.w / node.n if node.n else 0.0
        persp = mq if node.player_just_moved == observer else -mq
        blocks.append(Block(x, y, w, row_h, _compact_action(node.incoming_move), persp,
                            move_color(node.incoming_move), id(node) in on_path_ids,
                            node.incoming_move, node, node.n, 100.0 * node.n / root_n))
    return blocks


_MIN_LABEL_W = 30


def draw_icicle(surface, fonts, result, rect: Tuple[int, int, int, int], *,
                played_path=None, zoom_root=None, dim: bool = False,
                top_k: int = 6, max_turns: int = 6) -> List[Block]:
    """Draw the ply-banded icicle for a SearchResult into ``rect``; returns the laid-out blocks.

    ``played_path`` (moves) highlights the played line (trail + the current, deepest box). ``zoom_root``
    lays out that node's subtree full-panel instead of the whole tree. ``dim`` fades a persisted/stale
    tree (a forced move that had no search)."""
    x0, y0, W, H = rect
    small = fonts["small"]
    if result is None or getattr(result, "root", None) is None or not result.root.children:
        _text(surface, small, "(no search tree)", (x0 + 6, y0 + 6), MUTE)
        return []
    on_ids = path_node_ids(result.root, played_path)
    layout_root = zoom_root if (zoom_root is not None and zoom_root.children) else result.root
    blocks = layout_icicle(layout_root, rect, result.info.observer,
                           top_k=top_k, max_turns=max_turns, on_path_ids=on_ids)
    line_h = small.get_linesize()
    for b in blocks:
        r = pygame.Rect(int(b.x), int(b.y), max(1, int(b.w) - 1), max(1, int(b.h) - 1))
        pygame.draw.rect(surface, b.color, r)
        if b.w > _MIN_LABEL_W and b.h >= 11:
            ink = _ink_for(b.color)
            surface.blit(small.render(_truncate(small, b.label, int(b.w) - 6), True, ink),
                         (int(b.x) + 3, int(b.y) + 1))
            if b.h >= 2 * line_h:                      # room for a second line -> eval below the action
                surface.blit(small.render(f"{b.persp_eval:+.2f}", True, ink),
                             (int(b.x) + 3, int(b.y) + 1 + line_h))
        if b.on_path:
            pygame.draw.rect(surface, GOLD, r, 2)
    current = max((b for b in blocks if b.on_path), key=lambda b: b.y, default=None)
    if current is not None:                            # emphasize the box just stepped to
        pygame.draw.rect(surface, INK, pygame.Rect(int(current.x), int(current.y),
                         max(1, int(current.w) - 1), max(1, int(current.h) - 1)), 3)
    if dim:
        fade = pygame.Surface((int(W), int(H)), pygame.SRCALPHA)
        fade.fill((18, 20, 26, 150))
        surface.blit(fade, (int(x0), int(y0)))
    return blocks


def block_at(blocks: List[Block], pos) -> Optional[Block]:
    """The block under ``pos`` (icicle blocks don't overlap, so the first containing rect wins)."""
    x, y = pos
    for b in blocks:
        if b.x <= x < b.x + b.w and b.y <= y < b.y + b.h:
            return b
    return None


def draw_tooltip(surface, fonts, block: Block, pos) -> None:
    """A floating box near ``pos`` showing a hovered block's action, eval, and visit share."""
    small = fonts["small"]
    lines = [block.label, f"eval {block.persp_eval:+.2f}", f"visits {block.visits} ({block.visit_pct:.0f}%)"]
    lh = small.get_linesize()
    w = max(small.size(s)[0] for s in lines) + 12
    h = len(lines) * lh + 8
    tx = min(pos[0] + 14, surface.get_width() - w - 4)
    ty = min(pos[1] + 14, surface.get_height() - h - 4)
    pygame.draw.rect(surface, (16, 18, 24), (tx, ty, w, h))
    pygame.draw.rect(surface, MUTE, (tx, ty, w, h), 1)
    for i, s in enumerate(lines):
        surface.blit(small.render(s, True, INK), (tx + 6, ty + 4 + i * lh))


_ROW_H = 20


def _flatten_outline(root, expanded: Set[tuple], top_k: int) -> List[tuple]:
    rows: List[tuple] = []

    def rec(node, depth, path):
        kids = sorted(node.children.values(), key=lambda c: c.n, reverse=True)[:top_k]
        rows.append((node, depth, path, bool(kids)))
        if kids and path in expanded:
            for c in kids:
                rec(c, depth + 1, path + (c.incoming_move,))

    for c in sorted(root.children.values(), key=lambda c: c.n, reverse=True)[:top_k]:
        rec(c, 0, (c.incoming_move,))
    return rows


def draw_outline(surface, fonts, result, rect: Tuple[int, int, int, int], *,
                 expanded: Set[tuple], scroll: int = 0, played_move: Optional[Action] = None,
                 top_k: int = 8) -> List[Tuple["pygame.Rect", tuple]]:
    """Draw a collapsible indented outline of the tree; returns (row_rect, node_path) for hit-testing.
    Click a path to toggle it in ``expanded``. ``scroll`` is a row offset."""
    x0, y0, W, H = rect
    small = fonts["small"]
    if result is None or getattr(result, "root", None) is None or not result.root.children:
        _text(surface, small, "(no search tree)", (x0 + 6, y0 + 6), MUTE)
        return []
    obs = result.info.observer
    rows = _flatten_outline(result.root, expanded, top_k)
    visible = max(1, int(H // _ROW_H))
    scroll = max(0, min(scroll, max(0, len(rows) - visible)))
    hitmap: List[Tuple[pygame.Rect, tuple]] = []
    for i, (node, depth, path, has) in enumerate(rows[scroll:scroll + visible]):
        ry = y0 + i * _ROW_H
        marker = ("▼" if path in expanded else "▶") if has else " "
        mq = node.w / node.n if node.n else 0.0
        persp = mq if node.player_just_moved == obs else -mq
        pygame.draw.rect(surface, move_color(node.incoming_move), (x0 + 6 + depth * 16, ry + 4, 10, 10))
        label = f"{marker} {_compact_action(node.incoming_move)}  n={node.n}  {persp:+.2f}"
        played = depth == 0 and played_move is not None and node.incoming_move == played_move
        _text(surface, small, label, (x0 + 22 + depth * 16, ry + 2), GOLD if played else INK)
        hitmap.append((pygame.Rect(x0, ry, W, _ROW_H), path))
    if len(rows) > visible:
        _text(surface, small, f"[{scroll + 1}-{scroll + visible} / {len(rows)}  ↑↓ scroll]",
              (x0 + 6, y0 + H - 18), MUTE)
    return hitmap
