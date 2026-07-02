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
from ..cards import card_name
from .render import DIVIDER, GOLD, INK, MUTE, _compact_action, _text

RGB = Tuple[int, int, int]

# One color per card name (14 in cards.CARD_NAMES), matching the card art (user-supplied hex codes).
CARD_COLORS: Dict[str, RGB] = {
    "Queen": (245, 87, 14), "Princess": (242, 92, 15), "Sentry": (224, 127, 59),
    "KingsHand": (199, 103, 43), "Warlord": (240, 154, 9), "Mystic": (238, 154, 6),
    "Oathbound": (226, 202, 32), "Soldier": (80, 172, 123), "Judge": (64, 145, 69),
    "Inquisitor": (76, 155, 168), "Zealot": (122, 156, 194), "Elder": (106, 154, 220),
    "Assassin": (168, 171, 224), "Fool": (167, 113, 175),
}
NEUTRAL: RGB = (80, 84, 94)          # non-card micro-decisions (declare/mute/guess/react)
_CARD_KINDS = {ActionKind.PLAY_CARD, ActionKind.HIDE_CARD,
               ActionKind.DISCARD_CARD, ActionKind.CHOOSE_HAND_CARD}


def move_color(move: Optional[Action]) -> RGB:
    """Card color for a card move; a neutral grey for abilities/selections/reactions."""
    if move is not None and move.kind in _CARD_KINDS and move.card is not None:
        return CARD_COLORS.get(card_name(move.card), NEUTRAL)
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
    is_played: bool
    move: Action


def layout_icicle(root, rect: Tuple[float, float, float, float], observer: int, *,
                  top_k: int = 6, max_turns: int = 6,
                  played_move: Optional[Action] = None) -> List[Block]:
    """Ply-banded icicle layout for ``root``'s subtree within ``rect`` = (x, y, w, h).

    x: recursive visit partition (child width = parent width * child.n / parent.n, top-``top_k`` kids).
    y: turn bands -- ``turn_index`` increments each time the mover changes; each band's height is the
    deepest micro-turn within it, so shorter branches pad blank until the next player's band.
    """
    x0, y0, W, H = rect
    raw: List[list] = []                  # [node, x, w, turn_index, local_depth, is_played]
    band_maxlocal: Dict[int, int] = {}

    def walk(node, x, w, turn_index, local_depth, is_root_child):
        band_maxlocal[turn_index] = max(band_maxlocal.get(turn_index, 0), local_depth)
        played = bool(is_root_child and played_move is not None and node.incoming_move == played_move)
        raw.append([node, x, w, turn_index, local_depth, played])
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
                walk(c, cx, cw, ct, cl, False)
            cx += cw

    total = root.n or 1
    cx = x0
    for c in sorted(root.children.values(), key=lambda c: c.n, reverse=True)[:top_k]:
        walk(c, cx, W * (c.n / total), 0, 0, True)
        cx += W * (c.n / total)

    band_top: Dict[int, int] = {}
    acc = 0
    for b in range(max_turns):
        if b not in band_maxlocal:
            break
        band_top[b] = acc
        acc += band_maxlocal[b] + 1
    row_h = H / max(1, acc)

    blocks: List[Block] = []
    for node, x, w, ti, ld, played in raw:
        y = y0 + (band_top[ti] + ld) * row_h
        mq = node.w / node.n if node.n else 0.0
        persp = mq if node.player_just_moved == observer else -mq
        blocks.append(Block(x, y, w, row_h, _compact_action(node.incoming_move), persp,
                            move_color(node.incoming_move), played, node.incoming_move))
    return blocks


_MIN_LABEL_W = 30


def draw_icicle(surface, fonts, result, rect: Tuple[int, int, int, int], *,
                played_move: Optional[Action] = None, top_k: int = 6, max_turns: int = 6) -> List[Block]:
    """Draw the ply-banded icicle for a SearchResult into ``rect``; returns the laid-out blocks."""
    x0, y0, W, H = rect
    small = fonts["small"]
    if result is None or getattr(result, "root", None) is None or not result.root.children:
        _text(surface, small, "(no search tree)", (x0 + 6, y0 + 6), MUTE)
        return []
    blocks = layout_icicle(result.root, rect, result.info.observer,
                           top_k=top_k, max_turns=max_turns, played_move=played_move)
    for b in blocks:
        r = pygame.Rect(int(b.x), int(b.y), max(1, int(b.w) - 1), max(1, int(b.h) - 1))
        pygame.draw.rect(surface, b.color, r)
        if b.w > _MIN_LABEL_W and b.h >= 11:
            txt = _truncate(small, f"{b.label} {b.persp_eval:+.2f}", int(b.w) - 6)
            surface.blit(small.render(txt, True, _ink_for(b.color)), (int(b.x) + 3, int(b.y) + 1))
        if b.is_played:
            pygame.draw.rect(surface, GOLD, r, 2)
    return blocks


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
