"""Screen geometry -- where every zone of the game screen lives. Pure arithmetic, no drawing.

Splitting the numbers out from the painting is what lets a zone be re-positioned without reading the paint
code, and it keeps the ONE rule that has bitten us twice: a rect used for drawing and the rect used for
hit-testing must come from the same place. ``action_rects`` is the canonical example -- ``render_frame``
draws with it and ``app._hover_index`` hit-tests with it, so the highlight can never point at the wrong row.
"""
from __future__ import annotations

import math
from typing import List

import pygame

from .theme import CARD, KING, SMALL, WINDOW

# --- vertical bands of the right-hand side panel -------------------------------------------------
PANEL_X = 1290              # side panel starts here (wider panel -> less PV wrapping in reasoning/hint)
KNOW_X = 1050               # hand-knowledge column occupies [KNOW_X, PANEL_X] (~240 wide)
PANEL_W = WINDOW[0] - PANEL_X - 12
ROW_MAX_X = KNOW_X - 12     # right edge of the play area (before the knowledge column)

BTN_TOP = 88        # y of the first action button
BTN_H = 28          # preferred row height; action_rects() shrinks it when a decision has more options
BTN_MIN_H = 20      # floor before spilling into a second column
ACT_BOTTOM = 490    # action buttons must fit in [BTN_TOP, ACT_BOTTOM) -- see action_rects()
LOG_TOP = 505       # "Log" section header
LOG_LINES = 7       # recent log lines shown
REASON_TOP = 690    # "Bot reasoning" section header (+ toggle)
HINT_TOP = 875      # "Your hint" section header (+ toggle)

# The panel's top-right buttons (Attention | Settings) -- the decision header must stop before them.
SETTINGS_X = WINDOW[0] - 12 - 84
ATTN_X = SETTINGS_X - 8 - 92
HDR_MAX_X = ATTN_X - 8

# --- the life strip (a king + its hidden card), right-aligned against the knowledge column --------
LIFE_GAP = 8
LIFE_W = SMALL[0] + LIFE_GAP + KING[0]      # hidden-card slot, then the king (king nearest the column)
LIFE_X = ROW_MAX_X - LIFE_W
KING_X = LIFE_X + SMALL[0] + LIFE_GAP
ROW_MAX_X_CARDS = LIFE_X - 16               # card rows stop here so they never run under a life
OPP_LIFE_Y = 78                             # clear of the top-right buttons
OWN_LIFE_Y = 826

# --- board zone anchors ---------------------------------------------------------------------------
OPP_HAND_Y = 46
OPP_ANTE_Y = 150            # opponent's antechamber: above the stack
OWN_ANTE_Y = 706            # yours: between the stack and your hand
LEFTOVER_Y = 360
STACK_Y = 466
HAND_Y = 832


def row_x(x0: int, count: int, gap: int, card_w: int, x_max: int = ROW_MAX_X) -> List[int]:
    """X positions for a row of ``count`` cards starting at ``x0``, shrinking the gap (cards may
    overlap) so the row always fits within ``x_max`` -- the rightmost card never gets clipped."""
    if count <= 1:
        return [x0]
    span = x_max - x0 - card_w
    g = min(gap, max(12, span / (count - 1)))
    return [int(x0 + i * g) for i in range(count)]


def action_rects(n: int) -> List["pygame.Rect"]:
    """Hit-boxes for ``n`` action buttons, guaranteed to FIT the panel band [BTN_TOP, ACT_BOTTOM).

    Every legal move must be clickable -- there is no CLI to fall back to. So rows shrink toward
    ``BTN_MIN_H`` as the option count grows (the worst real decision is Inquisitor's flattened
    ABILITY_MAY: 14 card names + decline = 15), and only if even that will not fit do we spill into a
    second column. Shared by ``render_frame`` and the app's hover hit-test so the two can never disagree."""
    if n <= 0:
        return []
    avail = ACT_BOTTOM - BTN_TOP
    cols, rows, h = 1, n, BTN_H
    if n * BTN_H > avail:
        h = avail // n
        if h < BTN_MIN_H:                                  # too many to stack -> two columns
            cols, rows = 2, (n + 1) // 2
            h = min(BTN_H, avail // rows)
    w = (PANEL_W - (cols - 1) * 8) // cols
    px = PANEL_X + 12
    out = []
    for i in range(n):
        c, r = divmod(i, rows)
        out.append(pygame.Rect(px + c * (w + 8), BTN_TOP + r * h, w, max(12, h - 4)))
    return out
