"""The right-hand side panel and the hand-knowledge column.

Four stacked sections -- ACTIONS (the clickable legal moves), LOG, the bot's read and your own read -- plus
the knowledge column that sits between the play area and the panel.

The action rows come from ``layout.action_rects``, the SAME function the app hit-tests with, so the drawn
row and the row under the cursor can never disagree. That guarantee is why every legal move stays clickable
even at the worst decision in the game (Inquisitor's flattened ABILITY_MAY: 14 card names + decline = 15).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import pygame

from ..actions import Action
from ..explain import format_action
from . import labels, widgets
from .layout import (ACT_BOTTOM, BTN_TOP, HDR_MAX_X, HINT_TOP, LOG_LINES, LOG_TOP, PANEL_X, REASON_TOP,
                     action_rects)
from .theme import (AMBER, BTN, BTN_HOVER, CARD_COLORS, DIVIDER, GOLD, INK, MUTE, PANEL, P_COLORS, WINDOW)


def draw_background(surface) -> None:
    pygame.draw.rect(surface, PANEL, pygame.Rect(PANEL_X, 0, WINDOW[0] - PANEL_X, WINDOW[1]))


def draw_actions(surface, fonts, view, legal_moves: List[Action], hover: Optional[int],
                 status: str = "") -> List[Tuple["pygame.Rect", Action]]:
    """The decision header + one clickable row per legal move. Returns ``[(rect, move), ...]``."""
    small = fonts["small"]
    px = PANEL_X + 12
    widgets.text_fit(surface, fonts, labels.decision_label(view), (px, 14), HDR_MAX_X - px)
    if status:
        widgets.text(surface, small, status, (px, 42), GOLD)
    ctx = labels.reaction_context(view)
    if ctx:
        widgets.text(surface, small, ctx, (px, 60), INK)

    out: List[Tuple[pygame.Rect, Action]] = []
    for i, (move, rect) in enumerate(zip(legal_moves, action_rects(len(legal_moves)))):
        pygame.draw.rect(surface, BTN_HOVER if hover == i else BTN, rect, border_radius=4)
        label = f"{i + 1}. {format_action(move, view)}"
        while label and small.size(label)[0] > rect.w - 12:      # narrow (2-col) rows: clip, never overflow
            label = label[:-1]
        widgets.text(surface, small, label, (rect.x + 8, rect.y + (rect.h - small.get_height()) // 2))
        out.append((rect, move))
    return out


def draw_log(surface, fonts, log: Optional[List[str]]) -> None:
    small = fonts["small"]
    px = PANEL_X + 12
    pygame.draw.line(surface, DIVIDER, (PANEL_X + 8, LOG_TOP - 12), (WINDOW[0] - 8, LOG_TOP - 12))
    widgets.text(surface, small, "Log:", (px, LOG_TOP), MUTE)
    for i, line in enumerate((log or [])[-LOG_LINES:]):
        widgets.text(surface, small, line, (px, LOG_TOP + 22 + i * 20), INK)


def draw_explain(surface, fonts, result, top: int, depth: int = 5, own_eval=None, seat=None) -> None:
    """The top-2 principal-variation lines for a search ``result`` (chess-engine style: [eval] then move
    labels colored by the player who moved). With ``own_eval``/``seat``, a prominent ``P{seat} sees +X.XX``
    line comes first -- correct even when that seat is not the mover. Header/toggle drawn by the caller."""
    small = fonts["small"]
    x = PANEL_X + 12
    max_x = WINDOW[0] - 12
    widgets.text(surface, small, f"{result.iterations} sims, {result.elapsed:.2f}s", (x, top), MUTE)
    widgets.text(surface, small, "P0", (x + 150, top), P_COLORS[0])
    widgets.text(surface, small, "P1", (x + 178, top), P_COLORS[1])

    y = top + 22
    if own_eval is not None and seat is not None:
        widgets.text(surface, small, f"P{seat} sees {own_eval:+.2f}", (x, y), P_COLORS.get(seat, INK))
        y += 20
    lines = result.principal_variations(top=2, depth=depth)
    if not lines:
        widgets.text(surface, small, "(no lines)", (x, y), MUTE)
        return
    y += 2
    for line in lines:
        toks = [(f"[{line[0].mean_q:+.2f}]", INK)]
        toks += [(labels.compact_action(step.move), P_COLORS.get(step.player, INK)) for step in line]
        y = widgets.tokens(surface, small, toks, x, y, max_x, 19)
        y += 4   # gap between lines


def draw_reasoning_section(surface, fonts, top, title, result, shown, placeholder,
                           own_eval=None, seat=None, mouse=None, pending: bool = False) -> "pygame.Rect":
    """Header + [hide]/[show] toggle at ``top``; render the PV lines when ``shown``. Returns the toggle
    Rect. Shared by the bot-reasoning and human-hint panels.

    ``pending``: the search for this panel is queued or running on the worker. Searches became ASYNCHRONOUS
    (they used to block the UI thread), which opened a window where the panel is switched ON but its result
    has not landed yet -- and the panel then showed its "toggle to see this" placeholder, telling you to
    turn on something that was already on. Say "thinking" instead: the panel is working, not idle."""
    small = fonts["small"]
    px = PANEL_X + 12
    pygame.draw.line(surface, DIVIDER, (PANEL_X + 8, top - 12), (WINDOW[0] - 8, top - 12))
    widgets.text(surface, small, title, (px, top), GOLD)
    toggle = widgets.button(surface, small, pygame.Rect(WINDOW[0] - 12 - 64, top - 3, 64, 22),
                            "[hide]" if shown else "[show]", mouse)
    if shown:
        if result is not None:
            draw_explain(surface, fonts, result, top + 26, own_eval=own_eval, seat=seat)
        else:
            msg, colour = ("(thinking...)", GOLD) if pending else (placeholder, MUTE)
            widgets.text(surface, small, msg, (px, top + 26), colour)
    return toggle


def draw_reads(surface, fonts, view, *, bot_result, show_reasoning, bot_eval,
               hint_result, show_hint, hint_eval, mouse,
               bot_pending: bool = False, hint_pending: bool = False):
    """The two PV sections: the bot's read and your own. Returns ``(reasoning_toggle, hint_toggle)``."""
    opp = 1 - view.observer
    reasoning = draw_reasoning_section(surface, fonts, REASON_TOP, f"Bot P{opp} read (MCTS):",
                                       bot_result, show_reasoning, "(no search yet)",
                                       own_eval=bot_eval, seat=opp, mouse=mouse, pending=bot_pending)
    hint = draw_reasoning_section(surface, fonts, HINT_TOP, f"Your P{view.observer} read (MCTS):",
                                  hint_result, show_hint, "(toggle for your read of this position)",
                                  own_eval=hint_eval, seat=view.observer, mouse=mouse,
                                  pending=hint_pending)
    return reasoning, hint


# --- the hand-knowledge column (between the play area and the panel) -------------------------------

def _know_panel(surface, fonts, x, y, max_x, title, facts) -> None:
    """One knower's read on the other's hand: title + PERFECT/50-50 chip + a tick row and a cross row."""
    from .layout import KNOW_X  # noqa: F401  (kept local: the column's own geometry)
    small = fonts["small"]
    has, lacks, level = facts
    widgets.text(surface, small, title, (x, y), MUTE)
    yy = y + 20
    if level:
        txt = "PERFECT INFO" if level == "perfect" else "50-50"
        chip = pygame.Rect(x, yy, small.size(txt)[0] + 12, 20)
        pygame.draw.rect(surface, GOLD if level == "perfect" else AMBER, chip, border_radius=4)
        surface.blit(small.render(txt, True, (20, 20, 20)), (x + 6, yy + 2))
        yy += 26
    widgets.tick(surface, x, yy + 2)
    pos = [(n, CARD_COLORS.get(n, INK)) for n in sorted(has)] or [("(none)", MUTE)]
    yy = widgets.tokens(surface, small, pos, x + 20, yy, max_x, 20)
    widgets.cross(surface, x, yy + 4)
    neg = [(n, CARD_COLORS.get(n, INK)) for n in sorted(lacks)] or [("(none)", MUTE)]
    widgets.tokens(surface, small, neg, x + 20, yy + 2, max_x, 20)


def draw_knowledge(surface, fonts, view, knowledge) -> None:
    """What each player has deduced about the other's hand -- all from PUBLIC events, so showing both reads
    leaks nothing."""
    from .layout import KNOW_X
    if knowledge is None:
        return
    pygame.draw.line(surface, DIVIDER, (KNOW_X, 0), (KNOW_X, WINDOW[1]))
    x0, max_x = KNOW_X + 10, PANEL_X - 10
    widgets.text(surface, fonts["med"], "Hand knowledge", (x0, 12), INK)
    obs, opp = view.observer, 1 - view.observer
    # Each knower's read sits on that player's side: opponent (P{opp}) at top, you (P{obs}) at bottom.
    _know_panel(surface, fonts, x0, 64, max_x, f"P{opp} knows — your hand", knowledge[opp])
    _know_panel(surface, fonts, x0, 804, max_x, f"You (P{obs}) know — P{opp}'s hand", knowledge[obs])
