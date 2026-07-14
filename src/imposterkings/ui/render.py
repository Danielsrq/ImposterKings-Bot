"""Compose the game screen: paint every zone, and return the rects the app loop must hit-test.

``render.py`` used to be the UI's junk drawer -- palette, geometry, drawing helpers, action formatting,
every board zone, the side panel, and three modals, in 900 lines. It is now just the COMPOSITION ROOT:
it wires the components together and hands back a :class:`Frame`. Each component is maintained on its own:

    theme.py         colours, card sizes, fonts          (imports nothing from ui -- nothing can cycle back)
    layout.py        where each zone lives + action_rects (pure geometry; draw and hit-test share it)
    widgets.py       text / button / card / wrap / ...    (the generic toolkit)
    labels.py        Action -> human-readable string
    board.py         the play area, one painter per zone
    side_panel.py    actions / log / reads / knowledge column
    how_to_play.py   the rules + card reference modal
    modals.py        the card zoom + engine settings
    attention_view.py the attention drawer (it owns the heatmap, so it owns the drawer)

This module also RE-EXPORTS the public names, so ``from .render import WINDOW, make_fonts, ...`` keeps
working for every existing consumer and test. New code should import from the module that owns the name.
"""
from __future__ import annotations

from typing import List, NamedTuple, Optional, Tuple

import pygame

from ..actions import Action
from . import board, side_panel, widgets
from .board import Paint
from .layout import (ACT_BOTTOM, ATTN_X, BTN_H, BTN_MIN_H, BTN_TOP, HDR_MAX_X, HINT_TOP, KING_X, KNOW_X,
                     LIFE_W, LIFE_X, LOG_LINES, LOG_TOP, PANEL_W, PANEL_X, REASON_TOP, ROW_MAX_X,
                     ROW_MAX_X_CARDS, SETTINGS_X, action_rects, row_x)
from .theme import (AMBER, BG, BTN, BTN_HOVER, CARD, CARD_COLORS, DIVIDER, GOLD, INK, KING, MUTE, NEUTRAL,
                    PANEL, P_COLORS, RED, SMALL, TICK, WINDOW, make_fonts)

# --- back-compat façade: the public API other modules and the tests already import from here -------
from .attention_view import draw_attention_drawer                       # noqa: F401
from .how_to_play import draw_how_to_play, how_to_play_height           # noqa: F401
from .labels import (ABILITY_MAY_LABEL, DECISION_LABELS,                # noqa: F401
                     compact_action as _compact_action)
from .modals import (ENGINE_PILLS, draw_card_preview,                   # noqa: F401
                     draw_settings_overlay)
from .widgets import (button as _button, card as _draw_card,            # noqa: F401
                      cross as _cross, text as _text, text_fit as _text_fit,
                      tick as _tick, tokens as _draw_tokens, wrap as _wrap)


class Frame(NamedTuple):
    """What render_frame returns so the app can hit-test every control."""
    buttons: List[Tuple["pygame.Rect", Action]]
    new_game: "pygame.Rect"
    reasoning_toggle: Optional["pygame.Rect"]
    hint_toggle: Optional["pygame.Rect"]
    review: Optional["pygame.Rect"]        # "Review game" button, shown only at game over
    settings: "pygame.Rect"                # opens the engine-settings modal
    scenario: "pygame.Rect"                # opens the scenario-setup screen
    attn_toggle: Optional["pygame.Rect"] = None   # "Attention" button (attention drawer); None if no ckpt
    # Right-click zoom targets: every face-up card on screen -> (rect, assets/ filename, upside_down).
    previews: Tuple[Tuple["pygame.Rect", str, bool], ...] = ()
    how_to: Optional["pygame.Rect"] = None        # "How to play" button (rules + card reference)


def render_frame(surface, view, fonts, legal_moves: List[Action], *,
                 hover: Optional[int] = None, status: str = "", log: Optional[List[str]] = None,
                 bot_result=None, show_reasoning: bool = True, seed=None,
                 hint_result=None, show_hint: bool = False, knowledge=None,
                 bot_eval=None, hint_eval=None, attn_available: bool = False,
                 mouse: Optional[Tuple[int, int]] = None,
                 bot_pending: bool = False, hint_pending: bool = False) -> Frame:
    """Paint the whole screen and hand back every clickable rect.

    ``mouse`` drives BOTH the chrome-button hover and the playable-card highlight. The latter self-disables
    on the bot's turn because ``legal_moves`` is empty then, so no card is ever flagged playable."""
    surface.fill(BG)
    small = fonts["small"]

    # --- the play area (board.py owns each zone; Paint collects the zoom targets + clickable cards) ---
    p = Paint(surface=surface, fonts=fonts, view=view, legal_moves=legal_moves, mouse=mouse)
    chrome = board.draw_play_area(p, seed=seed)

    # --- the side panel (side_panel.py) --------------------------------------------------------------
    side_panel.draw_background(surface)
    buttons = side_panel.draw_actions(surface, fonts, view, legal_moves, hover, status)
    buttons += p.buttons                     # a hand card / the king plays through the SAME click routing
    side_panel.draw_log(surface, fonts, log)
    reasoning_toggle, hint_toggle = side_panel.draw_reads(
        surface, fonts, view, bot_result=bot_result, show_reasoning=show_reasoning, bot_eval=bot_eval,
        hint_result=hint_result, show_hint=show_hint, hint_eval=hint_eval, mouse=mouse,
        bot_pending=bot_pending, hint_pending=hint_pending)
    side_panel.draw_knowledge(surface, fonts, view, knowledge)

    settings = widgets.button(surface, small, pygame.Rect(SETTINGS_X, 12, 84, 24), "Settings", mouse)
    analysis = pygame.Rect(ATTN_X, 12, 92, 24)          # attention-drawer toggle (left of Settings)
    if attn_available:
        widgets.button(surface, small, analysis, "Attention", mouse)
    else:                                               # no checkpoint -> disabled, never highlights
        pygame.draw.rect(surface, DIVIDER, analysis, border_radius=4)
        widgets.text(surface, small, "Attention", (analysis.x + 10, analysis.y + 4), MUTE)

    return Frame(buttons, chrome["new_game"], reasoning_toggle, hint_toggle, chrome["review"], settings,
                 chrome["scenario"], attn_toggle=(analysis if attn_available else None),
                 previews=tuple(p.previews), how_to=chrome["how_to"])
