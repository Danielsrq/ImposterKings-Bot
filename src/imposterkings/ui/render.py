"""Render an InformationSet to a PyGame surface and return clickable action buttons.

Pure drawing: given a surface, the observer's view, and the legal actions, it paints every zone
(opponent backs, stack with leading/disgraced state, antechambers, kings, your hand) and a right-hand
panel of action buttons, returning ``[(rect, action), ...]`` for the app loop to hit-test.
"""
from __future__ import annotations

from typing import List, NamedTuple, Optional, Tuple

import pygame

from .. import cards
from ..actions import Action, ActionKind, StepKind
from ..explain import format_action
from . import assets

WINDOW = (1380, 1000)
CARD = (96, 131)
SMALL = (64, 87)

BG = (18, 64, 48)
PANEL = (28, 30, 36)
INK = (235, 235, 235)
MUTE = (150, 150, 160)
GOLD = (235, 200, 90)
RED = (200, 70, 70)
BTN = (52, 56, 66)
BTN_HOVER = (78, 96, 120)
P_COLORS = {0: (95, 160, 240), 1: (240, 170, 90)}   # PV move colors by seat (P0 blue, P1 orange)

PANEL_X = 1030
PANEL_W = WINDOW[0] - PANEL_X - 12
ROW_MAX_X = PANEL_X - 12     # right edge of the play area (cards never cross into the panel)
DIVIDER = (60, 62, 70)

# The side panel is three stacked sections: ACTIONS, LOG, REASONING.
BTN_TOP = 88        # y of the first action button (kept in sync with app's hover hit-test)
BTN_H = 28
ACT_BOTTOM = 520    # action buttons capped above this -> ~15 rows, fits all 14 guess names
LOG_TOP = 540       # "Log" section header
REASON_TOP = 740    # "Bot reasoning" section header (+ toggle)


class Frame(NamedTuple):
    """What render_frame returns so the app can hit-test every control."""
    buttons: List[Tuple["pygame.Rect", Action]]
    new_game: "pygame.Rect"
    reasoning_toggle: Optional["pygame.Rect"]

# Friendly labels for the decision header (the raw StepKind names are long/cryptic).
DECISION_LABELS = {
    StepKind.SETUP_HIDE: "Hide a card", StepKind.SETUP_DISCARD: "Discard a card",
    StepKind.MAIN: "Your turn", StepKind.ABILITY_MAY: "Use ability?",
    StepKind.ABILITY_CHOICE: "Take the effect?", StepKind.ABILITY_GUESS: "Name a card",
    StepKind.ABILITY_NUMBER: "Pick a value (1-8)", StepKind.ABILITY_HAND_CARD: "Choose a hand card",
    StepKind.ABILITY_STACK_TARGET: "Choose a stack card", StepKind.ABILITY_SWAP_RESPOND: "Card to swap",
    StepKind.OATHBOUND_SECOND: "Play a follow-up", StepKind.REACTION_KINGSHAND: "King's Hand?",
    StepKind.REACTION_ASSASSIN: "Assassin?", StepKind.REACTION_KH_VS_ASSASSIN: "King's Hand vs Assassin?",
}


def _text(surf, font, s, pos, color=INK):
    surf.blit(font.render(s, True, color), pos)


def _row_x(x0: int, count: int, gap: int, card_w: int, x_max: int = ROW_MAX_X) -> List[int]:
    """X positions for a row of ``count`` cards starting at ``x0``, shrinking the gap (cards may
    overlap) so the row always fits within ``x_max`` -- the rightmost card never gets clipped."""
    if count <= 1:
        return [x0]
    span = x_max - x0 - card_w
    g = min(gap, max(12, span / (count - 1)))
    return [int(x0 + i * g) for i in range(count)]


def _draw_card(surf, image, pos, *, highlight=False, dim=False):
    rect = pygame.Rect(pos, CARD)
    if dim:
        image = image.copy()
        image.fill((90, 90, 90), special_flags=pygame.BLEND_RGB_MULT)
    surf.blit(image, pos)
    pygame.draw.rect(surf, GOLD if highlight else (10, 10, 10), rect, 3 if highlight else 1)
    return rect


_REACTION_KINDS = (StepKind.REACTION_KINGSHAND, StepKind.REACTION_ASSASSIN,
                   StepKind.REACTION_KH_VS_ASSASSIN)


def _reaction_context(view) -> str:
    """A human-readable note about what a reaction window is reacting to (reaction steps only)."""
    if not view.pending:
        return ""
    step = view.pending[-1]
    if step.kind not in _REACTION_KINDS or step.source is None:
        return ""
    src = cards.card_name(step.source)
    if step.guess is not None:
        return f"Counter {src}? (guessed {step.guess})"
    return f"Counter opponent's {src}?"


_SHORT_ACTION = {
    ActionKind.DECLARE_ABILITY: "declare", ActionKind.DECLINE_ABILITY: "decline",
    ActionKind.FLIP_KING: "flip-king", ActionKind.STOP: "stop",
    ActionKind.REVEAL_KINGSHAND: "KingsHand!", ActionKind.REVEAL_ASSASSIN: "Assassin!",
    ActionKind.DECLINE_REACTION: "no-react",
}


def _compact_action(action: Action) -> str:
    """A short action label for the narrow reasoning panel (drops the play_card()/#id noise)."""
    k = action.kind
    if k in (ActionKind.PLAY_CARD, ActionKind.HIDE_CARD, ActionKind.DISCARD_CARD,
             ActionKind.CHOOSE_HAND_CARD):
        cdef = cards.card_def(action.card)
        return f"{cdef.name}({cdef.value})"
    if k == ActionKind.GUESS_CARD:
        return f"guess {action.name}"
    if k == ActionKind.CHOOSE_NUMBER:
        return f"mute {action.number}"
    if k == ActionKind.CHOOSE_STACK_TARGET:
        return f"target@{action.target}"
    return _SHORT_ACTION.get(k, k.name.lower())


def _draw_tokens(surface, font, tokens, x0: int, y: int, max_x: int, line_h: int, indent: int = 14) -> int:
    """Draw ``(text, color)`` tokens left-to-right, wrapping (with a small indent) at ``max_x``.
    Returns the y just below the block."""
    space = font.size(" ")[0]
    x = x0
    for text, color in tokens:
        w = font.size(text)[0]
        if x > x0 and x + w > max_x:           # wrap (but always draw >=1 token per row)
            y += line_h
            x = x0 + indent
        surface.blit(font.render(text, True, color), (x, y))
        x += w + space
    return y + line_h


def _draw_explain(surface, fonts, bot_result, top: int):
    """Render the top-2 principal-variation lines for the bot's last decision at ``top`` (chess-engine
    style: [eval] then move labels colored by the player who moved). Header/toggle drawn by caller."""
    small = fonts["small"]
    x = PANEL_X + 12
    max_x = WINDOW[0] - 12
    _text(surface, small, f"{bot_result.iterations} sims, {bot_result.elapsed:.2f}s", (x, top), MUTE)
    _text(surface, small, "P0", (x + 150, top), P_COLORS[0])
    _text(surface, small, "P1", (x + 178, top), P_COLORS[1])

    lines = bot_result.principal_variations(top=2, depth=6)
    if not lines:
        _text(surface, small, "(no lines)", (x, top + 22), MUTE)
        return
    y = top + 24
    for line in lines:
        tokens = [(f"[{line[0].mean_q:+.2f}]", INK)]
        tokens += [(_compact_action(step.move), P_COLORS.get(step.player, INK)) for step in line]
        y = _draw_tokens(surface, small, tokens, x, y, max_x, 19)
        y += 4   # gap between lines


def render_frame(surface, view, fonts, legal_moves: List[Action], *,
                 hover: Optional[int] = None, status: str = "", log: Optional[List[str]] = None,
                 bot_result=None, show_reasoning: bool = True, seed=None) -> Frame:
    surface.fill(BG)
    big, med, small = fonts["big"], fonts["med"], fonts["small"]
    opp = 1 - view.observer

    # --- opponent (top): face-down backs + info -----------------------------------------
    _text(surface, med, f"Opponent (seat {opp})  -  {view.opp_hand_count} cards   "
                        f"king: {'USED' if view.kings[opp] else 'up'}"
                        f"{'   (has hidden)' if view.opp_has_hidden else ''}", (24, 16))
    back = assets.back_surface(SMALL)
    for i, x in enumerate(_row_x(24, view.opp_hand_count, 34, SMALL[0])):
        surface.blit(back, (x, 46))

    # --- New Game button (top-right of the play area, left of the panel) -----------------
    new_game = pygame.Rect(ROW_MAX_X - 120, 12, 120, 30)
    pygame.draw.rect(surface, BTN, new_game, border_radius=4)
    _text(surface, small, "New Game", (new_game.x + 16, new_game.y + 7))
    if seed is not None:
        seed_s = f"seed {seed}"
        _text(surface, small, seed_s, (new_game.x - 12 - small.size(seed_s)[0], new_game.y + 7), MUTE)

    # --- antechambers -------------------------------------------------------------------
    y = 150
    for seat, ante in enumerate(view.antechambers):
        if ante:
            _text(surface, small, f"antechamber[{seat}] (ascends next turn):", (24, y))
            for c, x in zip(ante, _row_x(360, len(ante), 70, SMALL[0])):
                _draw_card(surface, assets.card_surface(c, SMALL), (x, y - 8))
            y += 84

    # --- stack (center) -----------------------------------------------------------------
    _text(surface, med, "Throne / stack:", (24, 430))
    if not view.stack:
        _text(surface, small, "(empty)", (200, 432), MUTE)
    from ..explain import _stack_value
    xs = _row_x(24, len(view.stack), 70, CARD[0])
    for i, (sc, x) in enumerate(zip(view.stack, xs)):
        lead = (i == len(view.stack) - 1)
        img = assets.card_surface(sc.card, CARD)
        _draw_card(surface, img, (x, 466), highlight=lead, dim=sc.disgraced)
        # Compact value label (the card art already shows the name); avoids overlap on dense stacks.
        val = _stack_value(view, sc)
        label = "x0" if sc.disgraced else f"={val}"
        _text(surface, small, label, (x + 4, 466 + CARD[1] + 2), RED if sc.disgraced else GOLD)
        if lead:
            _text(surface, small, "lead", (x + 4, 450), GOLD)

    # --- your hand (bottom) -------------------------------------------------------------
    _text(surface, med, f"Your hand (seat {view.observer})   "
                        f"king: {'USED' if view.kings[view.observer] else 'up'}"
                        f"{'   hidden: ' + cards.card_name(view.own_hidden) if view.own_hidden is not None else ''}",
          (24, 800))
    for c, x in zip(view.own_hand, _row_x(24, len(view.own_hand), 104, CARD[0])):
        _draw_card(surface, assets.card_surface(c, CARD), (x, 832))
    if view.muted_values:
        _text(surface, small, f"muted values -> 3: {sorted(view.muted_values)}", (24, 976), MUTE)

    # --- side panel: ACTIONS / LOG / REASONING ------------------------------------------
    px = PANEL_X + 12
    pygame.draw.rect(surface, PANEL, pygame.Rect(PANEL_X, 0, WINDOW[0] - PANEL_X, WINDOW[1]))

    # ACTIONS
    kind = view.pending[-1].kind if view.pending else None
    decision = DECISION_LABELS.get(kind, "GAME OVER") if kind is not None else "GAME OVER"
    _text(surface, med, decision, (px, 14))
    if status:
        _text(surface, small, status, (px, 42), GOLD)
    ctx = _reaction_context(view)
    if ctx:
        _text(surface, small, ctx, (px, 60), INK)

    buttons: List[Tuple[pygame.Rect, Action]] = []
    max_rows = max(1, (ACT_BOTTOM - BTN_TOP) // BTN_H)
    for i, move in enumerate(legal_moves[:max_rows]):
        rect = pygame.Rect(px, BTN_TOP + i * BTN_H, PANEL_W, BTN_H - 4)
        pygame.draw.rect(surface, BTN_HOVER if hover == i else BTN, rect, border_radius=4)
        _text(surface, small, f"{i + 1}. {format_action(move, view)}", (rect.x + 8, rect.y + 5))
        buttons.append((rect, move))
    if len(legal_moves) > max_rows:
        _text(surface, small, f"... +{len(legal_moves) - max_rows} more (use CLI)",
              (px, BTN_TOP + max_rows * BTN_H), MUTE)

    # LOG
    pygame.draw.line(surface, DIVIDER, (PANEL_X + 8, LOG_TOP - 12), (WINDOW[0] - 8, LOG_TOP - 12))
    _text(surface, small, "Log:", (px, LOG_TOP), MUTE)
    for i, line in enumerate((log or [])[-8:]):
        _text(surface, small, line, (px, LOG_TOP + 22 + i * 20), INK)

    # REASONING (with show/hide toggle)
    pygame.draw.line(surface, DIVIDER, (PANEL_X + 8, REASON_TOP - 12), (WINDOW[0] - 8, REASON_TOP - 12))
    _text(surface, small, "Bot reasoning (MCTS):", (px, REASON_TOP), GOLD)
    reasoning_toggle = pygame.Rect(WINDOW[0] - 12 - 64, REASON_TOP - 3, 64, 22)
    pygame.draw.rect(surface, BTN, reasoning_toggle, border_radius=4)
    _text(surface, small, "[hide]" if show_reasoning else "[show]", (reasoning_toggle.x + 7, reasoning_toggle.y + 3))
    if show_reasoning:
        if bot_result is not None:
            _draw_explain(surface, fonts, bot_result, REASON_TOP + 26)
        else:
            _text(surface, small, "(no search yet)", (px, REASON_TOP + 26), MUTE)

    return Frame(buttons, new_game, reasoning_toggle)


def make_fonts():
    pygame.font.init()
    return {
        "big": pygame.font.SysFont("consolas,arial", 30),
        "med": pygame.font.SysFont("consolas,arial", 20),
        "small": pygame.font.SysFont("consolas,arial", 16),
    }
