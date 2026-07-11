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

WINDOW = (1740, 1060)
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

# Per-card-name colors (from the card art), shared by the tree view and the hand-knowledge column.
CARD_COLORS = {
    "Queen": (245, 87, 14), "Princess": (242, 92, 15), "Sentry": (224, 127, 59),
    "KingsHand": (199, 103, 43), "Warlord": (240, 154, 9), "Mystic": (238, 154, 6),
    "Oathbound": (226, 202, 32), "Soldier": (80, 172, 123), "Judge": (64, 145, 69),
    "Inquisitor": (76, 155, 168), "Zealot": (122, 156, 194), "Elder": (106, 154, 220),
    "Assassin": (168, 171, 224), "Fool": (167, 113, 175),
}
NEUTRAL = (80, 84, 94)      # non-card items (abilities, unknowns)

PANEL_X = 1290              # side panel starts here (wider panel -> less PV wrapping in reasoning/hint)
KNOW_X = 1050               # hand-knowledge column occupies [KNOW_X, PANEL_X] (~240 wide)
PANEL_W = WINDOW[0] - PANEL_X - 12
ROW_MAX_X = KNOW_X - 12     # right edge of the play area (before the knowledge column)
DIVIDER = (60, 62, 70)

# The side panel stacks four sections: ACTIONS, LOG, (bot) REASONING, (your) HINT.
BTN_TOP = 88        # y of the first action button (kept in sync with app's hover hit-test)
BTN_H = 28
ACT_BOTTOM = 490    # action buttons capped above this -> fits all 14 guess names (88 + 14*28 = 480)
LOG_TOP = 505       # "Log" section header
LOG_LINES = 7       # recent log lines shown
REASON_TOP = 690    # "Bot reasoning" section header (+ toggle)
HINT_TOP = 875      # "Your hint" section header (+ toggle)


class Frame(NamedTuple):
    """What render_frame returns so the app can hit-test every control."""
    buttons: List[Tuple["pygame.Rect", Action]]
    new_game: "pygame.Rect"
    reasoning_toggle: Optional["pygame.Rect"]
    hint_toggle: Optional["pygame.Rect"]
    review: Optional["pygame.Rect"]        # "Review game" button, shown only at game over
    settings: "pygame.Rect"                # opens the engine-settings modal
    scenario: "pygame.Rect"                # opens the scenario-setup screen
    attn_toggle: Optional["pygame.Rect"] = None   # "Analysis" button (attention drawer); None if no ckpt

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

# The flattened abilities declare their parameter at ABILITY_MAY, so give them a clearer header.
_ABILITY_MAY_LABEL = {
    cards.Ability.MYSTIC: "Mystic: pick a value (or decline)",
    cards.Ability.INQUISITOR: "Interrogate: name a card (or decline)",
    cards.Ability.FOOL: "Fool: take a stack card (or decline)",
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


def _draw_card(surf, image, pos, *, highlight=False, dim=False, size=CARD):
    rect = pygame.Rect(pos, size)                      # border matches the image size (pass SMALL for minis)
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


_CARD_PREFIX = {ActionKind.PLAY_CARD: "", ActionKind.HIDE_CARD: "hide ",
                ActionKind.DISCARD_CARD: "discard ", ActionKind.CHOOSE_HAND_CARD: "give "}


def _compact_action(action: Action) -> str:
    """A short action label for the narrow reasoning panel (drops the play_card()/#id noise).
    Card actions keep their kind (hide/discard/give) so the setup phase reads correctly."""
    k = action.kind
    if k in _CARD_PREFIX:
        cdef = cards.card_def(action.card)
        return f"{_CARD_PREFIX[k]}{cdef.name}({cdef.value})"
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


def _draw_explain(surface, fonts, result, top: int, depth: int = 5, own_eval=None, seat=None):
    """Render the top-2 principal-variation lines for a search ``result`` at ``top`` (chess-engine
    style: [eval] then move labels colored by the player who moved). When ``own_eval``/``seat`` are given
    (the panel's own seat and its own-perspective value), draw a prominent ``P{seat} sees +X.XX`` line
    first -- correct even when that seat isn't the mover. Header/toggle drawn by caller."""
    small = fonts["small"]
    x = PANEL_X + 12
    max_x = WINDOW[0] - 12
    _text(surface, small, f"{result.iterations} sims, {result.elapsed:.2f}s", (x, top), MUTE)
    _text(surface, small, "P0", (x + 150, top), P_COLORS[0])
    _text(surface, small, "P1", (x + 178, top), P_COLORS[1])

    y = top + 22
    if own_eval is not None and seat is not None:
        _text(surface, small, f"P{seat} sees {own_eval:+.2f}", (x, y), P_COLORS.get(seat, INK))
        y += 20
    lines = result.principal_variations(top=2, depth=depth)
    if not lines:
        _text(surface, small, "(no lines)", (x, y), MUTE)
        return
    y += 2
    for line in lines:
        tokens = [(f"[{line[0].mean_q:+.2f}]", INK)]
        tokens += [(_compact_action(step.move), P_COLORS.get(step.player, INK)) for step in line]
        y = _draw_tokens(surface, small, tokens, x, y, max_x, 19)
        y += 4   # gap between lines


def _draw_reasoning_section(surface, fonts, top, title, result, shown, placeholder,
                            own_eval=None, seat=None):
    """Header + [hide]/[show] toggle at ``top``; render the PV lines when ``shown``. Returns the toggle
    Rect. Shared by the bot-reasoning and human-hint panels."""
    small = fonts["small"]
    px = PANEL_X + 12
    pygame.draw.line(surface, DIVIDER, (PANEL_X + 8, top - 12), (WINDOW[0] - 8, top - 12))
    _text(surface, small, title, (px, top), GOLD)
    toggle = pygame.Rect(WINDOW[0] - 12 - 64, top - 3, 64, 22)
    pygame.draw.rect(surface, BTN, toggle, border_radius=4)
    _text(surface, small, "[hide]" if shown else "[show]", (toggle.x + 7, toggle.y + 3))
    if shown:
        if result is not None:
            _draw_explain(surface, fonts, result, top + 26, own_eval=own_eval, seat=seat)
        else:
            _text(surface, small, placeholder, (px, top + 26), MUTE)
    return toggle


_TICK = (90, 200, 110)
_AMBER = (224, 150, 60)


def _tick(surface, x, y):     # small green checkmark (consolas lacks the glyph, so draw it)
    pygame.draw.lines(surface, _TICK, False, [(x + 1, y + 8), (x + 5, y + 12), (x + 13, y + 1)], 2)


def _cross(surface, x, y):    # small red X
    pygame.draw.line(surface, RED, (x + 2, y + 2), (x + 12, y + 12), 2)
    pygame.draw.line(surface, RED, (x + 12, y + 2), (x + 2, y + 12), 2)


def _draw_know_panel(surface, fonts, x, y, max_x, title, facts):
    """One knower's read on the other's hand: title + PERFECT/50-50 chip + a tick row and a cross row."""
    small = fonts["small"]
    has, lacks, level = facts
    _text(surface, small, title, (x, y), MUTE)
    yy = y + 20
    if level:
        txt = "PERFECT INFO" if level == "perfect" else "50-50"
        chip = pygame.Rect(x, yy, small.size(txt)[0] + 12, 20)
        pygame.draw.rect(surface, GOLD if level == "perfect" else _AMBER, chip, border_radius=4)
        surface.blit(small.render(txt, True, (20, 20, 20)), (x + 6, yy + 2))
        yy += 26
    _tick(surface, x, yy + 2)
    pos = [(n, CARD_COLORS.get(n, INK)) for n in sorted(has)] or [("(none)", MUTE)]
    yy = _draw_tokens(surface, small, pos, x + 20, yy, max_x, 20)
    _cross(surface, x, yy + 4)
    neg = [(n, CARD_COLORS.get(n, INK)) for n in sorted(lacks)] or [("(none)", MUTE)]
    _draw_tokens(surface, small, neg, x + 20, yy + 2, max_x, 20)


def _draw_knowledge(surface, fonts, view, knowledge):
    """Hand-knowledge column between the play area and the side panel: what each player has deduced
    about the other's hand (all from public events, so both reads are legitimately shown)."""
    if knowledge is None:
        return
    pygame.draw.line(surface, DIVIDER, (KNOW_X, 0), (KNOW_X, WINDOW[1]))
    x0, max_x = KNOW_X + 10, PANEL_X - 10
    _text(surface, fonts["med"], "Hand knowledge", (x0, 12), INK)
    obs, opp = view.observer, 1 - view.observer
    # Each knower's read sits on that player's side: opponent (P{opp}) at top, you (P{obs}) at bottom.
    _draw_know_panel(surface, fonts, x0, 64, max_x, f"P{opp} knows — your hand", knowledge[opp])
    _draw_know_panel(surface, fonts, x0, 804, max_x, f"You (P{obs}) know — P{opp}'s hand", knowledge[obs])


def render_frame(surface, view, fonts, legal_moves: List[Action], *,
                 hover: Optional[int] = None, status: str = "", log: Optional[List[str]] = None,
                 bot_result=None, show_reasoning: bool = True, seed=None,
                 hint_result=None, show_hint: bool = False, knowledge=None,
                 bot_eval=None, hint_eval=None, attn_available: bool = False) -> Frame:
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
    scenario = pygame.Rect(new_game.x - 12 - 100, 12, 100, 30)      # build a custom position (S key)
    pygame.draw.rect(surface, BTN, scenario, border_radius=4)
    _text(surface, small, "Scenario", (scenario.x + 14, scenario.y + 7))
    # "Review game" appears only at game over (no pending decision -> terminal).
    review = None
    left_anchor = scenario.x
    if not view.pending:
        review = pygame.Rect(scenario.x - 12 - 128, 12, 128, 30)
        pygame.draw.rect(surface, BTN_HOVER, review, border_radius=4)
        _text(surface, small, "Review game", (review.x + 14, review.y + 7))
        left_anchor = review.x
    if seed is not None:
        seed_s = f"seed {seed}"
        _text(surface, small, seed_s, (left_anchor - 12 - small.size(seed_s)[0], new_game.y + 7), MUTE)

    # --- antechambers -------------------------------------------------------------------
    y = 150
    for seat, ante in enumerate(view.antechambers):
        if ante:
            _text(surface, small, f"antechamber[{seat}] (ascends next turn):", (24, y))
            for c, x in zip(ante, _row_x(360, len(ante), 70, SMALL[0])):
                _draw_card(surface, assets.card_surface(c, SMALL), (x, y - 8))
            y += 84

    # --- face-up leftover (known to both from the start; info only, NOT the leading card) -----
    # Sits in the free band above the stack (which fills top-y 466 rightward on a dense stack) and left
    # of the knowledge column; its border is SMALL-sized so it no longer overhangs past KNOW_X.
    if view.leftover_faceup is not None and 0 <= view.leftover_faceup < cards.DECK_SIZE:
        lx = ROW_MAX_X - SMALL[0] - 16
        _text(surface, small, "leftover (face-up, info):", (lx - 130, 342), MUTE)
        _draw_card(surface, assets.card_surface(view.leftover_faceup, SMALL), (lx, 360), size=SMALL)

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
    if kind == StepKind.ABILITY_MAY and view.pending[-1].source is not None:
        decision = _ABILITY_MAY_LABEL.get(cards.card_ability(view.pending[-1].source), decision)
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
    for i, line in enumerate((log or [])[-LOG_LINES:]):
        _text(surface, small, line, (px, LOG_TOP + 22 + i * 20), INK)

    # REASONING (bot) + HINT (you) -- two PV sections, each with its own show/hide toggle.
    reasoning_toggle = _draw_reasoning_section(surface, fonts, REASON_TOP, f"Bot P{opp} read (MCTS):",
                                               bot_result, show_reasoning, "(no search yet)",
                                               own_eval=bot_eval, seat=opp)
    hint_toggle = _draw_reasoning_section(surface, fonts, HINT_TOP, f"Your P{view.observer} read (MCTS):",
                                          hint_result, show_hint, "(toggle for your read of this position)",
                                          own_eval=hint_eval, seat=view.observer)
    _draw_knowledge(surface, fonts, view, knowledge)

    settings = pygame.Rect(WINDOW[0] - 12 - 84, 12, 84, 24)   # engine-settings button (panel top-right)
    pygame.draw.rect(surface, BTN, settings, border_radius=4)
    _text(surface, small, "Settings", (settings.x + 10, settings.y + 4))
    analysis = pygame.Rect(settings.x - 8 - 92, 12, 92, 24)   # attention-drawer toggle (left of Settings)
    pygame.draw.rect(surface, BTN if attn_available else DIVIDER, analysis, border_radius=4)
    _text(surface, small, "Analysis", (analysis.x + 12, analysis.y + 4), INK if attn_available else MUTE)
    return Frame(buttons, new_game, reasoning_toggle, hint_toggle, review, settings, scenario,
                 attn_toggle=(analysis if attn_available else None))


# The engine (bot + analysis) modes, in pill order: (mode key, label). ``nn`` = NN-MCTS (hybrid-only).
ENGINE_PILLS = [("mcts", "Fixed N"), ("branching", "Branch"), ("hybrid", "Hybrid"), ("nn", "NN+MCTS")]


_SLIDER_RANGES = {"N": (25, 1024), "k": (10, 100), "l": (1, 8)}


def draw_settings_overlay(surface, fonts, engine, mouse, nn_available=True):
    """Draw the engine-settings modal over the board and return its clickable controls:
    ``{"pills": {mode: rect}, "sliders": [(track_rect, lo, hi, key), ...], "close": rect}``.

    ``engine`` = ``{"mode", "N", "k", "l"}``. Fixed mode shows one ``N`` slider; branch/hybrid/nn show ``k``
    and ``l`` (l = effective legal-moves for a sub-decision card at selection). ``nn`` (NN+MCTS) is
    hybrid-only; when ``nn_available`` is False that pill is drawn disabled and its click is ignored."""
    med, small = fonts["med"], fonts["small"]
    W, H = WINDOW
    dim = pygame.Surface((W, H), pygame.SRCALPHA)
    dim.fill((0, 0, 0, 160))
    surface.blit(dim, (0, 0))
    is_fixed = engine["mode"] == "mcts"
    bw, bh = 680, (250 if is_fixed else 320)
    bx, by = (W - bw) // 2, (H - bh) // 2
    pygame.draw.rect(surface, PANEL, (bx, by, bw, bh), border_radius=8)
    pygame.draw.rect(surface, GOLD, (bx, by, bw, bh), 2, border_radius=8)
    _text(surface, med, "Engine settings  (bot + hint)", (bx + 20, by + 16), INK)

    n = len(ENGINE_PILLS)
    pills, pw = {}, (bw - 40 - 8 * (n - 1)) // n
    x = bx + 20
    for mode, label in ENGINE_PILLS:
        r = pygame.Rect(x, by + 56, pw, 36)
        disabled = (mode == "nn") and not nn_available
        sel = engine["mode"] == mode
        fill = MUTE if disabled else (GOLD if sel else BTN)
        pygame.draw.rect(surface, fill, r, border_radius=18)
        tw = small.size(label)[0]
        _text(surface, small, label, (r.centerx - tw // 2, r.y + 9), (20, 20, 20) if sel else INK)
        pills[mode] = r
        x += pw + 8

    def slider_row(sy, key):
        lo, hi = _SLIDER_RANGES[key]
        val = engine[key]
        _text(surface, small, f"{key} = {val}", (bx + 20, sy - 24), GOLD)
        rng = f"[{lo} .. {hi}]"
        _text(surface, small, rng, (bx + bw - 20 - small.size(rng)[0], sy - 24), MUTE)
        track = pygame.Rect(bx + 20, sy, bw - 40, 8)
        pygame.draw.rect(surface, BTN, track, border_radius=4)
        kx = int(track.x + (val - lo) / (hi - lo) * track.w)
        pygame.draw.circle(surface, GOLD, (kx, track.centery), 10)
        return (track, lo, hi, key)

    if is_fixed:
        sliders = [slider_row(by + 140, "N")]
        preview = f"~ {engine['N']} simulations / decision"
    else:
        sliders = [slider_row(by + 130, "k"), slider_row(by + 196, "l")]
        if engine["mode"] == "nn":
            preview = "NN eval-head + hybrid clamp(k * eff_n(l) * (1 + opp_cards), 64, 4096)"
        elif engine["mode"] == "hybrid":
            preview = "clamp(k * eff_n(l) * (1 + opp_cards), 64, 4096)"
        else:
            preview = "clamp(k * eff_n(l), 64, 4096)"
    _text(surface, small, preview, (bx + 20, by + bh - 72), MUTE)

    close = pygame.Rect(bx + bw - 20 - 78, by + bh - 44, 78, 28)
    pygame.draw.rect(surface, BTN_HOVER if close.collidepoint(mouse) else BTN, close, border_radius=4)
    _text(surface, small, "Close", (close.x + 18, close.y + 5), INK)
    return {"pills": pills, "sliders": sliders, "close": close}


def draw_attention_drawer(surface, fonts, entries, mouse, *, mode="absolute", hover=None,
                          selected=0, hide_board=False, result=None, depth=5,
                          seat_labels=None, seat_selected=0):
    """Right-side "analysis mode" drawer over a dim scrim. ``entries`` = the top recommendations as
    ``[(move, payload), ...]`` (payload = an AttentionExplanation); ``selected`` picks which entry the
    heatmap shows (clickable rec pills switch). ``hide_board`` drops the board token from the heatmap and
    renormalizes the remaining attention rows. In "signed" mode the bottom shows per-entry Top-contributor
    columns (the side-by-side comparison); otherwise the PV. ``seat_labels`` (e.g. ("P0","P1"), used by the
    review screen) draws clickable seat pills selecting whose read is explained. Returns clickable controls
    ``{"close", "mode_toggle", "board_toggle", "rec_pills", "seat_pills", "hits"}``."""
    from . import attention_view                          # lazy: attention_view imports palette from here
    med, small = fonts["med"], fonts["small"]
    W, H = WINDOW
    dim = pygame.Surface((W, H), pygame.SRCALPHA)
    dim.fill((0, 0, 0, 140))
    surface.blit(dim, (0, 0))
    dw = int(0.46 * W)
    dx = W - dw
    pygame.draw.rect(surface, PANEL, (dx, 0, dw, H))
    pygame.draw.rect(surface, GOLD, (dx, 0, dw, H), 2)
    pad = 16

    _text(surface, med, "Attention  -  why this move", (dx + pad, 14), INK)
    close = pygame.Rect(dx + dw - pad - 82, 12, 82, 26)
    pygame.draw.rect(surface, BTN_HOVER if close.collidepoint(mouse) else BTN, close, border_radius=4)
    _text(surface, small, "Close [A]", (close.x + 10, close.y + 5), INK)

    # Rec pills: the search's top recommendations; click switches which one the heatmap explains.
    selected = max(0, min(selected, len(entries) - 1))
    rec_pills = []
    xx = dx + pad
    for i, (mv, pl) in enumerate(entries):
        label = f"{i + 1}. {_compact_action(mv)}   q = {pl.q:+.2f}"
        wpx = small.size(label)[0] + 20
        r = pygame.Rect(xx, 44, wpx, 26)
        pygame.draw.rect(surface, BTN_HOVER if r.collidepoint(mouse) else BTN, r, border_radius=13)
        if i == selected:
            pygame.draw.rect(surface, GOLD, r, 2, border_radius=13)
        _text(surface, small, label, (r.x + 10, r.y + 5), GOLD if i == selected else INK)
        rec_pills.append(r)
        xx += wpx + 10

    mode_toggle = pygame.Rect(dx + pad, 78, 190, 24)
    pygame.draw.rect(surface, BTN_HOVER if mode_toggle.collidepoint(mouse) else BTN,
                     mode_toggle, border_radius=12)
    _mode_label = {"absolute": "absolute", "row_norm": "row-norm", "signed": "signed (Δq)"}.get(mode, mode)
    _text(surface, small, f"scale: {_mode_label}", (mode_toggle.x + 10, mode_toggle.y + 4))
    board_toggle = pygame.Rect(mode_toggle.right + 10, 78, 170, 24)
    pygame.draw.rect(surface, BTN_HOVER if board_toggle.collidepoint(mouse) else BTN,
                     board_toggle, border_radius=12)
    _text(surface, small, f"board: {'hidden' if hide_board else 'shown'}",
          (board_toggle.x + 10, board_toggle.y + 4))
    seat_pills = []
    if seat_labels:                                       # review: whose read is being explained
        sx = board_toggle.right + 10
        for i, lab in enumerate(seat_labels):
            r = pygame.Rect(sx, 78, small.size(lab)[0] + 20, 24)
            pygame.draw.rect(surface, BTN_HOVER if r.collidepoint(mouse) else BTN, r, border_radius=12)
            if i == seat_selected:
                pygame.draw.rect(surface, GOLD, r, 2, border_radius=12)
            _text(surface, small, lab, (r.x + 10, r.y + 4), GOLD if i == seat_selected else INK)
            seat_pills.append(r)
            sx = r.right + 6

    move, payload = entries[selected]
    exclude = ()
    if hide_board and "board" in payload.seq_labels:
        exclude = (payload.seq_labels.index("board"),)

    pv_h = 128
    heat_rect = (dx + pad, 112, dw - 2 * pad, H - 112 - pv_h)
    hits = attention_view.draw_attention(surface, fonts, payload, heat_rect, mode=mode,
                                         emphasize_rows=(0,),
                                         candidate_index=payload.candidate_seq_index, hover=hover,
                                         exclude_indices=exclude)
    if hover is not None:
        hit = next((h for h in hits if (h.i, h.j, h.head) == hover), None)
        if hit is not None:
            attention_view.draw_tooltip(surface, fonts, payload, hit, mouse)

    pv_y = H - pv_h + 6
    pygame.draw.line(surface, DIVIDER, (dx + pad, pv_y - 6), (dx + dw - pad, pv_y - 6))
    if mode == "signed" and any(getattr(pl, "attribution", None) is not None for _, pl in entries):
        # per-entry Top-contributor columns: which parts of the position drove each candidate's value
        _text(surface, small, "Top contributors (Δq to q-logit):", (dx + pad, pv_y), MUTE)
        col_w = (dw - 2 * pad) // max(1, len(entries))
        for i, (mv, pl) in enumerate(entries):
            cx = dx + pad + i * col_w
            _text(surface, small, f"{i + 1}. {_compact_action(mv)}", (cx, pv_y + 20),
                  GOLD if i == selected else INK)
            att = getattr(pl, "attribution", None)
            if att is None:
                continue
            order = sorted(range(len(att)), key=lambda k: -abs(float(att[k])))
            for row, k in enumerate(order[:4]):
                val = float(att[k])
                col = (74, 190, 110) if val >= 0 else (214, 72, 72)
                _text(surface, small, f"{pl.seq_labels[k]} {val:+.2f}", (cx, pv_y + 40 + row * 18), col)
    else:
        _text(surface, small, "Principal variation:", (dx + pad, pv_y), MUTE)
        if result is not None and getattr(result, "root", None) is not None:
            try:
                pvs = result.principal_variations(top=2, depth=depth)
            except Exception:                            # noqa: BLE001 -- PV is best-effort decoration
                pvs = []
            yy = pv_y + 22
            for line in pvs:
                xx = dx + pad
                for step in line:
                    t = small.render(_compact_action(step.move), True, P_COLORS.get(step.player, INK))
                    if xx + t.get_width() > dx + dw - pad:
                        break
                    surface.blit(t, (xx, yy))
                    xx += t.get_width() + 8
                yy += 20
    return {"close": close, "mode_toggle": mode_toggle, "board_toggle": board_toggle,
            "rec_pills": rec_pills, "seat_pills": seat_pills, "hits": hits}


def make_fonts():
    pygame.font.init()
    return {
        "big": pygame.font.SysFont("consolas,arial", 30),
        "med": pygame.font.SysFont("consolas,arial", 20),
        "small": pygame.font.SysFont("consolas,arial", 16),
    }
