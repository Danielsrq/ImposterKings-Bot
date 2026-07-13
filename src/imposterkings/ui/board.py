"""The play area -- one painter per zone of the game screen.

Each zone (opponent's backs, the two lives, the antechambers, the leftover, the throne, your hand) is now a
standalone function that can be read, moved or restyled on its own. They share a :class:`Paint` context
rather than threading `surface, fonts, view, mouse, legal_moves` plus two accumulator lists through every
signature.

The two accumulators are the only real coupling between zones:

* ``previews`` -- every FACE-UP card on screen becomes a right-click zoom target. A zone appends the rect it
  just drew, so the zoom can never point at art that is not there.
* ``buttons``  -- a hand card whose play is unambiguous, plus the king when it can be flipped. These are
  merged with the side panel's action rows so the app has ONE click-routing list.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pygame

from .. import cards
from ..actions import Action, ActionKind
from . import assets, widgets
from .layout import (HAND_Y, KING_X, KNOW_X, LEFTOVER_Y, LIFE_X, OPP_ANTE_Y, OPP_HAND_Y, OPP_LIFE_Y,
                     OWN_ANTE_Y, OWN_LIFE_Y, PANEL_X, ROW_MAX_X, ROW_MAX_X_CARDS, STACK_Y, row_x)
from .theme import BTN_HOVER, CARD, GOLD, KING, MUTE, RED, SMALL, WINDOW


@dataclass
class Paint:
    """Everything a zone painter needs, plus what it hands back."""
    surface: "pygame.Surface"
    fonts: dict
    view: object
    legal_moves: List[Action]
    mouse: Optional[Tuple[int, int]] = None
    previews: List[Tuple["pygame.Rect", str, bool]] = field(default_factory=list)  # right-click zoom targets
    buttons: List[Tuple["pygame.Rect", Action]] = field(default_factory=list)      # clickable cards / king

    @property
    def me(self) -> int:
        return self.view.observer

    @property
    def opp(self) -> int:
        return 1 - self.view.observer


def draw_opponent(p: Paint) -> None:
    """Their hand: face-down backs only -- we know the COUNT, never the cards."""
    v, med = p.view, p.fonts["med"]
    widgets.text(p.surface, med, f"Opponent (seat {p.opp})  -  {v.opp_hand_count} cards", (24, 16))
    back = assets.back_surface(SMALL)
    for x in row_x(24, v.opp_hand_count, 34, SMALL[0], ROW_MAX_X_CARDS):
        p.surface.blit(back, (x, OPP_HAND_Y))


def _life(surface, fonts, y: int, *, true_king: bool, flipped: bool, has_hidden: bool,
          hidden_card=None, label: str = "", highlight: bool = False):
    """One seat's life: the king (TrueKing = the player who started the game, King = the other) sits nearest
    the knowledge column, with the face-down hidden card to its LEFT. A FLIPPED (used) king is drawn as an
    upside-down card back -- the life is spent. A taken hidden card is simply absent.

    Returns ``(king_rect, king_asset)`` so the caller can make the king clickable (flip-king) and
    right-click previewable."""
    small = fonts["small"]
    asset = (cards.CARD_BACK_ASSET if flipped else
             (cards.TRUE_KING_ASSET if true_king else cards.KING_ASSET))
    if flipped:                                                   # spent life -> upside-down back
        img = pygame.transform.rotate(assets.back_surface(KING), 180)
    else:
        img = assets.king_surface(KING, true_king=true_king)
    rect = widgets.card(surface, img, (KING_X, y), size=KING, dim=flipped, highlight=highlight)
    if label:
        widgets.text(surface, small, label, (LIFE_X, y - 20), MUTE)
    if has_hidden:                                                # the set-aside card, still face-down
        hy = y + (KING[1] - SMALL[1]) // 2
        widgets.card(surface, assets.back_surface(SMALL), (LIFE_X, hy), size=SMALL)
        name = cards.card_name(hidden_card) if hidden_card is not None else "hidden"
        widgets.text(surface, small, name, (LIFE_X, hy + SMALL[1] + 2), MUTE)
    return rect, asset


def draw_lives(p: Paint) -> None:
    """A king per seat + its hidden card, in a strip left of the knowledge column. Your king becomes a
    BUTTON exactly when flipping it is legal."""
    v = p.view
    flip = next((m for m in p.legal_moves if m.kind == ActionKind.FLIP_KING), None)

    r, a = _life(p.surface, p.fonts, OPP_LIFE_Y, true_king=(p.opp == v.starting_player),
                 flipped=v.kings[p.opp], has_hidden=v.opp_has_hidden, label=f"P{p.opp} life")
    p.previews.append((r, a, v.kings[p.opp]))

    hot = (flip is not None and p.mouse is not None
           and pygame.Rect((KING_X, OWN_LIFE_Y), KING).collidepoint(p.mouse))
    r, a = _life(p.surface, p.fonts, OWN_LIFE_Y, true_king=(p.me == v.starting_player),
                 flipped=v.kings[p.me], has_hidden=v.own_hidden is not None,
                 hidden_card=v.own_hidden, label=f"P{p.me} life (you)", highlight=bool(hot))
    p.previews.append((r, a, v.kings[p.me]))
    if flip is not None:
        p.buttons.append((r, flip))


def draw_antechambers(p: Paint) -> None:
    """Each seat's queue on ITS OWN side of the throne: theirs above, yours down by your hand."""
    def one(seat: int, y: int) -> None:
        ante = p.view.antechambers[seat]
        if not ante:
            return
        widgets.text(p.surface, p.fonts["small"], f"antechamber[{seat}] (ascends next turn):", (24, y))
        for c, x in zip(ante, row_x(360, len(ante), 70, SMALL[0], ROW_MAX_X_CARDS)):
            r = widgets.card(p.surface, assets.card_surface(c, SMALL), (x, y - 8), size=SMALL)
            p.previews.append((r, cards.asset_path(c), False))

    one(p.opp, OPP_ANTE_Y)
    one(p.me, OWN_ANTE_Y)


def draw_leftover(p: Paint) -> None:
    """The face-up leftover: known to both from the start. Information only -- NOT the leading card."""
    v, small = p.view, p.fonts["small"]
    if v.leftover_faceup is None or not (0 <= v.leftover_faceup < cards.DECK_SIZE):
        return
    lx = ROW_MAX_X - SMALL[0] - 16
    widgets.text(p.surface, small, "leftover (face-up, info):", (lx - 130, 342), MUTE)
    r = widgets.card(p.surface, assets.card_surface(v.leftover_faceup, SMALL), (lx, LEFTOVER_Y), size=SMALL)
    p.previews.append((r, cards.asset_path(v.leftover_faceup), False))


def draw_stack(p: Paint) -> None:
    """The throne. The top card LEADS (gold border); a disgraced card is dimmed and worth 0."""
    from ..explain import _stack_value
    v, med, small = p.view, p.fonts["med"], p.fonts["small"]
    widgets.text(p.surface, med, "Throne / stack:", (24, 430))
    if not v.stack:
        widgets.text(p.surface, small, "(empty)", (200, 432), MUTE)
    for i, (sc, x) in enumerate(zip(v.stack, row_x(24, len(v.stack), 70, CARD[0], ROW_MAX_X_CARDS))):
        lead = (i == len(v.stack) - 1)
        r = widgets.card(p.surface, assets.card_surface(sc.card, CARD), (x, STACK_Y),
                         highlight=lead, dim=sc.disgraced)
        p.previews.append((r, cards.asset_path(sc.card), False))
        # Compact value label (the card art already shows the name); avoids overlap on dense stacks.
        label = "x0" if sc.disgraced else f"={_stack_value(v, sc)}"
        widgets.text(p.surface, small, label, (x + 4, STACK_Y + CARD[1] + 2),
                     RED if sc.disgraced else GOLD)
        if lead:
            widgets.text(p.surface, small, "lead", (x + 4, 450), GOLD)


def draw_hand(p: Paint) -> None:
    """Your hand. A card is a BUTTON when exactly ONE legal move plays it; a card with no legal move is
    dimmed, so the hand doubles as a legality display. (A card with several possible moves stays
    panel-only -- a click could not say WHICH move you meant.)"""
    v, med = p.view, p.fonts["med"]
    widgets.text(p.surface, med, f"Your hand (seat {p.me})", (24, 800))
    by_card: dict = {}
    for m in p.legal_moves:
        if m.card is not None:
            by_card.setdefault(m.card, []).append(m)
    for c, x in zip(v.own_hand, row_x(24, len(v.own_hand), 104, CARD[0], ROW_MAX_X_CARDS)):
        moves = by_card.get(c, ())
        playable = len(moves) == 1
        r = pygame.Rect((x, HAND_Y), CARD)
        hot = playable and p.mouse is not None and r.collidepoint(p.mouse)
        widgets.card(p.surface, assets.card_surface(c, CARD), (x, HAND_Y), highlight=hot,
                     dim=(not playable and bool(p.legal_moves)))
        p.previews.append((r, cards.asset_path(c), False))
        if playable:
            p.buttons.append((r, moves[0]))
    if v.muted_values:
        widgets.text(p.surface, p.fonts["small"],
                     f"muted values -> 3: {sorted(v.muted_values)}", (24, 976), MUTE)


def draw_chrome(p: Paint, seed=None) -> dict:
    """The play area's own buttons: How to play | New Game, with Review game slotting in to their LEFT at
    game over. The chain grows leftward and the seed label follows the last of them, so nothing can ever
    collide when Review appears. Scenario sits at the FOOT of the knowledge column instead -- it is a setup
    tool, not a play control."""
    small = p.fonts["small"]
    new_game = widgets.button(p.surface, small, pygame.Rect(ROW_MAX_X - 120, 12, 120, 30),
                              "New Game", p.mouse)
    how_to = widgets.button(p.surface, small, pygame.Rect(new_game.x - 12 - 132, 12, 132, 30),
                            "How to play", p.mouse)
    scenario = widgets.button(p.surface, small,
                              pygame.Rect(KNOW_X + 10, WINDOW[1] - 44, PANEL_X - KNOW_X - 20, 32),
                              "Scenario  [S]", p.mouse)
    review = None
    left_anchor = how_to.x
    if not p.view.pending:                              # no pending decision -> terminal
        review = widgets.button(p.surface, small, pygame.Rect(how_to.x - 12 - 128, 12, 128, 30),
                                "Review game", p.mouse, base=BTN_HOVER)
        left_anchor = review.x
    if seed is not None:
        s = f"seed {seed}"
        widgets.text(p.surface, small, s, (left_anchor - 12 - small.size(s)[0], new_game.y + 7), MUTE)
    return {"new_game": new_game, "how_to": how_to, "scenario": scenario, "review": review}


def draw_play_area(p: Paint, seed=None) -> dict:
    """Every zone of the play area, in the original back-to-front order (chrome sits between the lives and
    the antechambers -- kept exactly, so not a pixel moves)."""
    draw_opponent(p)
    draw_lives(p)
    chrome = draw_chrome(p, seed)
    draw_antechambers(p)
    draw_leftover(p)
    draw_stack(p)
    draw_hand(p)
    return chrome
