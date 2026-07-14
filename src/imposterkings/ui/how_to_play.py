"""The "How to play" modal: the rules summary + a reference row for all 14 cards.

The copy itself lives in ``card_text`` (pygame-free) -- this module only paints it. Card
name/value/copies come from ``DECK_SPEC`` via ``card_text.deck_entries()``, so the reference can never
drift from the deck.
"""
from __future__ import annotations

import pygame

from .. import cards
from . import card_text
from . import assets, widgets
from .theme import BTN, BTN_HOVER, CARD_COLORS, DIVIDER, GOLD, INK, MUTE, NEUTRAL, PANEL, WINDOW


_HTP = (0.90, 0.94)            # "How to play" modal, as a fraction of the window
_HTP_THUMB = (58, 79)          # card art thumbnail in the reference list


def how_to_play_height(fonts) -> int:
    """Total pixel height of the panel's scrollable body -- so the caller can clamp its scroll offset."""
    small = fonts["small"]
    lh = small.get_linesize()
    w = int(_HTP[0] * WINDOW[0]) - 2 * 24
    rules_h = sum(len(widgets.wrap(small, prose, w - 130)) * lh + 6 for _, prose in card_text.RULES)
    col_w = (w - 30) // 2
    txt_w = col_w - _HTP_THUMB[0] - 12
    rows = card_text.deck_entries()
    heights = [max(_HTP_THUMB[1], 22 + len(widgets.wrap(small, t, txt_w)) * lh) + 6
               for _, _, _, t in rows]
    half = (len(heights) + 1) // 2                      # 2 columns -> only the taller column matters
    cards_h = max(sum(heights[:half]), sum(heights[half:]))
    return rules_h + cards_h + 110                      # + the two section headings and their gaps


def draw_how_to_play(surface, fonts, mouse, scroll: int = 0) -> dict:
    """The "How to play" modal: a short rules summary + every card's art and ability.

    Content comes from ``card_text`` (pure data, no pygame) -- the wording lives in one place and
    its numbers are interpolated from ``rules.py``, so this panel cannot drift from the engine. Body is
    clipped and offset by ``scroll`` (px) so the list can never spill out of the box.

    Returns ``{"close", "body", "total", "scroll", "previews"}``. ``previews`` is ``[(rect, asset)]`` for the
    card thumbnails -- the SAME shape ``Frame.previews`` uses for the board, so the caller feeds both into
    one ``draw_card_preview`` path instead of growing a second zoom implementation."""
    big, med, small = fonts["big"], fonts["med"], fonts["small"]
    W, H = WINDOW
    dim = pygame.Surface((W, H), pygame.SRCALPHA)
    dim.fill((0, 0, 0, 175))
    surface.blit(dim, (0, 0))
    bw, bh = int(_HTP[0] * W), int(_HTP[1] * H)
    bx, by = (W - bw) // 2, (H - bh) // 2
    pygame.draw.rect(surface, PANEL, (bx, by, bw, bh), border_radius=8)
    pygame.draw.rect(surface, GOLD, (bx, by, bw, bh), 2, border_radius=8)

    pad = 24
    widgets.text(surface, big, "ImposterKings", (bx + pad, by + 16), GOLD)
    close = pygame.Rect(bx + bw - pad - 84, by + 18, 84, 28)
    pygame.draw.rect(surface, BTN_HOVER if close.collidepoint(mouse) else BTN, close, border_radius=4)
    widgets.text(surface, small, "Close [H]", (close.x + 9, close.y + 6), INK)

    body = pygame.Rect(bx + pad, by + 60, bw - 2 * pad, bh - 60 - 16)
    total = how_to_play_height(fonts)
    scroll = max(0, min(scroll, max(0, total - body.h)))
    prev_clip = surface.get_clip()
    surface.set_clip(body)                              # nothing may render outside the body
    x0, y = body.x, body.y - scroll
    lh = small.get_linesize()

    widgets.text(surface, med, "RULES", (x0, y), GOLD)
    y += 30
    for label, prose in card_text.RULES:
        widgets.text(surface, small, label, (x0, y), INK)
        for i, line in enumerate(widgets.wrap(small, prose, body.w - 130)):
            widgets.text(surface, small, line, (x0 + 120, y + i * lh), MUTE)
        y += max(lh, len(widgets.wrap(small, prose, body.w - 130)) * lh) + 6

    y += 14
    widgets.text(surface, med, "CARDS", (x0, y), GOLD)
    hint = "right-click any card to zoom it"          # true HERE too now, not just on the board
    widgets.text(surface, small, hint, (body.right - small.size(hint)[0], y + 4), MUTE)
    y += 32

    col_w = (body.w - 30) // 2
    txt_w = col_w - _HTP_THUMB[0] - 12
    rows = card_text.deck_entries()
    half = (len(rows) + 1) // 2
    previews = []                                     # (rect, assets/ filename) -- the SAME shape the board
    for col, chunk in enumerate((rows[:half], rows[half:])):        # hands out, so one zoom path serves both
        cx, cy = x0 + col * (col_w + 30), y
        for name, value, copies, text in chunk:
            lines = widgets.wrap(small, text, txt_w)
            rh = max(_HTP_THUMB[1], 22 + len(lines) * lh) + 6
            if cy + rh > body.y - 40 and cy < body.bottom + 40:      # cheap cull for scrolled-away rows
                cid = cards.card_ids_for_name(name)[0]
                thumb = pygame.Rect(cx, cy, *_HTP_THUMB)
                try:
                    surface.blit(assets.card_surface(cid, _HTP_THUMB), (cx, cy))
                    if thumb.colliderect(body):                      # only the visible part is clickable
                        previews.append((thumb, cards.asset_path(cid)))
                except Exception:                                    # noqa: BLE001 -- art is decoration
                    pygame.draw.rect(surface, MUTE, thumb)
                pygame.draw.rect(surface, CARD_COLORS.get(name, NEUTRAL), thumb, 1)
                tx = cx + _HTP_THUMB[0] + 12
                head = f"{name}  {value}" + (f"   x{copies}" if copies > 1 else "")
                tags = [t.name.lower() for t in cards.card_def(cards.card_ids_for_name(name)[0]).tags]
                widgets.text(surface, small, head, (tx, cy), CARD_COLORS.get(name, INK))
                if tags:
                    widgets.text(surface, small, "  ".join(tags),
                          (tx + small.size(head)[0] + 14, cy), MUTE)
                for i, line in enumerate(lines):
                    widgets.text(surface, small, line, (tx, cy + 22 + i * lh), INK)
            cy += rh

    surface.set_clip(prev_clip)
    if total > body.h:                                  # scrollbar: only when there IS more to see
        track = pygame.Rect(bx + bw - 10, body.y, 4, body.h)
        pygame.draw.rect(surface, DIVIDER, track, border_radius=2)
        kh = max(24, int(body.h * body.h / total))
        ky = body.y + int((body.h - kh) * scroll / max(1, total - body.h))
        pygame.draw.rect(surface, GOLD, (track.x, ky, 4, kh), border_radius=2)
    return {"close": close, "body": body, "total": total, "scroll": scroll, "previews": previews}

