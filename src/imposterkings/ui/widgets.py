"""Generic pygame drawing primitives -- the UI's shared toolkit.

These were private to ``render`` (``_text``, ``_draw_card``, ``_tick`` ...) yet four other modules imported
them across the package boundary. Names shared that widely are not private, so they say so now. Everything
here is stateless and knows nothing about the game -- only surfaces, rects and fonts.
"""
from __future__ import annotations

from typing import List

import pygame

from .theme import BTN, BTN_HOVER, CARD, GOLD, INK, MUTE, RED, TICK


def text(surf, font, s, pos, color=INK) -> None:
    surf.blit(font.render(s, True, color), pos)


def button(surf, font, rect, label, mouse, *, base=BTN, color=INK, pad=None) -> "pygame.Rect":
    """A chrome button that LIGHTS UP under the cursor -- the same tactile feedback the action buttons and
    the modal Close buttons give. ``mouse`` may be None (headless / bot's turn), in which case it simply
    never highlights. Centers ``label`` unless ``pad`` gives an explicit left inset."""
    hot = mouse is not None and rect.collidepoint(mouse)
    pygame.draw.rect(surf, BTN_HOVER if hot else base, rect, border_radius=4)
    if hot:
        pygame.draw.rect(surf, GOLD, rect, 1, border_radius=4)      # a thin gold rim on hover
    tx = rect.x + pad if pad is not None else rect.centerx - font.size(label)[0] // 2
    text(surf, font, label, (tx, rect.centery - font.get_height() // 2), color)
    return rect


def wrap(font, s: str, max_w: int) -> List[str]:
    """Greedy word-wrap ``s`` to lines no wider than ``max_w`` px (pygame has no wrapping of its own)."""
    lines: List[str] = []
    line = ""
    for word in s.split():
        probe = f"{line} {word}".strip()
        if line and font.size(probe)[0] > max_w:
            lines.append(line)
            line = word
        else:
            line = probe
    if line:
        lines.append(line)
    return lines


def text_fit(surf, fonts, s, pos, max_w: int, color=INK) -> None:
    """Draw ``s`` at ``pos`` without ever running past ``max_w``: try the medium font, drop to small, then
    ellipsize. (The decision header sits beside the Attention/Settings buttons; long ability prompts like
    "Interrogate: name a card (or decline)" would otherwise render underneath them.)"""
    for font in (fonts["med"], fonts["small"]):
        if font.size(s)[0] <= max_w:
            text(surf, font, s, pos, color)
            return
    font = fonts["small"]
    while s and font.size(s + "...")[0] > max_w:
        s = s[:-1]
    text(surf, font, s.rstrip() + "...", pos, color)


def card(surf, image, pos, *, highlight=False, dim=False, size=CARD) -> "pygame.Rect":
    rect = pygame.Rect(pos, size)                      # border matches the image size (pass SMALL for minis)
    if dim:
        image = image.copy()
        image.fill((90, 90, 90), special_flags=pygame.BLEND_RGB_MULT)
    surf.blit(image, pos)
    pygame.draw.rect(surf, GOLD if highlight else (10, 10, 10), rect, 3 if highlight else 1)
    return rect


def tokens(surface, font, items, x0: int, y: int, max_x: int, line_h: int, indent: int = 14) -> int:
    """Draw ``(text, color)`` items left-to-right, wrapping (with a small indent) at ``max_x``.
    Returns the y just below the block."""
    space = font.size(" ")[0]
    x = x0
    for s, color in items:
        w = font.size(s)[0]
        if x > x0 and x + w > max_x:           # wrap (but always draw >=1 token per row)
            y += line_h
            x = x0 + indent
        surface.blit(font.render(s, True, color), (x, y))
        x += w + space
    return y + line_h


def tick(surface, x, y) -> None:     # small green checkmark (consolas lacks the glyph, so draw it)
    pygame.draw.lines(surface, TICK, False, [(x + 1, y + 8), (x + 5, y + 12), (x + 13, y + 1)], 2)


def cross(surface, x, y) -> None:    # small red X
    pygame.draw.line(surface, RED, (x + 2, y + 2), (x + 12, y + 12), 2)
    pygame.draw.line(surface, RED, (x + 12, y + 2), (x + 2, y + 12), 2)
