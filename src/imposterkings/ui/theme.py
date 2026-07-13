"""Colours, card sizes and fonts -- the UI's visual vocabulary.

The FOUNDATION layer: this imports nothing from ``ui``, so nothing can cycle back into it. (Before the
split, the palette lived in ``render`` and ``attention_view`` imported it from there, while ``render``
lazily imported ``attention_view`` back inside a function -- a genuine circular dependency, admitted in a
comment. Both halves now depend on this module instead, and the cycle is gone.)
"""
from __future__ import annotations

import pygame

WINDOW = (1740, 1060)
CARD = (96, 131)            # a hand / stack card
SMALL = (64, 87)            # antechamber, leftover, opponent backs
KING = (int(CARD[0] * 1.1), int(CARD[1] * 1.1))    # the king reads as a LIFE -- bigger than a hand card

BG = (18, 64, 48)
PANEL = (28, 30, 36)
INK = (235, 235, 235)
MUTE = (150, 150, 160)
GOLD = (235, 200, 90)
RED = (200, 70, 70)
BTN = (52, 56, 66)
BTN_HOVER = (78, 96, 120)
DIVIDER = (60, 62, 70)
NEUTRAL = (80, 84, 94)      # non-card items (abilities, unknowns)
TICK = (90, 200, 110)       # the hand-knowledge checkmark
AMBER = (224, 150, 60)

P_COLORS = {0: (95, 160, 240), 1: (240, 170, 90)}   # PV move colors by seat (P0 blue, P1 orange)

# Per-card-name colors (from the card art), shared by the board, the tree view and the knowledge column.
CARD_COLORS = {
    "Queen": (245, 87, 14), "Princess": (242, 92, 15), "Sentry": (224, 127, 59),
    "KingsHand": (199, 103, 43), "Warlord": (240, 154, 9), "Mystic": (238, 154, 6),
    "Oathbound": (226, 202, 32), "Soldier": (80, 172, 123), "Judge": (64, 145, 69),
    "Inquisitor": (76, 155, 168), "Zealot": (122, 156, 194), "Elder": (106, 154, 220),
    "Assassin": (168, 171, 224), "Fool": (167, 113, 175),
}


def make_fonts():
    pygame.font.init()
    return {
        "big": pygame.font.SysFont("consolas,arial", 30),
        "med": pygame.font.SysFont("consolas,arial", 20),
        "small": pygame.font.SysFont("consolas,arial", 16),
    }
