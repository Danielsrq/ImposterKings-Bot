"""Load and cache card art from ``assets/``, scaled on demand.

The JPGs are value-prefixed (``09_Queen.jpg``) with a couple of normalization quirks the registry
already encodes (``06_oathbound_alt.jpg`` lowercase, ``08_King_s-Hand_alt.jpg``). Kings and the card
back are non-deck art. Surfaces are cached by ``(filename, size)``.
"""
from __future__ import annotations

from typing import Dict, Tuple

import pygame

from .. import cards
from ..paths import asset_dir

ASSETS_DIR = asset_dir()        # frozen-aware: sys._MEIPASS in a build, the repo root from source

_CACHE: Dict[Tuple[str, int, int], "pygame.Surface"] = {}


def _load_scaled(filename: str, size: Tuple[int, int]) -> "pygame.Surface":
    key = (filename, size[0], size[1])
    if key not in _CACHE:
        surf = pygame.image.load(str(ASSETS_DIR / filename))
        _CACHE[key] = pygame.transform.smoothscale(surf, size)
    return _CACHE[key]


def card_surface(card: int, size: Tuple[int, int], alt: bool = False) -> "pygame.Surface":
    return _load_scaled(cards.asset_path(card, alt=alt), size)


def image(filename: str, size: Tuple[int, int]) -> "pygame.Surface":
    """Load and cache a non-deck image from ``assets/`` (e.g. ``Crown.jpg``) scaled to ``size``."""
    return _load_scaled(filename, size)


def back_surface(size: Tuple[int, int]) -> "pygame.Surface":
    return _load_scaled(cards.CARD_BACK_ASSET, size)


def king_surface(size: Tuple[int, int], true_king: bool = False) -> "pygame.Surface":
    return _load_scaled(cards.TRUE_KING_ASSET if true_king else cards.KING_ASSET, size)
