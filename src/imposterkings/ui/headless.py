"""Headless capture: render the app board or the post-game review to a PNG without a window.

Formalizes the throwaway screenshot scripts into a reusable debug/test tool. Combined with
``scenario.build`` + ``review.build_trajectory(initial_state=...)`` / ``review.scripted_trajectory(...)``
this gives a fully programmatic pipeline -- construct a position, drive it to a trajectory, and eyeball or
assert the board / review (icicle, graft, stack-target labels, eval graph, popup) with no interactivity.

    from imposterkings import scenario as sb
    from imposterkings.ui import review, headless
    st = sb.build(hand0=["Oathbound", "Inquisitor", "Queen"], hand1=["Elder", "KingsHand"], stack=["Sentry"])
    traj = review.scripted_trajectory(st, [sb.play_card(sb.cid("Oathbound")), ...])
    headless.review_png(traj, "out.png", cursor=3, mode="icicle")
"""
from __future__ import annotations

import os
from typing import List, Optional

_INITED = False


def session():
    """Init a headless PyGame (SDL dummy video/audio) and return ``(screen, fonts)``. Idempotent."""
    global _INITED
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame
    from .render import WINDOW, make_fonts
    if not _INITED:
        pygame.init()
        _INITED = True
    screen = pygame.display.set_mode(WINDOW)
    return screen, make_fonts()


def board_png(state, path: str, *, observer: Optional[int] = None, screen=None, fonts=None) -> None:
    """Render the in-game board (``render_frame`` from ``observer``'s view, default the player to move) to
    ``path``. ``legal`` is populated so action buttons show."""
    import pygame
    from .render import render_frame
    if screen is None or fonts is None:
        screen, fonts = session()
    obs = state.to_play if observer is None else observer
    view = state.information_set(obs)
    legal = view.legal_moves() if (not state.is_terminal() and state.to_play == obs) else []
    render_frame(screen, view, fonts, legal, status="(headless)")
    pygame.image.save(screen.copy(), path)


def review_png(traj: List, path: str, *, cursor: int = 0, mode: str = "icicle", renorm: bool = False,
               screen=None, fonts=None) -> None:
    """Render one post-game review frame (the shared ``render_review_frame``) for ``traj`` to ``path``."""
    import pygame
    from .review import render_review_frame, turns_of
    if not traj:
        raise ValueError("review_png: empty trajectory")
    if screen is None or fonts is None:
        screen, fonts = session()
    cursor = max(0, min(cursor, len(traj) - 1))
    render_review_frame(screen, fonts, traj, turns_of(traj), cursor=cursor, mode=mode, renorm=renorm)
    pygame.image.save(screen.copy(), path)
