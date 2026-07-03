"""Icicle layout / card colors / review helpers (headless where pygame drawing is involved)."""
from __future__ import annotations

import os

import numpy as np
import pytest

pygame = pytest.importorskip("pygame")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from imposterkings.actions import Action, ActionKind, StepKind  # noqa: E402
from imposterkings.cards import card_name  # noqa: E402
from imposterkings.mcts import SearchConfig, search  # noqa: E402
from imposterkings.state import GameState  # noqa: E402
from imposterkings.ui.review import PlyRecord, build_trajectory, panels_for_cursor  # noqa: E402
from imposterkings.ui.render import make_fonts  # noqa: E402
from imposterkings.ui.tree_view import (  # noqa: E402
    CARD_COLORS, NEUTRAL, Block, draw_icicle, draw_outline, layout_icicle, move_color,
)


def _searched():
    rng = np.random.default_rng(1)
    st = GameState.deal(rng, starting_player=0)
    while st.phase in (StepKind.SETUP_HIDE, StepKind.SETUP_DISCARD):
        st = st.apply(st.legal_moves()[0])
    view = st.information_set(st.to_play)
    return search(view, SearchConfig(rng=np.random.default_rng(0), iterations=300))


def test_move_color_card_guess_and_number():
    assert move_color(Action(ActionKind.PLAY_CARD, card=0)) == CARD_COLORS[card_name(0)]
    assert move_color(Action(ActionKind.DECLARE_ABILITY)) == NEUTRAL
    # a guess is colored by the guessed card; a Mystic mute by a card of that value
    assert move_color(Action(ActionKind.GUESS_CARD, name="Queen")) == CARD_COLORS["Queen"]
    assert move_color(Action(ActionKind.CHOOSE_NUMBER, number=7)) in (CARD_COLORS["Warlord"],
                                                                      CARD_COLORS["Mystic"])


def test_layout_icicle_partition_and_bands():
    res = _searched()
    rect = (0.0, 0.0, 600.0, 400.0)
    blocks = layout_icicle(res.root, rect, res.info.observer, top_k=4, max_turns=4)
    assert blocks and all(isinstance(b, Block) for b in blocks)
    for b in blocks:                                        # every block stays inside the rect
        assert 0.0 <= b.x and b.x + b.w <= 600.0 + 1e-6
        assert 0.0 <= b.y and b.y + b.h <= 400.0 + 1e-6
    # the first band (turn 0, y == 0) is the root's own moves; widest = most-visited = best_move
    top = [b for b in blocks if b.y < 1e-6]
    assert sum(b.w for b in top) <= 600.0 + 1e-6
    assert max(top, key=lambda b: b.w).move == res.best_move
    # deeper bands sit strictly below the first band (ply-banding pushes each turn down)
    assert any(b.y > 1e-6 for b in blocks)
    # exactly one block is flagged as the played move
    played = layout_icicle(res.root, rect, res.info.observer, played_move=res.best_move)
    assert sum(1 for b in played if b.is_played) == 1


def test_panels_for_cursor():
    def rec(seat):
        return PlyRecord(seat, Action(ActionKind.FLIP_KING), None, None)
    traj = [rec(0), rec(1), rec(0), rec(0), rec(1)]
    assert panels_for_cursor(traj, 0) == (0, None)          # only P0 has moved
    assert panels_for_cursor(traj, 1) == (0, 1)
    assert panels_for_cursor(traj, 3) == (3, 1)             # latest P0 is index 3
    assert panels_for_cursor(traj, 4) == (3, 4)


def test_draw_icicle_and_outline_headless():
    pygame.display.init()
    screen = pygame.display.set_mode((800, 600))
    fonts = make_fonts()
    res = _searched()
    assert draw_icicle(screen, fonts, res, (0, 0, 800, 400), played_move=res.best_move)
    assert draw_outline(screen, fonts, res, (0, 0, 800, 400), expanded=set(), played_move=res.best_move)
    # a None result renders the placeholder without error and returns empty
    assert draw_icicle(screen, fonts, None, (0, 0, 800, 400)) == []
    pygame.display.quit()


def test_build_trajectory_small():
    traj = build_trajectory(iters=30, seed=0)
    assert traj and all(r.seat in (0, 1) for r in traj)
    assert any(r.result is not None and r.result.root is not None for r in traj)
