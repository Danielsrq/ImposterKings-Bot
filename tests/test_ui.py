"""Headless smoke test for the PyGame renderer (skipped if pygame is unavailable)."""
from __future__ import annotations

import os

import pytest

pygame = pytest.importorskip("pygame")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from imposterkings.ui.render import WINDOW, make_fonts, render_frame  # noqa: E402

from .helpers import cid, mainstate, sc  # noqa: E402


def test_render_frame_draws_and_returns_buttons(tmp_path):
    pygame.display.init()
    screen = pygame.display.set_mode(WINDOW)
    fonts = make_fonts()
    st = mainstate(hand0=(cid("Queen"), cid("Fool")), hand1=(cid("Soldier"),),
                   stack=(sc("Elder"), sc("Princess", disgraced=True)),
                   antechambers=((), (cid("Mystic"),)), muted={4})
    view = st.information_set(0)
    frame = render_frame(screen, view, fonts, view.legal_moves(), hover=0, status="test",
                         log=["You: play_card(Fool(1)#17)"], show_reasoning=True, seed=123)
    assert frame.buttons, "expected at least one clickable action button"
    assert all(hasattr(rect, "collidepoint") for rect, _ in frame.buttons)
    assert hasattr(frame.new_game, "collidepoint")               # New Game button is hit-testable
    assert hasattr(frame.reasoning_toggle, "collidepoint")       # reasoning toggle is hit-testable
    out = tmp_path / "frame.png"
    pygame.image.save(screen, str(out))
    assert out.stat().st_size > 0
    pygame.display.quit()
