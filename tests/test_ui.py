"""Headless smoke test for the PyGame renderer (skipped if pygame is unavailable)."""
from __future__ import annotations

import os

import pytest

pygame = pytest.importorskip("pygame")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from imposterkings.ui.render import (  # noqa: E402
    WINDOW, draw_settings_overlay, make_fonts, render_frame,
)

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
    assert hasattr(frame.hint_toggle, "collidepoint")            # hint toggle is hit-testable
    assert hasattr(frame.settings, "collidepoint")               # Settings button is hit-testable
    assert frame.review is None                                  # Review button only shows at game over
    out = tmp_path / "frame.png"
    pygame.image.save(screen, str(out))
    assert out.stat().st_size > 0
    pygame.display.quit()


def test_render_frame_draws_pv_lines_from_a_real_search():
    import numpy as np
    from imposterkings.agents import MCTSAgent
    from imposterkings.state import GameState
    from imposterkings.actions import StepKind

    pygame.display.init()
    screen = pygame.display.set_mode(WINDOW)
    fonts = make_fonts()
    rng = np.random.default_rng(0)
    st = GameState.deal(rng, starting_player=0)
    while st.phase in (StepKind.SETUP_HIDE, StepKind.SETUP_DISCARD):
        st = st.apply(st.legal_moves()[0])
    view = st.information_set(st.to_play)
    agent = MCTSAgent(iterations=120)
    agent.select_move(view, rng)                     # populates last_result (with retained tree)
    assert agent.last_result.principal_variations()  # non-empty lines
    # renders both live panels (each with its own-perspective eval) without error
    rv = agent.last_result.root_value()
    frame = render_frame(screen, view, fonts, view.legal_moves(), show_reasoning=True,
                         bot_result=agent.last_result, show_hint=True, hint_result=agent.last_result,
                         bot_eval=-rv, hint_eval=rv, seed=1)
    assert frame.buttons
    assert hasattr(frame.hint_toggle, "collidepoint")
    pygame.display.quit()


def test_settings_overlay_renders_and_returns_controls():
    pygame.display.init()
    screen = pygame.display.set_mode(WINDOW)
    fonts = make_fonts()
    for engine in ({"mode": "mcts", "N": 500, "k": 100, "l": 3},
                   {"mode": "branching", "N": 800, "k": 40, "l": 3},
                   {"mode": "hybrid", "N": 800, "k": 100, "l": 5}):
        ctrl = draw_settings_overlay(screen, fonts, engine, (0, 0))
        assert set(ctrl["pills"]) == {"mcts", "branching", "hybrid", "nn"}
        assert all(hasattr(r, "collidepoint") for r in ctrl["pills"].values())
        assert hasattr(ctrl["close"], "collidepoint")
        keys = [key for _track, _lo, _hi, key in ctrl["sliders"]]
        assert all(hasattr(t, "collidepoint") for t, *_ in ctrl["sliders"])
        # fixed -> just N; branch/hybrid -> k and l
        assert keys == (["N"] if engine["mode"] == "mcts" else ["k", "l"])
    pygame.display.quit()


def test_render_frame_draws_knowledge_column():
    pygame.display.init()
    screen = pygame.display.set_mode(WINDOW)
    fonts = make_fonts()
    st = mainstate(hand0=(cid("Queen"),), hand1=(cid("Fool"),), stack=(sc("Elder"),))
    view = st.information_set(0)
    knowledge = [
        (frozenset({"Princess", "Warlord"}), frozenset({"Fool"}), "50-50"),
        (frozenset({"Queen"}), frozenset(), "perfect"),
    ]
    frame = render_frame(screen, view, fonts, view.legal_moves(), knowledge=knowledge, seed=1)
    assert frame.review is None and hasattr(frame.new_game, "collidepoint")  # drew without error
    pygame.display.quit()
