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


def test_hand_cards_are_clickable_buttons():
    """Each hand card that has EXACTLY ONE legal move is a button playing that move; a card with no
    legal move (or several) is not clickable -- a click could not say which move you meant."""
    from imposterkings.ui.render import CARD

    pygame.display.init()
    screen = pygame.display.set_mode(WINDOW)
    fonts = make_fonts()
    # Fool(1) cannot be played over an Elder(3) lead, so it must NOT become a button; Queen(9) must.
    st = mainstate(hand0=(cid("Queen"), cid("Fool")), hand1=(cid("Soldier"),), stack=(sc("Elder"),))
    view = st.information_set(0)
    legal = view.legal_moves()
    frame = render_frame(screen, view, fonts, legal, mouse=(30, 840))

    card_btns = [(r, m) for r, m in frame.buttons if r.size == CARD]
    by_move = {m for _, m in card_btns}
    assert by_move, "expected the playable hand card to be a button"
    assert all(m in legal for m in by_move)                    # never offers an illegal move
    assert all(m.card is not None for m in by_move)
    played = {m.card for m in by_move}
    assert cid("Queen") in played                              # Queen(9) beats the Elder(3) lead
    assert cid("Fool") not in played                           # Fool(1) does not -> not clickable
    # hit-testing a card's center returns exactly its own move (no overlap with the panel buttons)
    for r, m in card_btns:
        assert next(mv for rr, mv in frame.buttons if rr.collidepoint(r.center)) == m
    pygame.display.quit()


def test_king_is_clickable_when_flip_is_legal_and_cards_are_previewable():
    """Your king is a button exactly when FLIP_KING is legal, and every face-up card is a right-click
    zoom target carrying its own art (so the preview can never show the wrong card)."""
    from imposterkings.actions import ActionKind
    from imposterkings.cards import asset_path
    from imposterkings.ui.render import CARD, KING, draw_card_preview

    pygame.display.init()
    screen = pygame.display.set_mode(WINDOW)
    fonts = make_fonts()
    st = mainstate(hand0=(cid("Queen"), cid("Fool")), hand1=(cid("Soldier"),),
                   stack=(sc("Elder"),), antechambers=((), (cid("Mystic"),)))
    view = st.information_set(0)
    legal = view.legal_moves()
    flip = next((m for m in legal if m.kind == ActionKind.FLIP_KING), None)
    frame = render_frame(screen, view, fonts, legal)

    king_btns = [(r, m) for r, m in frame.buttons if r.size == KING]
    if flip is not None:
        assert len(king_btns) == 1 and king_btns[0][1] == flip     # the king plays flip_king
    else:
        assert not king_btns                                       # not flippable -> not a button

    # previews: both kings, the stack card, the antechamber card and every hand card
    by_rect = {(r.x, r.y): a for r, a, _ in frame.previews}
    assert len(frame.previews) == len(by_rect), "overlapping preview targets"
    assert asset_path(cid("Elder")) in by_rect.values()            # stack card
    assert asset_path(cid("Mystic")) in by_rect.values()           # antechamber card
    for r, a, _ in frame.previews:                                 # a hand card previews ITS OWN art
        if r.y == 832 and r.size == CARD:
            hand_at_x = [c for c, x in zip(view.own_hand, [b.x for b, _, _ in frame.previews
                                                           if b.y == 832]) if x == r.x]
            if hand_at_x:
                assert a == asset_path(hand_at_x[0])
    draw_card_preview(screen, fonts, frame.previews[0][1], flipped=frame.previews[0][2])  # renders
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
