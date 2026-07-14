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


def test_how_to_play_panel_renders_and_clamps_scroll():
    from imposterkings.ui.card_text import deck_entries
    from imposterkings.ui.render import draw_how_to_play, how_to_play_height

    pygame.display.init()
    screen = pygame.display.set_mode(WINDOW)
    fonts = make_fonts()
    # renders at any scroll offset (incl. absurd ones) and always clamps to its own content height
    for scroll in (0, 50, 400, 99999):
        ctrl = draw_how_to_play(screen, fonts, (0, 0), scroll=scroll)
        assert hasattr(ctrl["close"], "collidepoint")
        assert 0 <= ctrl["scroll"] <= max(0, ctrl["total"] - ctrl["body"].h)
    assert ctrl["total"] == how_to_play_height(fonts) > 0
    assert len(deck_entries()) == 14                       # every card is listed
    pygame.display.quit()


def test_chrome_buttons_give_hover_feedback():
    """Every clickable chrome button must LIGHT UP under the cursor -- the same tactile cue the action
    buttons give. Compared pixel-wise, so a button that quietly stops responding gets caught."""
    from imposterkings.ui.render import BTN, BTN_HOVER

    pygame.display.init()
    screen = pygame.display.set_mode(WINDOW)
    fonts = make_fonts()
    st = mainstate(hand0=(cid("Queen"), cid("Fool")), hand1=(cid("Soldier"),), stack=(sc("Elder"),))
    view = st.information_set(0)
    legal = view.legal_moves()

    def edge(rect, mouse):                       # a pixel just inside the button's top edge
        render_frame(screen, view, fonts, legal, attn_available=True, mouse=mouse)
        return screen.get_at((rect.centerx, rect.y + 2))[:3]

    f = render_frame(screen, view, fonts, legal, attn_available=True, mouse=(0, 0))
    for name in ("how_to", "new_game", "scenario", "settings", "attn_toggle",
                 "reasoning_toggle", "hint_toggle"):
        r = getattr(f, name)
        assert r is not None, name
        assert edge(r, (0, 0)) == BTN, f"{name} idle colour changed"
        assert edge(r, r.center) == BTN_HOVER, f"{name} gives NO hover feedback"
    pygame.display.quit()


def test_scenario_sits_at_the_foot_of_the_knowledge_column():
    from imposterkings.ui.render import KNOW_X, PANEL_X

    pygame.display.init()
    screen = pygame.display.set_mode(WINDOW)
    fonts = make_fonts()
    st = mainstate(hand0=(cid("Queen"),), hand1=(cid("Fool"),), stack=(sc("Elder"),))
    view = st.information_set(0)
    f = render_frame(screen, view, fonts, view.legal_moves())
    assert KNOW_X <= f.scenario.x and f.scenario.right <= PANEL_X   # inside the knowledge column
    assert f.scenario.bottom <= WINDOW[1]                            # on-screen
    assert f.scenario.y > WINDOW[1] // 2                             # at its FOOT, not the top
    for other in (f.how_to, f.new_game):                             # and clear of the top-right row
        assert not f.scenario.colliderect(other)
    pygame.display.quit()


def test_top_right_buttons_never_overlap_even_at_game_over():
    """The button chain grows leftward (new_game <- scenario <- how_to <- review); Review only appears at
    game over, which is exactly when a naive fixed layout would collide."""
    pygame.display.init()
    screen = pygame.display.set_mode(WINDOW)
    fonts = make_fonts()
    st = mainstate(hand0=(cid("Queen"),), hand1=(cid("Fool"),), stack=(sc("Elder"),))
    view = st.information_set(0)
    frame = render_frame(screen, view, fonts, view.legal_moves(), seed=1)
    assert frame.how_to is not None
    rects = [r for r in (frame.review, frame.how_to, frame.scenario, frame.new_game) if r]
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            assert not a.colliderect(b)
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


def test_checkpoint_label_never_overflows_its_box():
    """Checkpoint paths are ABSOLUTE (they resolve against the bundle root, not the cwd), so rendering the
    path ran straight over the < > arrows. Show the bare name; ellipsize anything that still would not fit."""
    from imposterkings.ui.modals import _ckpt_label, draw_settings_overlay

    long_abs = r"C:\Users\Somebody\Very\Deeply\Nested\Path\models\gen1_v3c_v2feat\attn_d64_L2.npz"
    assert _ckpt_label(long_abs) == "attn_d64_L2"                # the name, not the path
    assert _ckpt_label("models/mlp_256.pt") == "mlp_256"
    # same basename in two dirs (rife in a dev models/ tree) -> disambiguate with the parent, not the path
    clash = ["models/sweep_v3a/attn_d64_L2.pt", "models/gen1_v3c_v2feat/attn_d64_L2.pt"]
    assert _ckpt_label(clash[0], clash) == "sweep_v3a/attn_d64_L2"
    assert _ckpt_label(clash[1], clash) == "gen1_v3c_v2feat/attn_d64_L2"

    pygame.display.init()
    screen = pygame.display.set_mode(WINDOW)
    fonts = make_fonts()
    ckpts = [long_abs, "models/" + "x" * 120 + "/an_absurdly_long_checkpoint_name.npz"]
    ctrl = draw_settings_overlay(screen, fonts, {"mode": "nn", "N": 800, "k": 20, "l": 3}, (0, 0),
                                 nn_ckpts=ckpts, nn_ckpt_ix=1)
    assert ctrl["ckpt_prev"] and ctrl["ckpt_next"]
    assert not ctrl["ckpt_prev"].colliderect(ctrl["ckpt_next"])   # the arrows survive a monstrous name
    pygame.display.quit()


def test_review_backlog_is_exactly_what_annotate_would_have_recomputed():
    """The 'Review game' stall was annotate_dual_evals searching every gap at once, on the UI thread. The app
    now drains those gaps on the idle worker during play. This pins the two views of 'what is missing' to
    each other: if missing_dual_evals under-reports, the stall silently comes back."""
    import numpy as np
    from imposterkings.agents import MCTSAgent
    from imposterkings.ui.review import (annotate_dual_evals, build_trajectory, missing_dual_evals,
                                         set_seat_result, turns_of, _search_from)

    traj = build_trajectory(iters=30, seed=3, cross_eval=False)   # no dual evals yet -> a full backlog
    todo = missing_dual_evals(traj)
    assert todo, "expected gaps in a freshly played game"

    rng = np.random.default_rng(0)
    for i, s in todo:                                             # what the idle worker does, one at a time
        set_seat_result(traj[i], s, _search_from(traj[i].state, s, 30, rng))

    assert missing_dual_evals(traj) == []                         # backlog drained...
    assert annotate_dual_evals(traj, 30, rng) == 0                # ...so the review computes NOTHING: no stall
    for start, _e, _o in turns_of(traj):                          # and the graph has both seats' evals
        if traj[start].state is not None:
            assert traj[start].eval_by_seat is not None


def test_a_panel_that_is_ON_but_still_searching_says_thinking_not_toggle_me_on():
    """Searches are ASYNCHRONOUS now (they used to block the UI thread). That opened a window where a panel
    is switched ON but its result has not landed -- and it rendered "(toggle for your read of this
    position)", i.e. it told you to enable something that was already enabled. While the worker is on it,
    the panel must say it is thinking."""
    from imposterkings.ui.side_panel import draw_reasoning_section
    from imposterkings.ui.theme import GOLD, MUTE

    pygame.display.init()
    screen = pygame.display.set_mode(WINDOW)
    fonts = make_fonts()

    def render(shown, pending):
        screen.fill((0, 0, 0))
        draw_reasoning_section(screen, fonts, 200, "Your read:", None, shown,
                               "(toggle for your read of this position)", pending=pending)
        # was anything drawn in the placeholder row, and in which colour?
        row = [screen.get_at((x, y))[:3] for y in range(224, 240) for x in range(1300, 1700)]
        return {"gold": GOLD in row, "mute": MUTE in row}

    on_pending = render(shown=True, pending=True)
    on_ready = render(shown=True, pending=False)
    off = render(shown=False, pending=False)

    assert on_pending["gold"] and not on_pending["mute"], "an ON, still-searching panel must say (thinking...)"
    assert on_ready["mute"] and not on_ready["gold"], "an ON, idle panel keeps its normal placeholder"
    assert not off["gold"] and not off["mute"], "a hidden panel draws no placeholder at all"
    pygame.display.quit()


def test_how_to_play_cards_are_right_click_zoomable_like_the_board():
    """The panel told you to "right-click any card to zoom it" while its OWN 14 thumbnails were dead -- and
    worse, a right-click there fell through to the board previews underneath, zooming whichever card the
    panel happened to be covering. The panel now hands out previews in the SAME (rect, asset) shape the
    board uses, so one draw_card_preview path serves both."""
    from imposterkings.cards import CARD_NAMES, asset_path, card_ids_for_name
    from imposterkings.ui.render import draw_card_preview, draw_how_to_play

    pygame.display.init()
    screen = pygame.display.set_mode(WINDOW)
    fonts = make_fonts()

    ctrl = draw_how_to_play(screen, fonts, (0, 0), scroll=0)
    previews = ctrl["previews"]
    assert previews, "the panel's card thumbnails are not clickable"

    # every target is a real card's art, and lies inside the scrollable body (not under the chrome)
    by_asset = {a for _r, a in previews}
    assert by_asset <= {asset_path(card_ids_for_name(n)[0]) for n in CARD_NAMES}
    for r, a in previews:
        assert r.colliderect(ctrl["body"])
        draw_card_preview(screen, fonts, a)               # the SAME zoom the board uses -> renders

    # scrolling to the bottom exposes the other column's cards; across both offsets every card is reachable
    seen = set(by_asset)
    bottom = draw_how_to_play(screen, fonts, (0, 0), scroll=ctrl["total"])
    seen |= {a for _r, a in bottom["previews"]}
    assert len(seen) == len(CARD_NAMES), f"only {len(seen)}/{len(CARD_NAMES)} cards are zoomable"
    pygame.display.quit()
