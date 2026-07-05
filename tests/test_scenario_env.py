"""The scenario debug/test environment end-to-end: construct a position, drive it to a review-ready
trajectory (scripted OR agent), and render/inspect the board + post-game review headlessly."""
from __future__ import annotations

import os

import pytest

pygame = pytest.importorskip("pygame")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from imposterkings import cards, scenario as sb  # noqa: E402
from imposterkings.actions import StepKind  # noqa: E402
from imposterkings.ui import headless  # noqa: E402
from imposterkings.ui.review import (  # noqa: E402
    _stack_target_cards, build_trajectory, render_review_frame, scripted_trajectory, turns_of,
)


def _kings_hand_line():
    """The Oathbound->Inquisitor->King's-Hand counter, as a scripted review-ready trajectory."""
    st = sb.build(hand0=["Oathbound", "Inquisitor", "Queen"], hand1=["Elder", "KingsHand"],
                  stack=["Sentry"], turn_player=0)
    moves = [sb.play_card(sb.cid("Oathbound")), sb.play_card(sb.cid("Inquisitor")),
             sb.guess("Elder"), sb.REVEAL_KINGSHAND]
    return st, scripted_trajectory(st, moves, iters=40, seed=1)


def test_scripted_trajectory_captures_rules_line_for_review():
    st, traj = _kings_hand_line()
    assert [r.seat for r in traj] == [0, 0, 0, 1]
    assert all(r.state is not None for r in traj) and traj[0].result is not None   # searched, review-ready
    # the fixed rules outcome (turn returns to the active player) is visible in the trajectory
    after = traj[-1].state.apply(traj[-1].move)
    assert after.to_play == 0 and after.phase == StepKind.MAIN
    assert cards.card_name(after.leading.card) == "Oathbound"
    # and it renders in the review without error
    screen, fonts = headless.session()
    hit = render_review_frame(screen, fonts, traj, turns_of(traj), cursor=len(traj) - 1, mode="icicle")
    assert hit["btns"] and 0 in hit["blocks"] and 1 in hit["blocks"]


def test_build_trajectory_from_scenario_is_review_ready():
    st = sb.build(hand0=["Queen", "Warlord", "Elder"], hand1=["Soldier", "Mystic", "Fool"],
                  stack=["Zealot"], turn_player=0)
    traj = build_trajectory(iters=30, seed=2, initial_state=st)
    assert traj and all(r.state is not None for r in traj)
    for s, e, o in turns_of(traj):                       # cross-eval populated both seats on every turn
        assert traj[s].result_by_seat is not None
    render_review_frame(headless.session()[0], headless.session()[1], traj, turns_of(traj), cursor=0)


def test_scripted_stack_target_resolves_in_icicle():
    # Deterministic replacement for seed-hunting a Sentry game: build one and drive to the swap target.
    st = sb.build(hand0=["Sentry", "Warlord"], hand1=["Elder", "Fool"], stack=["Zealot", "Elder"],
                  turn_player=0)
    moves = [sb.play_card(sb.cid("Sentry")), sb.DECLARE, sb.DECLINE_REACTION, sb.choose_stack_target(0)]
    traj = scripted_trajectory(st, moves, iters=80, seed=3)
    tgt = next(r for r in traj if r.state.phase == StepKind.ABILITY_STACK_TARGET)
    assert tgt.result is not None
    resolved = _stack_target_cards(tgt.result.root, tgt.state)   # the icicle's resolver
    names = {cards.card_name(c) for c in resolved.values()}
    assert {"Zealot", "Elder"} <= names                          # target@N -> real stack cards


def test_headless_board_and_review_png(tmp_path):
    st, traj = _kings_hand_line()
    screen, fonts = headless.session()
    bpath = str(tmp_path / "board.png")
    rpath = str(tmp_path / "review.png")
    headless.board_png(traj[-1].state.apply(traj[-1].move), bpath, screen=screen, fonts=fonts)
    headless.review_png(traj, rpath, cursor=len(traj) - 1, screen=screen, fonts=fonts)
    assert os.path.getsize(bpath) > 0 and os.path.getsize(rpath) > 0
    with pytest.raises(ValueError):                              # empty trajectory guarded
        headless.review_png([], rpath)
