"""Icicle layout / card colors / review helpers (headless where pygame drawing is involved)."""
from __future__ import annotations

import os
import types

import numpy as np
import pytest

pygame = pytest.importorskip("pygame")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from imposterkings.actions import Action, ActionKind, StepKind  # noqa: E402
from imposterkings.cards import card_name  # noqa: E402
from imposterkings.mcts import SearchConfig, search  # noqa: E402
from imposterkings.state import GameState  # noqa: E402
from imposterkings.ui.render import make_fonts  # noqa: E402
from imposterkings.ui.render import WINDOW  # noqa: E402
from imposterkings.ui.review import (  # noqa: E402
    TL_TOP, PlyRecord, _draw_graph, _draw_strip, _headline_card, annotate_dual_evals, build_trajectory,
    played_path, turn_for_seat, turns_of,
)
from imposterkings.ui.tree_view import (  # noqa: E402
    CARD_COLORS, NEUTRAL, Block, block_at, draw_icicle, draw_outline, draw_tooltip, layout_icicle,
    move_color, path_node_ids,
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
    assert move_color(Action(ActionKind.GUESS_CARD, name="Queen")) == CARD_COLORS["Queen"]
    assert move_color(Action(ActionKind.CHOOSE_NUMBER, number=7)) in (CARD_COLORS["Warlord"],
                                                                      CARD_COLORS["Mystic"])
    # revealed reactions keep their card colors; a bare decline stays neutral
    assert move_color(Action(ActionKind.REVEAL_KINGSHAND)) == CARD_COLORS["KingsHand"]
    assert move_color(Action(ActionKind.REVEAL_ASSASSIN)) == CARD_COLORS["Assassin"]
    assert move_color(Action(ActionKind.DECLINE_REACTION)) == NEUTRAL


def test_layout_icicle_partition_visits_and_path():
    res = _searched()
    rect = (0.0, 0.0, 600.0, 400.0)
    blocks = layout_icicle(res.root, rect, res.info.observer, top_k=4, max_turns=4)
    assert blocks and all(isinstance(b, Block) for b in blocks)
    for b in blocks:                                        # every block stays inside the rect
        assert 0.0 <= b.x and b.x + b.w <= 600.0 + 1e-6
        assert 0.0 <= b.y and b.y + b.h <= 400.0 + 1e-6
        assert b.visits >= 0 and 0.0 <= b.visit_pct <= 100.0 + 1e-6 and b.node is not None
        assert b.band >= 0 and b.mover in (0, 1)            # turn-band index + mover for separators
    assert any(b.band == 0 for b in blocks)                 # a first band exists
    # all blocks in a band share one mover, and a band spans one contiguous y-range (globally aligned)
    from collections import defaultdict
    by_band = defaultdict(list)
    for b in blocks:
        by_band[b.band].append(b)
    for band, bs in by_band.items():
        assert len({b.mover for b in bs}) == 1              # single mover per band
    top = [b for b in blocks if b.y < 1e-6]                 # first band = root's own moves
    assert max(top, key=lambda b: b.w).move == res.best_move
    assert any(b.y > 1e-6 for b in blocks)                  # deeper bands sit below
    # highlighting the played line marks exactly the walked path (best_move at the root)
    ids = path_node_ids(res.root, [res.best_move])
    on = [b for b in layout_icicle(res.root, rect, res.info.observer, on_path_ids=ids) if b.on_path]
    assert len(on) == 1 and on[0].move == res.best_move


def test_turns_selection_and_played_path():
    mv = Action(ActionKind.FLIP_KING)

    def rec(owner, seat):
        return PlyRecord(seat, mv, types.SimpleNamespace(turn_player=owner), None)

    # P0's turn (plies 0-1, incl. P1's reaction at ply 1), P1's turn (ply 2), P0's turn (plies 3-4)
    traj = [rec(0, 0), rec(0, 1), rec(1, 1), rec(0, 0), rec(0, 0)]
    assert turns_of(traj) == [(0, 1, 0), (2, 2, 1), (3, 4, 0)]
    turns = turns_of(traj)
    assert turn_for_seat(turns, 0, 0) == (0, 1)             # P0's first turn
    assert turn_for_seat(turns, 1, 1) is None               # P1 hasn't owned a turn yet
    assert turn_for_seat(turns, 1, 2) == (2, 2)
    assert turn_for_seat(turns, 0, 4) == (3, 4)             # P0's latest turn
    assert played_path(traj, 3, 3) == [traj[3].move]        # partial (within-turn step)
    assert played_path(traj, 3, 4) == [traj[3].move, traj[4].move]


def test_draw_hittest_tooltip_and_zoom_headless():
    pygame.display.init()
    screen = pygame.display.set_mode((800, 600))
    fonts = make_fonts()
    res = _searched()
    blocks = draw_icicle(screen, fonts, res, (0, 0, 800, 400), played_path=[res.best_move])
    assert blocks
    b0 = blocks[0]
    assert block_at(blocks, (b0.x + 2, b0.y + 2)) is b0     # hit-test finds the block
    assert block_at(blocks, (400, 590)) is None             # below the tree -> nothing
    draw_tooltip(screen, fonts, b0, (100, 100))             # draws without error
    child = max(res.root.children.values(), key=lambda c: c.n)
    assert draw_icicle(screen, fonts, res, (0, 0, 800, 400), zoom_root=child)  # zoomed layout
    draw_icicle(screen, fonts, res, (0, 0, 800, 400), dim=True)                 # persisted/faded
    assert draw_outline(screen, fonts, res, (0, 0, 800, 400), expanded=set(), played_move=res.best_move)
    assert draw_icicle(screen, fonts, None, (0, 0, 800, 400)) == []
    pygame.display.quit()


def test_headline_card_of_a_turn():
    play = Action(ActionKind.PLAY_CARD, card=3)

    def rec(mv, owner):
        return PlyRecord(owner, mv, types.SimpleNamespace(turn_player=owner), None)

    # a play turn (play + declare) -> the played card is the headline; a king-flip turn has none
    traj = [rec(play, 0), rec(Action(ActionKind.DECLARE_ABILITY), 0), rec(Action(ActionKind.FLIP_KING), 1)]
    assert _headline_card(traj, 0, 1) == 3
    assert _headline_card(traj, 2, 2) is None


def test_timeline_graph_and_strip_render_headless():
    pygame.display.init()
    screen = pygame.display.set_mode(WINDOW)
    fonts = make_fonts()
    traj = build_trajectory(iters=30, seed=0)
    turns = turns_of(traj)
    _draw_graph(screen, fonts, traj, turns, 0, TL_TOP)            # combined two-line graph, no error
    hits = _draw_strip(screen, fonts, traj, turns, 0)             # one clickable card per turn
    assert len(hits) == len(turns)
    assert all(hasattr(r, "collidepoint") and s == turns[i][0] for i, (r, s) in enumerate(hits))
    pygame.display.quit()


def test_build_trajectory_small():
    traj = build_trajectory(iters=30, seed=0)
    assert traj and all(r.seat in (0, 1) for r in traj)
    assert any(r.result is not None and r.result.root is not None for r in traj)
    assert all(r.state is not None for r in traj)           # full state still captured (for later use)


def test_build_trajectory_cross_evals_both_seats_every_turn():
    traj = build_trajectory(iters=30, seed=0)
    for s, e, owner in turns_of(traj):
        rec = traj[s]
        eb = rec.eval_by_seat                               # every turn start carries BOTH seats' reads
        assert eb is not None and len(eb) == 2 and all(-1.0 <= v <= 1.0 for v in eb)
        rbs = rec.result_by_seat                            # ...and BOTH seats' retained search trees
        assert rbs is not None and rbs[0] is not None and rbs[1] is not None
        assert rbs[0].info.observer == 0 and rbs[1].info.observer == 1
        if rec.result is not None:                          # mover side reuses that seat's own search
            assert abs(eb[rec.seat] - rec.result.root_value()) < 1e-9
    plain = build_trajectory(iters=20, seed=0, cross_eval=False)
    assert all(r.eval_by_seat is None and r.result_by_seat is None for r in plain)  # opt-out leaves unset


def test_annotate_dual_evals_fills_a_bare_trajectory_and_reuses():
    # The live app builds its trajectory without cross-evals, then annotates before opening the review.
    traj = build_trajectory(iters=20, seed=0, cross_eval=False)
    assert all(r.eval_by_seat is None for r in traj) and all(r.state is not None for r in traj)
    n = annotate_dual_evals(traj, 20, np.random.default_rng(0))
    assert isinstance(n, int) and n > 0                      # searched the bare trajectory's gaps
    for s, e, owner in turns_of(traj):
        rec = traj[s]
        assert rec.eval_by_seat is not None and all(-1.0 <= v <= 1.0 for v in rec.eval_by_seat)
        rbs = rec.result_by_seat
        assert rbs is not None and rbs[0].info.observer == 0 and rbs[1].info.observer == 1
    # a second pass reuses everything already present -> no recomputation
    assert annotate_dual_evals(traj, 20, np.random.default_rng(0)) == 0
    # a budget POLICY (callable) is also accepted: gap searches are sized per-turn (app review uses this)
    from imposterkings.budget import hybrid
    traj2 = build_trajectory(iters=20, seed=1, cross_eval=False)
    assert annotate_dual_evals(traj2, hybrid(20, 3), np.random.default_rng(0)) > 0
    assert all(traj2[s].result_by_seat is not None for s, e, o in turns_of(traj2))


def test_build_trajectory_with_budget_uses_variable_iters():
    # standalone ui.review can run the bot-vs-bot game + dual-eval under a budget policy (hybrid default).
    from imposterkings.budget import hybrid
    traj = build_trajectory(iters=20, seed=0, budget=hybrid(20, 3))
    its = {traj[s].result.iterations for s, e, o in turns_of(traj) if traj[s].result is not None}
    assert len(its) > 1                                       # per-turn budget varies -> not a flat number
    for s, e, o in turns_of(traj):
        assert traj[s].eval_by_seat is not None and traj[s].result_by_seat is not None
