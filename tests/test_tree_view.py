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
from imposterkings.mcts import Node, SearchConfig, search  # noqa: E402
from imposterkings.state import GameState, StackCard  # noqa: E402
from imposterkings.ui.render import make_fonts  # noqa: E402
from imposterkings.ui.render import WINDOW  # noqa: E402
from imposterkings.ui.review import (  # noqa: E402
    TL_TOP, PlyRecord, _draw_graph, _draw_strip, _grafted_tree, _headline_card, _stack_target_cards,
    annotate_dual_evals, build_trajectory, played_path, turn_for_seat, turns_of,
)
from imposterkings.ui.tree_view import (  # noqa: E402
    CARD_COLORS, NEUTRAL, Block, block_at, draw_icicle, draw_outline, draw_tooltip, graft_node,
    layout_icicle, move_color, path_node_ids, _stack_target_label,
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


def test_zoom_keeps_clicked_node_as_full_width_root_band():
    pygame.display.init()
    screen = pygame.display.set_mode((800, 600))
    fonts = make_fonts()
    res = _searched()
    rect = (10, 50, 600, 400)
    child = max((c for c in res.root.children.values() if c.children), key=lambda c: c.n)

    plain = draw_icicle(screen, fonts, res, rect)               # not zoomed -> root itself never emitted
    assert plain and all(b.node is not res.root for b in plain)

    zoomed = draw_icicle(screen, fonts, res, rect, zoom_root=child)
    root_blocks = [b for b in zoomed if b.node is child]
    assert len(root_blocks) == 1                                # the clicked node stays visible
    rb = root_blocks[0]
    assert abs(rb.x - 10) < 1e-6 and abs(rb.w - 600) < 1.0      # full-width band
    assert abs(rb.y - min(b.y for b in zoomed)) < 1e-6          # sitting at the top
    kids = [b for b in zoomed if b.node in child.children.values()]
    assert kids and all(k.y > rb.y for k in kids)               # its children hang below it
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


def test_board_popup_renders_headless():
    from imposterkings.ui.render import WINDOW
    from imposterkings.ui.review import _draw_board_popup
    pygame.display.init()
    screen = pygame.display.set_mode(WINDOW)
    fonts = make_fonts()
    traj = build_trajectory(iters=30, seed=0)
    turns = turns_of(traj)
    s = turns[min(6, len(turns) - 1)][0]                     # a turn-start state (hands/hidden/etc. present)
    assert traj[s].state is not None
    _draw_board_popup(screen, fonts, traj[s].state, (60, 60))  # true-board popup renders without error
    _draw_board_popup(screen, fonts, None, (60, 60))           # None state -> no-op
    pygame.display.quit()


def _res0_for(traj, turn):
    s, e, owner = turn
    rec0 = traj[s]
    return (rec0.result_by_seat[owner] if rec0.result_by_seat is not None else rec0.result)


def _graftable_turn(traj):
    """A turn (>=2 plies) whose owner also made the next ply and retained a search there (graftable)."""
    for s, e, owner in turns_of(traj):
        if e > s and traj[s + 1].seat == owner and traj[s + 1].result is not None:
            r0 = _res0_for(traj, (s, e, owner))
            if r0 is not None and getattr(r0, "root", None) is not None and r0.root.children:
                return (s, e, owner)
    return None


def test_grafted_tree_is_noop_at_turn_root():
    traj = build_trajectory(iters=30, seed=0)
    turn = _graftable_turn(traj)
    assert turn is not None
    s, e, owner = turn
    res0 = _res0_for(traj, turn)
    assert _grafted_tree(traj, owner, s, e, s, res0) is None      # cursor at the turn root -> no graft


def test_grafted_tree_replaces_subband_without_overflow_or_mutation():
    pygame.display.init()
    screen = pygame.display.set_mode(WINDOW)
    fonts = make_fonts()
    traj = build_trajectory(iters=30, seed=0)
    turn = _graftable_turn(traj)
    assert turn is not None
    s, e, owner = turn
    res0 = _res0_for(traj, turn)
    orig_child_keys = set(res0.root.children)                    # snapshot to prove res0 is untouched

    g = _grafted_tree(traj, owner, s, e, s + 1, res0)            # step one ply into the turn
    assert g is not None
    gres, graft_ids, dim_ids, tip_sims = g
    assert graft_ids and dim_ids                                 # a band was replaced + siblings greyed
    assert set(res0.root.children) == orig_child_keys           # the real cached tree was NOT mutated

    path = played_path(traj, s, s + 1)
    blocks = draw_icicle(screen, fonts, gres, (6, 120, 600, 400), played_path=path,
                         graft_ids=graft_ids, dim_ids=dim_ids, band_sims=tip_sims)
    assert blocks
    gblocks = [b for b in blocks if id(b.node) in graft_ids]
    assert gblocks
    for gb in gblocks:                                          # containment: children fill the cell exactly
        kids = [b for b in blocks if b.node in gb.node.children.values()]
        assert abs(sum(b.w for b in kids) - gb.w) < 1.0
        # a grafted band may legitimately total MORE visits than its parent cell (its own full budget)
        assert sum(k.visits for k in kids) >= gb.visits
    assert block_at(blocks, (blocks[0].x + 1, blocks[0].y + 1)) is not None  # hit-test still works
    pygame.display.quit()


def test_graft_node_shallow_clone_keeps_stats_swaps_children():
    traj = build_trajectory(iters=30, seed=0)
    turn = _graftable_turn(traj)
    s, e, owner = turn
    res0 = _res0_for(traj, turn)
    orig = next(iter(res0.root.children.values()))
    clone = graft_node(orig, {})
    assert clone is not orig and clone.children == {}           # new node, swapped (empty) children
    assert (clone.n, clone.w, clone.incoming_move, clone.player_just_moved) == \
           (orig.n, orig.w, orig.incoming_move, orig.player_just_moved)
    assert orig.children                                        # original's children left intact


def _graft_band_sum(blocks, graft_ids):
    """(chosen-cell width, sum of the grafted band's child widths) for the first graft node."""
    gb = next(b for b in blocks if id(b.node) in graft_ids)
    kids = [b for b in blocks if b.node in gb.node.children.values()]
    return gb.w, sum(b.w for b in kids)


def test_renormalise_makes_graft_band_full_width_with_childless_dims():
    pygame.display.init()
    screen = pygame.display.set_mode(WINDOW)
    fonts = make_fonts()
    traj = build_trajectory(iters=30, seed=0)
    turn = _graftable_turn(traj)
    assert turn is not None
    s, e, owner = turn
    res0 = _res0_for(traj, turn)
    rect = (6, 120, 600, 400)                                   # panel width W = 600
    path = played_path(traj, s, s + 1)

    # contain (default): the grafted band fits within the chosen parent cell
    gc = _grafted_tree(traj, owner, s, e, s + 1, res0, renormalise=False)
    bc = draw_icicle(screen, fonts, gc[0], rect, played_path=path,
                     graft_ids=gc[1], dim_ids=gc[2], band_sims=gc[3], renormalise=False)
    cell_c, sum_c = _graft_band_sum(bc, gc[1])
    assert abs(sum_c - cell_c) < 1.0                            # band contained in the parent cell

    # renormalise: the grafted band spans the full panel width and exceeds its parent cell
    gr = _grafted_tree(traj, owner, s, e, s + 1, res0, renormalise=True)
    gres, graft_ids, dim_ids, tip = gr
    assert dim_ids and all(not b_node_children(gres, i) for i in dim_ids)  # dropped subtrees -> childless
    br = draw_icicle(screen, fonts, gres, rect, played_path=path,
                     graft_ids=graft_ids, dim_ids=dim_ids, band_sims=tip, renormalise=True)
    cell_r, sum_r = _graft_band_sum(br, graft_ids)
    assert abs(sum_r - 600) < 1.0                               # band renormalised to the full width
    assert sum_r > cell_r + 1.0                                 # child band wider than its parent cell
    assert block_at(br, (br[0].x + 1, br[0].y + 1)) is not None
    pygame.display.quit()


def b_node_children(gres, node_id):
    """True if the node with ``node_id`` in the grafted tree has any children (helper for the dim test)."""
    stack = [gres.root]
    while stack:
        n = stack.pop()
        if id(n) == node_id:
            return bool(n.children)
        stack.extend(n.children.values())
    return False


def test_stack_target_cards_resolves_options_against_state_stack():
    # A stack-target decision node: every option resolves against the parent state's (public) stack.
    root = Node(None, None, 0)
    ca = Node(root, Action(ActionKind.CHOOSE_STACK_TARGET, target=0), 0)
    cb = Node(root, Action(ActionKind.CHOOSE_STACK_TARGET, target=1), 0)
    root.children = {ca.incoming_move: ca, cb.incoming_move: cb}
    st = types.SimpleNamespace(stack=(StackCard(card=5), StackCard(card=9)))    # stack ids at index 0,1
    out = _stack_target_cards(root, st)                          # (leaves need no legal_moves/apply)
    assert out == {id(ca): 5, id(cb): 9}                        # @0 -> card 5, @1 -> card 9 (one stack)
    assert _stack_target_cards(root, None) == {}                # no base state -> no-op
    assert _stack_target_cards(None, st) == {}                  # no tree -> no-op


def test_draw_icicle_stack_cards_overrides_label_and_color():
    pygame.display.init()
    screen = pygame.display.set_mode((800, 600))
    fonts = make_fonts()
    res = _searched()
    child = next(iter(res.root.children.values()))
    cid = 5                                                     # pretend this node took card id 5
    blocks = draw_icicle(screen, fonts, res, (0, 0, 800, 400), stack_cards={id(child): cid})
    b = next(b for b in blocks if b.node is child)
    assert b.label == _stack_target_label(cid)                 # e.g. "Soldier(5)", not "target@N"
    assert b.color == CARD_COLORS.get(card_name(cid), NEUTRAL)  # coloured by the resolved card
    # without the override the same node keeps its normal move label
    plain = draw_icicle(screen, fonts, res, (0, 0, 800, 400))
    assert next(x for x in plain if x.node is child).label != _stack_target_label(cid)
    pygame.display.quit()
