"""Post-game review screen: step through a completed MCTS-vs-MCTS game and inspect each decision's
search tree as a ply-banded icicle (or a collapsible outline), with P0's and P1's trees side by side.

    python -m imposterkings.ui.review --iters 800 --seed 0

The reviewed game is generated headlessly (both seats MCTS, so every ply has a real recorded tree),
then the window opens on the in-memory trajectory. Left panel = P0's most-recent decision, right =
P1's; each is drawn from its own seat's perspective (+ = good for that player).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .. import cards
from ..actions import Action, ActionKind, StepKind
from ..infoset import InformationSet
from .render import (BTN, BTN_HOVER, CARD_COLORS, DIVIDER, GOLD, INK, MUTE, P_COLORS, PANEL,
                     WINDOW, _compact_action, _text, make_fonts)
from . import assets
from .tree_view import block_at, draw_crown, draw_icicle, draw_outline, draw_tooltip

HEADER_H = 76        # title + ply line + button row
# Timeline band: one combined eval graph (both players' lines, shared x & y axes) then the card strip.
TL_TOP = 84
GRAPH_H = 130
GRAPH_GAP = 6
STRIP_H = 62
X0, PAD = 8, 12      # left edge + inner padding for the shared time axis
STRIP_TOP = TL_TOP + GRAPH_H + GRAPH_GAP


def _turn_x(i: int, n: int) -> float:
    """Shared x for turn index ``i`` of ``n`` turns — used by both graphs and the strip so they align."""
    w = WINDOW[0] - 16
    return X0 + PAD + (w - 2 * PAD) * (i / max(1, n - 1))


@dataclass
class PlyRecord:
    seat: int
    move: Action
    view: InformationSet
    result: object            # Optional[SearchResult] (with .root); None for a forced/no-search move
    state: object = None      # the full pre-move GameState (for the board view)
    eval_by_seat: Optional[Tuple[float, float]] = None  # each seat's own-perspective eval at a turn start
    result_by_seat: Optional[Tuple[object, object]] = None  # each seat's SearchResult of this turn's position


def _search_from(state, observer: int, iters: int, rng):
    """Run a search from seat ``observer``'s information set of ``state`` and return the SearchResult
    (its ``info.observer`` is ``observer``, so the icicle draws it in that seat's perspective)."""
    from ..mcts import SearchConfig, search
    return search(state.information_set(observer), SearchConfig(rng=rng, iterations=iters))


def _result_eval(res, state, observer: int) -> float:
    """A SearchResult's value from ``observer``'s perspective (+1 = good for observer). ``root_value()`` is
    the *mover's* perspective, so it is negated when the observer is not to move (zero-sum leaves)."""
    rv = res.root_value()
    return rv if state.to_play == observer else -rv


def build_trajectory(iters: int, seed: Optional[int], start: Optional[int] = None,
                     cross_eval: bool = True) -> List[PlyRecord]:
    """Play one MCTS-vs-MCTS game and record (seat, move, pre-move view, SearchResult, state) per ply.

    With ``cross_eval`` (default), also fill each turn-start ply's ``eval_by_seat`` with BOTH players'
    own-perspective evals of that position -- the mover's from its own search, the opponent's from one
    extra search of the same state -- so the review graph can show every player's read on every turn."""
    from ..agents import MCTSAgent
    from ..arena import play_game

    traj: List[PlyRecord] = []

    def collect(seat, view, move, agent, state):
        traj.append(PlyRecord(seat, move, view, getattr(agent, "last_result", None), state))

    rng = np.random.default_rng(seed)
    # evaluate_forced: search even on forced turns (ascensions, sole reactions) so every turn -- not just
    # ones with a real choice -- carries an eval. A position's value is well-defined even when forced.
    agents = [MCTSAgent(iterations=iters, evaluate_forced=True),
              MCTSAgent(iterations=iters, evaluate_forced=True)]
    play_game(agents, rng, on_decision=collect, starting_player=start)

    if cross_eval:
        xrng = np.random.default_rng(seed)
        for start_i, _end, _owner in turns_of(traj):
            rec = traj[start_i]
            if rec.state is None:
                continue
            mover = rec.state.to_play                 # the seat that searched here (== owner except in setup)
            other = 1 - mover
            mover_res = rec.result if rec.result is not None else _search_from(rec.state, mover, iters, xrng)
            other_res = _search_from(rec.state, other, iters, xrng)   # the opponent's read of this position
            rbs, eb = [None, None], [0.0, 0.0]
            rbs[mover], rbs[other] = mover_res, other_res
            eb[mover] = _result_eval(mover_res, rec.state, mover)
            eb[other] = _result_eval(other_res, rec.state, other)
            rec.result_by_seat = (rbs[0], rbs[1])
            rec.eval_by_seat = (eb[0], eb[1])
    return traj


_SETUP_KINDS = (StepKind.SETUP_HIDE, StepKind.SETUP_DISCARD)


def _owner_key(rec: PlyRecord) -> int:
    """Who owns this ply's turn. Normally ``turn_player`` (constant through a whole turn incl. the
    opponent's reactions). During SETUP there is no real turn order -- ``turn_player`` stays on the
    starting player while BOTH seats hide/discard -- so attribute setup plies to the actual mover."""
    pending = getattr(rec.view, "pending", None)
    if pending and pending[-1].kind in _SETUP_KINDS:
        return rec.seat
    return rec.view.turn_player


def turns_of(traj: List[PlyRecord]) -> List[Tuple[int, int, int]]:
    """Group plies into turns: maximal runs with the same owner (``turn_player`` normally; the mover
    during setup, which has no real turn order). Returns ``[(start, end, owner), ...]``."""
    out: List[Tuple[int, int, int]] = []
    i = 0
    while i < len(traj):
        owner = _owner_key(traj[i])
        j = i
        while j + 1 < len(traj) and _owner_key(traj[j + 1]) == owner:
            j += 1
        out.append((i, j, owner))
        i = j + 1
    return out


def turn_for_seat(turns, seat: int, cursor: int) -> Optional[Tuple[int, int]]:
    """The latest turn OWNED by ``seat`` whose start is at/before ``cursor`` -> (start, end)."""
    best = None
    for s, e, owner in turns:
        if owner == seat and s <= cursor:
            best = (s, e)
    return best


def played_path(traj: List[PlyRecord], start: int, upto: int) -> List[Action]:
    """The moves actually taken in a turn, from its first ply up to ``upto`` (the played line)."""
    return [traj[j].move for j in range(start, min(upto, len(traj) - 1) + 1)]


def _button(surface, font, label, rect, *, active=False, hover=False):
    import pygame
    r = pygame.Rect(rect)
    pygame.draw.rect(surface, GOLD if active else (BTN_HOVER if hover else BTN), r, border_radius=4)
    surface.blit(font.render(label, True, (20, 20, 20) if active else INK),
                 (r.x + 8, r.y + 4))
    return r


def _current_turn(turns, cursor) -> int:
    """Index into ``turns`` of the turn containing ``cursor`` (0 if none)."""
    return next((i for i, (s, e, o) in enumerate(turns) if s <= cursor <= e), 0)


def _turn_eval(rec, owner: int, seat: int) -> Optional[float]:
    """Seat ``seat``'s own-perspective eval at a turn start. Uses the both-players ``eval_by_seat`` when
    present; otherwise falls back to the mover's search (owner side only)."""
    if rec.eval_by_seat is not None:
        return rec.eval_by_seat[seat]
    if seat == owner and rec.result is not None:
        return rec.result.root_value()
    return None


def _draw_graph(surface, fonts, traj, turns, cursor, top):
    """The combined eval graph: BOTH players' lines on one set of shared axes (x = turns, y = -1..+1).
    With cross-evals every turn carries both seats' reads, so each line has a point on EVERY turn (a
    seat's line moves the turn right after it learns something). Each line stays in its own owner's
    perspective (+1 = good for that player), so the two lines together show the P0/P1 asymmetry."""
    import pygame
    small = fonts["small"]
    x0, w, h = X0, WINDOW[0] - 16, GRAPH_H
    mid = top + h / 2
    pygame.draw.rect(surface, (24, 26, 32), (x0, top, w, h))
    pygame.draw.line(surface, DIVIDER, (x0, mid), (x0 + w, mid))
    _text(surface, small, "eval  +1 / 0 / -1", (x0 + 4, top - 1), MUTE)
    _text(surface, small, "P0", (x0 + w - 74, top - 1), P_COLORS[0])
    _text(surface, small, "P1", (x0 + w - 44, top - 1), P_COLORS[1])
    n = len(turns)
    if n == 0:
        return
    ypx = lambda v: mid - (h / 2 - 8) * max(-1.0, min(1.0, v))
    for seat in (0, 1):                                # both lines on the same axes
        prev = None
        for i, (s, e, owner) in enumerate(turns):
            v = _turn_eval(traj[s], owner, seat)
            if v is None:                              # no eval for this seat here -> break the line
                prev = None
                continue
            x, y = _turn_x(i, n), ypx(v)
            if prev is not None:
                pygame.draw.line(surface, P_COLORS[seat], prev, (x, y), 2)
            pygame.draw.circle(surface, P_COLORS[seat], (int(x), int(y)), 3)
            prev = (x, y)
    cx = int(_turn_x(_current_turn(turns, cursor), n))
    pygame.draw.line(surface, GOLD, (cx, top), (cx, top + h), 1)


def _headline_card(traj, start, end):
    """The card the turn's owner PLAYED (the headline of that turn), or None (e.g. a king flip / setup)."""
    for j in range(start, end + 1):
        mv = traj[j].move
        if mv.kind == ActionKind.PLAY_CARD and mv.card is not None:
            return mv.card
    return None


def _turn_is_flip(traj, start, end) -> bool:
    """True if this turn was a king flip (its headline is FLIP_KING, not a card play)."""
    return any(traj[j].move.kind == ActionKind.FLIP_KING for j in range(start, end + 1))


def _draw_strip(surface, fonts, traj, turns, cursor):
    """Card replay strip: each turn's headline card as mini art, left->right, aligned to ``_turn_x`` so
    the columns line up with both graphs above. Current turn is raised + gold-bordered. Returns
    ``[(rect, turn_start)]`` for click-to-jump."""
    import pygame
    small = fonts["small"]
    n = len(turns)
    if n == 0:
        return []
    spacing = (WINDOW[0] - 16 - 2 * PAD) / max(1, n - 1)
    ch = STRIP_H - 12                                  # leave room for the raised current card + border
    cw = min(int(round(ch / 1.4)), int(max(12, spacing - 2)))
    ch = int(round(cw * 1.4))
    cur = _current_turn(turns, cursor)
    hits = []
    for i, (s, e, owner) in enumerate(turns):
        cx = _turn_x(i, n)
        is_cur = i == cur
        y = STRIP_TOP + (0 if is_cur else 6)
        rect = pygame.Rect(int(cx - cw / 2), int(y), cw, ch)
        card = _headline_card(traj, s, e)
        if card is not None:
            try:
                surface.blit(assets.card_surface(card, (cw, ch)), rect)
            except Exception:                          # missing art -> solid card-colored tile
                pygame.draw.rect(surface, CARD_COLORS.get(cards.card_name(card), MUTE), rect)
        elif _turn_is_flip(traj, s, e):                # king flip -> white cell + upside-down crown
            pygame.draw.rect(surface, (245, 245, 245), rect)
            draw_crown(surface, rect, flipped=True)
        else:                                          # first play (setup hide/discard) -> white + upright crown
            pygame.draw.rect(surface, (245, 245, 245), rect)
            draw_crown(surface, rect, flipped=False)
        pygame.draw.rect(surface, P_COLORS[owner], rect, 2)   # always the player's color (blue P0 / orange P1)
        if is_cur:                                            # selection: white ring (not gold, which reads as P1)
            pygame.draw.rect(surface, INK, rect.inflate(6, 6), 2)
        hits.append((rect, s))
    return hits


def _draw_panel(surface, fonts, traj, seat, turn, cursor, tree_rect, mode, ost, zoom_stack, last_tree):
    """Draw ``seat``'s read of the CURRENT turn's position. Both panels show the same turn: the mover's
    real search on its side, the opponent's search of the same state on the other -- so P1's tree updates
    during P0's turns too. The actually-played line is highlighted in both. Returns icicle blocks."""
    med, small = fonts["med"], fonts["small"]
    tx, ty = tree_rect[0], tree_rect[1]
    if turn is None:
        _text(surface, med, f"P{seat}: (no turn yet)", (tx + 4, ty - 42), MUTE)
        return []
    start, end, owner = turn
    rec0 = traj[start]
    is_mover = seat == owner
    path = played_path(traj, start, min(cursor, end))
    # this seat's SearchResult of the current position (mover reuses its own; opponent uses the cross search)
    res = (rec0.result_by_seat[seat] if rec0.result_by_seat is not None
           else (rec0.result if is_mover else None))
    if res is None:                                           # this seat has no search here -> persist dimmed
        head = (f"P{seat} — forced move: {_compact_action(traj[min(cursor, end)].move)}" if is_mover
                else f"P{seat} — (no read available)")
        _text(surface, med, head, (tx + 4, ty - 42), MUTE)
        prev = last_tree[seat]
        if prev is not None and mode == "icicle":
            draw_icicle(surface, fonts, prev[0], tree_rect, played_path=prev[1], dim=True)
        else:
            _text(surface, small, "(no search)", (tx + 4, ty + 6), MUTE)
        return []
    last_tree[seat] = (res, path)
    zoom_note = "  [zoomed — Backspace out]" if zoom_stack else ""
    forced_note = "  · forced" if len(res.stats) == 1 else ""
    header = (f"P{seat} — {_compact_action(rec0.move)}{forced_note}{zoom_note}  ◄ active" if is_mover
              else f"P{seat} — reads P{owner}'s turn{zoom_note}")
    _text(surface, med, header, (tx + 4, ty - 42), P_COLORS[seat])
    if rec0.eval_by_seat is not None:                         # this seat's eval of the current position
        _text(surface, small, f"P{seat} eval {rec0.eval_by_seat[seat]:+.2f} (their perspective)",
              (tx + 4, ty - 22), MUTE)
    if mode == "icicle":
        return draw_icicle(surface, fonts, res, tree_rect, played_path=path,
                           zoom_root=(zoom_stack[-1] if zoom_stack else None))
    return draw_outline(surface, fonts, res, tree_rect, expanded=ost["exp"],
                        scroll=ost["scroll"], played_move=(path[-1] if path else None))


def run_review(screen, fonts, traj: List[PlyRecord]) -> None:
    import pygame
    if not traj:
        return
    clock = pygame.time.Clock()
    W, H = WINDOW
    med, small = fonts["med"], fonts["small"]
    cursor, mode = 0, "icicle"
    ost = {0: {"exp": set(), "scroll": 0}, 1: {"exp": set(), "scroll": 0}}
    zoom = {0: [], 1: []}                       # per-panel node zoom stacks
    last_tree = {0: None, 1: None}              # last non-None (result, played_path) per seat
    mid = W // 2
    turns = turns_of(traj)

    running = True
    while running:
        screen.fill(PANEL)
        mouse = pygame.mouse.get_pos()
        rec = traj[cursor]
        pend = getattr(rec.view, "pending", None)
        is_setup = bool(pend) and pend[-1].kind in _SETUP_KINDS
        turn_label = "Setup" if is_setup else f"turn P{turns[_current_turn(turns, cursor)][2]}"

        _text(screen, med, "Post-game review  (MCTS vs MCTS)", (12, 6), INK)
        _text(screen, small, f"Ply {cursor + 1}/{len(traj)}  —  {turn_label}  ·  P{rec.seat} played "
                             f"{_compact_action(rec.move)}", (12, 32), GOLD)
        specs = [("prev", "◄ Prev", 12), ("next", "Next ►", 102), ("outline", "Outline", 210),
                 ("icicle", "Icicle", 300), ("zout", "⬆ Zoom out", 390)]
        btns = {}
        for key, label, bx in specs:
            r = pygame.Rect(bx, 50, 84 if key != "zout" else 96, 22)
            btns[key] = _button(screen, small, label, r, active=(key == mode), hover=r.collidepoint(mouse))
        _text(screen, small, "◄/► step · I/O view · click card/graph=jump · click box=zoom · "
                             "Backspace=out · Esc quit", (500, 54), MUTE)

        _draw_graph(screen, fonts, traj, turns, cursor, TL_TOP)
        strip_hits = _draw_strip(screen, fonts, traj, turns, cursor)
        ptop = STRIP_TOP + STRIP_H + 10
        pygame.draw.line(screen, DIVIDER, (mid, ptop - 4), (mid, H))

        top = ptop + 44                         # room for the panel header + eval subtitle above the icicle
        lrect = (6, top, mid - 12, H - top - 6)
        rrect = (mid + 6, top, W - mid - 12, H - top - 6)
        cur_turn = turns[_current_turn(turns, cursor)]        # both panels show the SAME (current) turn
        blocks = {
            0: _draw_panel(screen, fonts, traj, 0, cur_turn, cursor, lrect, mode,
                           ost[0], zoom[0], last_tree),
            1: _draw_panel(screen, fonts, traj, 1, cur_turn, cursor, rrect, mode,
                           ost[1], zoom[1], last_tree),
        }
        if mode == "icicle":                    # hover tooltip
            hseat = 0 if mouse[0] < mid else 1
            hb = block_at(blocks[hseat], mouse)
            if hb is not None:
                draw_tooltip(screen, fonts, hb, mouse)
        pygame.display.flip()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key == pygame.K_LEFT:
                    cursor = max(0, cursor - 1)
                elif e.key == pygame.K_RIGHT:
                    cursor = min(len(traj) - 1, cursor + 1)
                elif e.key == pygame.K_o:
                    mode = "outline"
                elif e.key == pygame.K_i:
                    mode = "icicle"
                elif e.key == pygame.K_BACKSPACE:
                    for z in zoom.values():
                        if z:
                            z.pop()
            elif e.type == pygame.MOUSEWHEEL and mode == "outline":
                ost[0 if mouse[0] < mid else 1]["scroll"] = max(
                    0, ost[0 if mouse[0] < mid else 1]["scroll"] - e.y)
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 3:   # right-click zooms out
                zseat = 0 if e.pos[0] < mid else 1
                if zoom[zseat]:
                    zoom[zseat].pop()
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                pos = e.pos
                if btns["prev"].collidepoint(pos):
                    cursor = max(0, cursor - 1)
                elif btns["next"].collidepoint(pos):
                    cursor = min(len(traj) - 1, cursor + 1)
                elif btns["outline"].collidepoint(pos):
                    mode = "outline"
                elif btns["icicle"].collidepoint(pos):
                    mode = "icicle"
                elif btns["zout"].collidepoint(pos):
                    for z in zoom.values():
                        if z:
                            z.pop()
                elif TL_TOP <= pos[1] < ptop - 6 and turns:   # timeline (graphs + card strip) -> jump
                    hit = next((s for r, s in strip_hits if r.collidepoint(pos)), None)
                    if hit is not None:
                        cursor = hit
                    else:
                        i = min(range(len(turns)), key=lambda k: abs(_turn_x(k, len(turns)) - pos[0]))
                        cursor = turns[i][0]
                elif mode == "icicle":                                # click a box -> zoom in
                    zseat = 0 if pos[0] < mid else 1
                    hb = block_at(blocks[zseat], pos)
                    if hb is not None and hb.node.children:
                        zoom[zseat].append(hb.node)
                elif mode == "outline":                               # click a row -> expand/collapse
                    for seat_, hitmap in blocks.items():
                        for row_rect, pth in hitmap:
                            if row_rect.collidepoint(pos):
                                ost[seat_]["exp"] ^= {pth}
                                break
        clock.tick(30)


def main(argv=None) -> None:
    import pygame

    p = argparse.ArgumentParser(description="Post-game review of an MCTS-vs-MCTS game.")
    p.add_argument("--iters", type=int, default=800, help="MCTS iterations per decision")
    p.add_argument("--seed", type=int, default=0, help="deck/deal seed")
    p.add_argument("--start", type=int, default=None, choices=[0, 1], help="force the starting player")
    args = p.parse_args(argv)

    print(f"Generating MCTS-vs-MCTS game (iters={args.iters}, seed={args.seed})...")
    traj = build_trajectory(args.iters, args.seed, args.start)
    print(f"  {len(traj)} decisions recorded. Opening review window...")

    pygame.init()
    screen = pygame.display.set_mode(WINDOW)
    pygame.display.set_caption(f"ImposterKings review  (seed {args.seed}, {args.iters} sims)")
    run_review(screen, make_fonts(), traj)
    pygame.quit()


if __name__ == "__main__":
    main()
