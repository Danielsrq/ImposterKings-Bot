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

from ..actions import Action
from ..infoset import InformationSet
from .render import (BG, BTN, BTN_HOVER, DIVIDER, GOLD, INK, MUTE, P_COLORS, PANEL, WINDOW,
                     _compact_action, _text, make_fonts)
from .tree_view import draw_icicle, draw_outline

HEADER_H = 92


@dataclass
class PlyRecord:
    seat: int
    move: Action
    view: InformationSet
    result: object            # Optional[SearchResult] (with .root); None for a forced/no-search move


def build_trajectory(iters: int, seed: Optional[int], start: Optional[int] = None) -> List[PlyRecord]:
    """Play one MCTS-vs-MCTS game and record (seat, move, pre-move view, SearchResult) per ply."""
    from ..agents import MCTSAgent
    from ..arena import play_game

    traj: List[PlyRecord] = []

    def collect(seat, view, move, agent, state):
        traj.append(PlyRecord(seat, move, view, getattr(agent, "last_result", None)))

    rng = np.random.default_rng(seed)
    play_game([MCTSAgent(iterations=iters), MCTSAgent(iterations=iters)], rng,
              on_decision=collect, starting_player=start)
    return traj


def panels_for_cursor(traj: List[PlyRecord], cursor: int) -> Tuple[Optional[int], Optional[int]]:
    """Indices of the latest seat-0 and seat-1 decisions at or before ``cursor`` (each panel shows
    that seat's most-recent decision)."""
    p0 = p1 = None
    for i in range(min(cursor, len(traj) - 1), -1, -1):
        if traj[i].seat == 0 and p0 is None:
            p0 = i
        if traj[i].seat == 1 and p1 is None:
            p1 = i
        if p0 is not None and p1 is not None:
            break
    return p0, p1


def _button(surface, font, label, rect, *, active=False, hover=False):
    import pygame
    r = pygame.Rect(rect)
    pygame.draw.rect(surface, GOLD if active else (BTN_HOVER if hover else BTN), r, border_radius=4)
    surface.blit(font.render(label, True, (20, 20, 20) if active else INK),
                 (r.x + 8, r.y + 4))
    return r


def _draw_panel(surface, fonts, traj, seat, idx, tree_rect, mode, ost):
    """Draw one seat's panel (title + tree). Returns the outline hitmap (empty for icicle)."""
    med, small = fonts["med"], fonts["small"]
    tx, ty, tw, th = tree_rect
    if idx is None:
        _text(surface, med, f"P{seat}: (no decision yet)", (tx + 4, ty - 24), MUTE)
        return []
    rec = traj[idx]
    _text(surface, med, f"P{seat} (mcts)  —  played {_compact_action(rec.move)}",
          (tx + 4, ty - 24), P_COLORS[seat])
    if mode == "icicle":
        draw_icicle(surface, fonts, rec.result, tree_rect, played_move=rec.move)
        return []
    return draw_outline(surface, fonts, rec.result, tree_rect,
                        expanded=ost["exp"], scroll=ost["scroll"], played_move=rec.move)


def run_review(screen, fonts, traj: List[PlyRecord]) -> None:
    import pygame
    if not traj:
        return
    clock = pygame.time.Clock()
    W, H = WINDOW
    med, small = fonts["med"], fonts["small"]
    cursor, mode = 0, "icicle"
    ost = {0: {"exp": set(), "scroll": 0}, 1: {"exp": set(), "scroll": 0}}
    mid = W // 2

    running = True
    while running:
        screen.fill(PANEL)
        mouse = pygame.mouse.get_pos()

        # --- header -------------------------------------------------------------------
        rec = traj[cursor]
        _text(screen, med, "Post-game review  (MCTS vs MCTS)", (12, 8), INK)
        _text(screen, small, f"Ply {cursor + 1}/{len(traj)}   —   P{rec.seat} (mcts) played "
                             f"{_compact_action(rec.move)}", (12, 36), GOLD)
        btns = {
            "prev": _button(screen, small, "◄ Prev", (12, 58, 84, 26),
                            hover=pygame.Rect(12, 58, 84, 26).collidepoint(mouse)),
            "next": _button(screen, small, "Next ►", (102, 58, 84, 26),
                            hover=pygame.Rect(102, 58, 84, 26).collidepoint(mouse)),
            "outline": _button(screen, small, "Outline", (214, 58, 84, 26), active=(mode == "outline")),
            "icicle": _button(screen, small, "Icicle", (304, 58, 84, 26), active=(mode == "icicle")),
        }
        _text(screen, small, "◄/► step  ·  I/O view  ·  wheel scroll  ·  Esc quit", (410, 63), MUTE)
        pygame.draw.line(screen, DIVIDER, (0, HEADER_H), (W, HEADER_H))
        pygame.draw.line(screen, DIVIDER, (mid, HEADER_H + 4), (mid, H))

        # --- two panels ---------------------------------------------------------------
        p0i, p1i = panels_for_cursor(traj, cursor)
        top = HEADER_H + 30
        lrect = (6, top, mid - 12, H - top - 6)
        rrect = (mid + 6, top, W - mid - 12, H - top - 6)
        lhits = _draw_panel(screen, fonts, traj, 0, p0i, lrect, mode, ost[0])
        rhits = _draw_panel(screen, fonts, traj, 1, p1i, rrect, mode, ost[1])
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
            elif e.type == pygame.MOUSEWHEEL and mode == "outline":
                seat = 0 if pygame.mouse.get_pos()[0] < mid else 1
                ost[seat]["scroll"] = max(0, ost[seat]["scroll"] - e.y)
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
                elif mode == "outline":
                    for hitmap, seat in ((lhits, 0), (rhits, 1)):
                        for row_rect, path in hitmap:
                            if row_rect.collidepoint(pos):
                                ost[seat]["exp"] ^= {path}
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
