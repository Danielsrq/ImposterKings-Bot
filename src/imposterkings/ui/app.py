"""PyGame app loop. Run with::

    python -m imposterkings.ui.app --p1 mcts --iters 800

The human plays one seat by clicking action buttons; the other seat is a random or MCTS bot that
moves automatically. The board is always drawn from the human's perspective.
"""
from __future__ import annotations

import argparse
from collections import deque

import numpy as np

from .. import cards
from ..actions import ActionKind, StepKind
from ..agents import MCTSAgent, RandomAgent
from ..explain import format_action
from ..state import GameState
from .render import BTN_H, BTN_TOP, PANEL_X, WINDOW, make_fonts, render_frame
from .review import PlyRecord, annotate_dual_evals, run_review, _result_eval, _search_from

# Opponent's setup hide/discard are private -- never reveal the card identity in the log.
_PRIVATE_OPP_STEPS = (StepKind.SETUP_HIDE, StepKind.SETUP_DISCARD)


def _make_bot(kind: str, iters: int):
    return MCTSAgent(iterations=iters) if kind == "mcts" else RandomAgent()


def _hover_index(mouse, n_legal: int):
    mx, my = mouse
    if mx < PANEL_X + 12:
        return None
    idx = (my - BTN_TOP) // BTN_H
    return idx if 0 <= idx < n_legal else None


def _describe(seat: int, view, move, human_seat: int, state) -> str:
    who = "You" if seat == human_seat else "Opp"
    if seat != human_seat and view.pending and view.pending[-1].kind in _PRIVATE_OPP_STEPS:
        return f"{who}: {view.pending[-1].kind.name.lower()} (hidden)"
    line = f"{who}: {format_action(move, view)}"
    if move.kind == ActionKind.GUESS_CARD:
        # The guesser is `seat`; correctness is revealed by the game, so surface it in the log.
        defender = 1 - seat
        correct = any(cards.card_name(c) == move.name for c in state.hands[defender])
        line += "  -> CORRECT" if correct else "  -> wrong"
    return line


def run(p1: str = "mcts", iters: int = 800, seed=None, human_seat: int = 0, start=None,
        hint_iters=None) -> None:
    import pygame  # local import so the engine/tests never require pygame

    pygame.init()
    screen = pygame.display.set_mode(WINDOW)
    clock = pygame.time.Clock()
    fonts = make_fonts()
    bot_seat = 1 - human_seat
    log: deque = deque(maxlen=40)
    show_reasoning, show_hint = True, False
    analysis_iters = iters if hint_iters is None else hint_iters
    analysis_rng = np.random.default_rng(1234567)   # dedicated so analysis never perturbs the game rng
    # Per-state dual analysis: BOTH seats' read of the current position (keyed by state identity), so the
    # two side panels are live every turn -- like ui.review. Each entry is a SearchResult (or None).
    analysis: dict = {"state": None, 0: None, 1: None}
    knowledge_cache: dict = {"state": None, "val": None}   # [seat -> (has, lacks, level)], per state
    trajectory: list = []                          # per-ply PlyRecord(seat, move, view, result) for review
    game: dict = {}  # holds the resettable per-game state: state, rng, seed, bot

    def new_game(new_seed=None):
        s = int(np.random.default_rng().integers(0, 2**31)) if new_seed is None else new_seed
        rng = np.random.default_rng(s)
        game.update(seed=s, rng=rng, state=GameState.deal(rng, starting_player=start),
                    bot=_make_bot(p1, iters))
        log.clear()
        trajectory.clear()
        analysis["state"], analysis[0], analysis[1] = None, None, None
        knowledge_cache["state"], knowledge_cache["val"] = None, None
        pygame.display.set_caption(f"ImposterKings  (seed {s})")
        print(f"ImposterKings  (deck seed {s} -- pass --seed {s} to replay this deal)")

    new_game(seed)

    def apply_logged(seat, move, result=None):
        st = game["state"]
        log.append(_describe(seat, st.information_set(human_seat), move, human_seat, st))
        rec = PlyRecord(seat, move, st.information_set(seat), result, state=st)
        # Keep the live dual-analysis already computed for this position so the post-game review reuses it
        # instead of recomputing (it only re-searches seats whose panel was off during play).
        if analysis["state"] is st and (analysis[0] is not None or analysis[1] is not None):
            rec.result_by_seat = (analysis[0], analysis[1])
        trajectory.append(rec)
        game["state"] = st.apply(move)

    running = True
    while running:
        state, bot = game["state"], game["bot"]
        view = state.information_set(human_seat)
        human_turn = (not state.is_terminal()) and state.to_play == human_seat
        legal = view.legal_moves() if human_turn else []

        # Auto-resolve ONLY a choiceless reaction window (a King's-Hand/Assassin prompt when you hold
        # no reaction card -> the sole option is to decline). Every real decision, including a forced
        # last card, is left for you to click.
        if human_turn and len(legal) == 1 and legal[0].kind == ActionKind.DECLINE_REACTION:
            apply_logged(human_seat, legal[0])
            clock.tick(60)
            continue

        hover = _hover_index(pygame.mouse.get_pos(), len(legal)) if human_turn else None
        if state.is_terminal():
            status = f"GAME OVER - Player {state.winner} wins"
        elif not human_turn:
            status = f"{bot.name} (seat {bot_seat}) is thinking..."
        else:
            status = "Your move - click an action"

        # Both seats' read of the CURRENT position, once per state; each gated (and lazily filled when its
        # toggle flips on) by its panel toggle so cost is opt-in. bot seat -> reasoning, human -> hint.
        if not state.is_terminal():
            if game["state"] is not analysis["state"]:
                analysis["state"], analysis[0], analysis[1] = game["state"], None, None
            for s in (0, 1):
                shown = show_reasoning if s == bot_seat else show_hint
                if shown and analysis[s] is None:
                    analysis[s] = _search_from(state, s, analysis_iters, analysis_rng)
        bot_res, hint_res = analysis[bot_seat], analysis[human_seat]
        bot_eval = _result_eval(bot_res, state, bot_seat) if bot_res is not None else None
        hint_eval = _result_eval(hint_res, state, human_seat) if hint_res is not None else None

        # Hand-knowledge for both seats (recomputed once per state).
        if game["state"] is not knowledge_cache["state"]:
            kn = []
            for s in (0, 1):
                v = game["state"].information_set(s)
                kn.append((v.opp_hand_has, v.opp_hand_lacks, v.knowledge_level()))
            knowledge_cache["state"], knowledge_cache["val"] = game["state"], kn

        frame = render_frame(screen, view, fonts, legal, hover=hover, status=status,
                             log=list(log), bot_result=bot_res,
                             show_reasoning=show_reasoning, seed=game["seed"],
                             hint_result=hint_res, show_hint=show_hint,
                             knowledge=knowledge_cache["val"],
                             bot_eval=bot_eval, hint_eval=hint_eval)
        pygame.display.flip()

        review_requested = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                if frame.new_game.collidepoint(pos):            # clickable any time
                    new_game()
                elif frame.review and frame.review.collidepoint(pos):
                    review_requested = True
                elif frame.reasoning_toggle and frame.reasoning_toggle.collidepoint(pos):
                    show_reasoning = not show_reasoning
                elif frame.hint_toggle and frame.hint_toggle.collidepoint(pos):
                    show_hint = not show_hint
                elif human_turn:
                    for rect, move in frame.buttons:
                        if rect.collidepoint(pos):
                            hres = analysis[human_seat] if analysis["state"] is game["state"] else None
                            apply_logged(human_seat, move, hres)
                            break

        # Deferred so the review's own event loop doesn't run mid-iteration of this one. Fill any per-turn
        # evals NOT already computed live (e.g. a seat whose panel was off) so the review shows the full
        # dual-eval graph + dual icicles; turns analyzed live during play are reused, not recomputed.
        if review_requested and trajectory:
            annotated = list(trajectory)
            n = annotate_dual_evals(annotated, analysis_iters, np.random.default_rng(7))
            if n:
                print(f"computing {n} dual-eval searches for turns not analyzed live during play...")
            run_review(screen, fonts, annotated)

        if (not game["state"].is_terminal()) and game["state"].to_play == bot_seat:
            pygame.time.delay(300)
            bview = game["state"].information_set(bot_seat)
            mv = game["bot"].select_move(bview, game["rng"])
            apply_logged(bot_seat, mv, getattr(game["bot"], "last_result", None))

        clock.tick(30)

    pygame.quit()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Play ImposterKings in a PyGame window.")
    parser.add_argument("--p1", default="mcts", choices=["random", "mcts"], help="opponent bot")
    parser.add_argument("--iters", type=int, default=800, help="MCTS iterations per decision")
    parser.add_argument("--seed", type=int, default=None,
                        help="fix the deck/deal (default: random each launch)")
    parser.add_argument("--human-seat", type=int, default=0, choices=[0, 1])
    parser.add_argument("--start", type=int, default=None, choices=[0, 1])
    parser.add_argument("--hint-iters", type=int, default=None,
                        help="MCTS iterations for the dual-eval side panels + review analysis "
                             "(default: same as --iters)")
    args = parser.parse_args(argv)
    run(p1=args.p1, iters=args.iters, seed=args.seed, human_seat=args.human_seat, start=args.start,
        hint_iters=args.hint_iters)


if __name__ == "__main__":
    main()
