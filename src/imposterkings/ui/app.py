"""PyGame app loop. Run with::

    python -m imposterkings.ui.app --p1 mcts --iters 800

The human plays one seat by clicking action buttons; the other seat is a random or MCTS bot that
moves automatically. The board is always drawn from the human's perspective.
"""
from __future__ import annotations

import argparse
from collections import deque

import numpy as np

from .. import budget as budget_mod
from .. import cards
from ..actions import ActionKind, StepKind
from ..agents import MCTSAgent, RandomAgent
from ..explain import format_action
from ..state import GameState
from .render import BTN_H, BTN_TOP, PANEL_X, WINDOW, draw_settings_overlay, make_fonts, render_frame
from .review import PlyRecord, annotate_dual_evals, budget_iters, run_review, _result_eval, _search_from
from .scenario_setup import run_setup

# Opponent's setup hide/discard are private -- never reveal the card identity in the log.
_PRIVATE_OPP_STEPS = (StepKind.SETUP_HIDE, StepKind.SETUP_DISCARD)


def _engine_budget(engine: dict):
    """A budget policy for the current engine config (drives BOTH the bot and the analysis)."""
    if engine["mode"] == "hybrid":
        return budget_mod.hybrid(engine["k"], engine["l"])
    if engine["mode"] == "branching":
        return budget_mod.branching(engine["k"], engine["l"])
    return budget_mod.fixed(engine["N"])   # "mcts": fixed N (l/k unused)


def _make_bot(random_bot: bool, engine: dict, nn_agent=None):
    if nn_agent is not None:                          # a loaded NN checkpoint takes the bot seat
        return nn_agent
    return RandomAgent() if random_bot else MCTSAgent(budget=_engine_budget(engine))


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
        k: int = 100, l: int = 3, setup: bool = False, nn: str = None) -> None:
    import pygame  # local import so the engine/tests never require pygame

    nn_agent = None
    if nn:                                          # lazy import so torch stays optional for normal play
        from ..machine_learning.agent import NNAgent
        nn_agent = NNAgent.from_checkpoint(nn)
        print(f"NN bot loaded from {nn}")

    pygame.init()
    screen = pygame.display.set_mode(WINDOW)
    clock = pygame.time.Clock()
    fonts = make_fonts()
    bot_seat = 1 - human_seat
    hotseat = False                                 # scenario Hotseat: the human drives BOTH sides
    log: deque = deque(maxlen=40)
    show_reasoning, show_hint = True, False
    # One engine config drives BOTH the bot and the dual-eval analysis (so the panels' sims match the bot).
    engine = {"mode": p1 if p1 in ("mcts", "branching", "hybrid") else "mcts",
              "N": iters, "k": k, "l": l}
    random_bot = (p1 == "random")
    engine_budget = _engine_budget(engine)          # rebuilt by apply_engine() on any settings change
    settings_open, dragging = False, None      # dragging = the active slider key ("N"/"k"/"l") or None
    analysis_rng = np.random.default_rng(1234567)   # dedicated so analysis never perturbs the game rng
    # Per-state dual analysis: BOTH seats' read of the current position (keyed by state identity), so the
    # two side panels are live every turn -- like ui.review. Each entry is a SearchResult (or None).
    analysis: dict = {"state": None, 0: None, 1: None}
    knowledge_cache: dict = {"state": None, "val": None}   # [seat -> (has, lacks, level)], per state
    trajectory: list = []                          # per-ply PlyRecord(seat, move, view, result) for review
    game: dict = {}  # holds the resettable per-game state: state, rng, seed, bot

    def apply_engine():
        """Rebuild the budget + bot from ``engine`` and re-analyze the current position (live retune)."""
        nonlocal engine_budget
        engine_budget = _engine_budget(engine)
        game["bot"] = _make_bot(random_bot, engine, nn_agent)
        analysis["state"], analysis[0], analysis[1] = None, None, None

    def new_game(new_seed=None, initial_state=None):
        s = int(np.random.default_rng().integers(0, 2**31)) if new_seed is None else new_seed
        rng = np.random.default_rng(s)
        st0 = initial_state if initial_state is not None else GameState.deal(rng, starting_player=start)
        game.update(seed=s, rng=rng, state=st0, bot=_make_bot(random_bot, engine, nn_agent))
        log.clear()
        trajectory.clear()
        analysis["state"], analysis[0], analysis[1] = None, None, None
        knowledge_cache["state"], knowledge_cache["val"] = None, None
        cap = "scenario" if initial_state is not None else f"seed {s}"
        pygame.display.set_caption(f"ImposterKings  ({cap})")
        if initial_state is None:
            print(f"ImposterKings  (deck seed {s} -- pass --seed {s} to replay this deal)")

    def open_setup():
        nonlocal human_seat, bot_seat, hotseat
        cfg = run_setup(screen, fonts, human_seat=human_seat)
        if cfg is not None:
            human_seat, bot_seat, hotseat = cfg["human_seat"], 1 - cfg["human_seat"], cfg["hotseat"]
            new_game(initial_state=cfg["state"])

    new_game(seed)
    if setup:
        open_setup()

    def apply_logged(seat, move, result=None, perspective=None):
        st = game["state"]
        persp = human_seat if perspective is None else perspective
        log.append(_describe(seat, st.information_set(persp), move, persp, st))
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
        # Hotseat: the human drives BOTH sides, so the view + turn follow whoever is to move.
        act = state.to_play
        view_seat = act if hotseat else human_seat
        view = state.information_set(view_seat)
        human_turn = (not state.is_terminal()) and (hotseat or act == human_seat)
        legal = view.legal_moves() if human_turn else []

        # Auto-resolve ONLY a choiceless reaction window (a King's-Hand/Assassin prompt when you hold
        # no reaction card -> the sole option is to decline). Every real decision, including a forced
        # last card, is left for you to click.
        if (not settings_open) and human_turn and len(legal) == 1 \
                and legal[0].kind == ActionKind.DECLINE_REACTION:
            apply_logged(act, legal[0], perspective=view_seat)
            clock.tick(60)
            continue

        hover = _hover_index(pygame.mouse.get_pos(), len(legal)) if human_turn else None
        if state.is_terminal():
            status = f"GAME OVER - Player {state.winner} wins"
        elif not human_turn:
            status = f"{bot.name} (seat {bot_seat}) is thinking..."
        elif hotseat:
            status = f"Hotseat - P{act} to move (click an action)"
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
                    its = budget_iters(state, s, engine_budget)
                    analysis[s] = _search_from(state, s, its, analysis_rng)
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

        mouse = pygame.mouse.get_pos()
        frame = render_frame(screen, view, fonts, legal, hover=hover, status=status,
                             log=list(log), bot_result=bot_res,
                             show_reasoning=show_reasoning, seed=game["seed"],
                             hint_result=hint_res, show_hint=show_hint,
                             knowledge=knowledge_cache["val"],
                             bot_eval=bot_eval, hint_eval=hint_eval)
        controls = draw_settings_overlay(screen, fonts, engine, mouse) if settings_open else None
        pygame.display.flip()

        def _slider_at(pos):                        # the modal slider whose (padded) track contains pos
            for sl in (controls["sliders"] if controls else []):
                if sl[0].inflate(24, 28).collidepoint(pos):
                    return sl
            return None

        def _slider_set(key, px):                   # map x -> N/k/l (rounded, clamped to the slider range)
            for track, lo, hi, k_ in (controls["sliders"] if controls else []):
                if k_ == key:
                    frac = max(0.0, min(1.0, (px - track.x) / track.w))
                    engine[key] = int(round(lo + frac * (hi - lo)))
                    return

        review_requested = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                if settings_open:
                    settings_open = False
                else:
                    running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s and not settings_open:
                open_setup()                            # S -> build a custom position
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if dragging is not None:            # end of a slider drag -> apply the new value
                    dragging = None
                    apply_engine()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                if settings_open:                   # modal: route clicks to it, ignore the board
                    sl = _slider_at(pos)
                    if controls["close"].collidepoint(pos) or frame.settings.collidepoint(pos):
                        settings_open = False
                    elif sl is not None:
                        dragging = sl[3]
                        _slider_set(dragging, pos[0])
                    else:
                        for mode, r in controls["pills"].items():
                            if r.collidepoint(pos):
                                engine["mode"], random_bot = mode, False
                                apply_engine()
                                break
                elif frame.settings.collidepoint(pos):
                    settings_open = True
                elif frame.scenario.collidepoint(pos):          # build a custom position
                    open_setup()
                elif frame.new_game.collidepoint(pos):          # clickable any time
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
                            hres = analysis[view_seat] if analysis["state"] is game["state"] else None
                            apply_logged(act, move, hres, perspective=view_seat)
                            break
        if dragging is not None:                    # live-drag the active slider while the button is held
            if pygame.mouse.get_pressed()[0] and controls is not None:
                _slider_set(dragging, mouse[0])
            else:
                dragging = None

        # Deferred so the review's own event loop doesn't run mid-iteration of this one. Fill any per-turn
        # evals NOT already computed live (e.g. a seat whose panel was off) so the review shows the full
        # dual-eval graph + dual icicles; turns analyzed live during play are reused, not recomputed.
        if review_requested and trajectory:
            annotated = list(trajectory)
            n = annotate_dual_evals(annotated, engine_budget, np.random.default_rng(7))
            if n:
                print(f"computing {n} dual-eval searches (at the current engine budget) for turns not "
                      f"analyzed live during play...")
            run_review(screen, fonts, annotated)

        if (not settings_open) and (not hotseat) and (not game["state"].is_terminal()) \
                and game["state"].to_play == bot_seat:
            pygame.time.delay(300)
            bview = game["state"].information_set(bot_seat)
            mv = game["bot"].select_move(bview, game["rng"])
            apply_logged(bot_seat, mv, getattr(game["bot"], "last_result", None))

        clock.tick(30)

    pygame.quit()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Play ImposterKings in a PyGame window.")
    parser.add_argument("--p1", default="mcts", choices=["random", "mcts", "hybrid", "branching"],
                        help="opponent bot (mcts=fixed --iters; hybrid/branching scale budget by --k)")
    parser.add_argument("--iters", type=int, default=800, help="MCTS iterations per decision (mcts mode)")
    parser.add_argument("--k", type=int, default=100,
                        help="budget multiplier for the hybrid/branching bot modes")
    parser.add_argument("--l", type=int, default=3,
                        help="effective legal-moves for a sub-decision card at selection (branch/hybrid)")
    parser.add_argument("--seed", type=int, default=None,
                        help="fix the deck/deal (default: random each launch)")
    parser.add_argument("--human-seat", type=int, default=0, choices=[0, 1])
    parser.add_argument("--start", type=int, default=None, choices=[0, 1])
    parser.add_argument("--setup", action="store_true",
                        help="open the scenario-setup screen first (also the in-game 'Scenario' button / S)")
    parser.add_argument("--nn", default=None,
                        help="seat a trained NN checkpoint as the bot (e.g. models/mlp_32.pt)")
    args = parser.parse_args(argv)
    run(p1=args.p1, iters=args.iters, seed=args.seed, human_seat=args.human_seat, start=args.start,
        k=args.k, l=args.l, setup=args.setup, nn=args.nn)


if __name__ == "__main__":
    main()
