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
    hint_agent = MCTSAgent(iterations=iters if hint_iters is None else hint_iters)
    hint_rng = np.random.default_rng(1234567)     # dedicated so hints don't perturb the game rng
    hint: dict = {"state": None, "result": None}  # cached hint search, keyed by state identity
    game: dict = {}  # holds the resettable per-game state: state, rng, seed, bot

    def new_game(new_seed=None):
        s = int(np.random.default_rng().integers(0, 2**31)) if new_seed is None else new_seed
        rng = np.random.default_rng(s)
        game.update(seed=s, rng=rng, state=GameState.deal(rng, starting_player=start),
                    bot=_make_bot(p1, iters))
        log.clear()
        hint["state"], hint["result"] = None, None
        pygame.display.set_caption(f"ImposterKings  (seed {s})")
        print(f"ImposterKings  (deck seed {s} -- pass --seed {s} to replay this deal)")

    new_game(seed)

    def apply_logged(seat, move):
        view_before = game["state"].information_set(human_seat)
        log.append(_describe(seat, view_before, move, human_seat, game["state"]))
        game["state"] = game["state"].apply(move)

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

        # Compute the hint once per human decision, and only while it's toggled on.
        if show_hint and human_turn and game["state"] is not hint["state"]:
            hint_agent.select_move(view, hint_rng)
            hint["state"], hint["result"] = game["state"], hint_agent.last_result
        hint_result = hint["result"] if (show_hint and human_turn) else None

        frame = render_frame(screen, view, fonts, legal, hover=hover, status=status,
                             log=list(log), bot_result=getattr(bot, "last_result", None),
                             show_reasoning=show_reasoning, seed=game["seed"],
                             hint_result=hint_result, show_hint=show_hint)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                if frame.new_game.collidepoint(pos):            # clickable any time
                    new_game()
                elif frame.reasoning_toggle and frame.reasoning_toggle.collidepoint(pos):
                    show_reasoning = not show_reasoning
                elif frame.hint_toggle and frame.hint_toggle.collidepoint(pos):
                    show_hint = not show_hint
                elif human_turn:
                    for rect, move in frame.buttons:
                        if rect.collidepoint(pos):
                            apply_logged(human_seat, move)
                            break

        if (not game["state"].is_terminal()) and game["state"].to_play == bot_seat:
            pygame.time.delay(300)
            bview = game["state"].information_set(bot_seat)
            apply_logged(bot_seat, game["bot"].select_move(bview, game["rng"]))

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
                        help="MCTS iterations for the 'Your hint' panel (default: same as --iters)")
    args = parser.parse_args(argv)
    run(p1=args.p1, iters=args.iters, seed=args.seed, human_seat=args.human_seat, start=args.start,
        hint_iters=args.hint_iters)


if __name__ == "__main__":
    main()
