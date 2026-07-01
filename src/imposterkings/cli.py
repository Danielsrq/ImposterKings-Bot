"""Terminal driver and the human agent. Human I/O lives here, NOT in the engine package.

Run with::

    python -m imposterkings.cli --p0 human --p1 random --seed 0

Each seat is one of {human, random} for now (mcts is added in Phase 3). The HumanAgent prints the
information set and a numbered list of legal actions and reads a choice from stdin.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .agents import MCTSAgent, RandomAgent
from .arena import play_game
from .explain import format_action, format_infoset, format_pv_lines, format_search_result
from .infoset import InformationSet


class HumanAgent:
    """Reads a move from stdin. Lives here (not in agents.py) because it does I/O."""

    name = "human"

    def select_move(self, view: InformationSet, rng: np.random.Generator):
        print("\n" + format_infoset(view))
        moves = view.legal_moves()
        print("Legal moves:")
        for i, m in enumerate(moves, 1):
            print(f"  {i}. {format_action(m, view)}")
        while True:
            raw = input("Your move (number): ").strip()
            if raw.isdigit() and 1 <= int(raw) <= len(moves):
                return moves[int(raw) - 1]
            print("  invalid choice, try again.")


def make_agent(kind: str, iters: int):
    if kind == "human":
        return HumanAgent()
    if kind == "random":
        return RandomAgent()
    if kind == "mcts":
        return MCTSAgent(iterations=iters)
    raise SystemExit(f"unknown agent kind: {kind!r} (choose from human, random, mcts)")


def main(argv=None) -> None:
    # Best-effort UTF-8 console on Windows so card glyphs/labels render.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Play ImposterKings in the terminal.")
    parser.add_argument("--p0", default="human", choices=["human", "random", "mcts"])
    parser.add_argument("--p1", default="random", choices=["human", "random", "mcts"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1000, help="MCTS iterations per decision")
    parser.add_argument("--explain", action="store_true", help="print the MCTS candidate table")
    parser.add_argument("--start", type=int, default=None, choices=[0, 1],
                        help="force the starting player (default: coin flip)")
    args = parser.parse_args(argv)

    agents = [make_agent(args.p0, args.iters), make_agent(args.p1, args.iters)]
    rng = np.random.default_rng(args.seed)

    def on_decision(seat, view, move, agent, state):
        if agent.name != "human":  # echo bot moves so a human can follow along
            print(f"  P{seat} ({agent.name}) -> {format_action(move, view)}")
            if args.explain and getattr(agent, "last_result", None) is not None:
                print(format_search_result(agent.last_result, top=5))
                pv = format_pv_lines(agent.last_result)
                if pv:
                    print(pv)

    print(f"ImposterKings: P0={args.p0} vs P1={args.p1} (seed {args.seed})")
    winner, reward, final = play_game(agents, rng, on_decision=on_decision,
                                      starting_player=args.start)
    print(f"\n*** Player {winner} wins ***  (reward {reward})")


if __name__ == "__main__":
    main()
