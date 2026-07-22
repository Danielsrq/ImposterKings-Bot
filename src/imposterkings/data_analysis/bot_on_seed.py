"""For each recorded human game, replay the SAME deal with the bot in the human's seat and compare outcomes.

    python -m imposterkings.data_analysis.bot_on_seed --data datasets/human

The bot-vs-bot replay is fully deterministic given the deal seed (the search RNG is seeded from the numpy
seed -- rng.py), so ONE playout per seed is definitive; identical seeds are computed once. The comparison
holds the DEAL fixed (same starting cards) but not the trajectory -- the opponent responds to the bot's
moves, not yours -- so it answers "from these cards at your seat, can a bot beat a bot?" versus "did you?".

Buckets each seed into:
  - you WON where the bot LOST   (you outplayed it on that deal),
  - you LOST where the bot WON   (it outplayed you),
  - agree (both won / both lost).
Assisted games (Hint panel used) are excluded by default.
"""
from __future__ import annotations

import argparse
from typing import Optional

import numpy as np

from ..arena import play_game
from .budget_scaling import make_agent
from .replay import replay_all


def _spec_and_ckpt(bot: dict):
    """Map a recorded bot config (gen['bot']) to a (spec, nn_ckpt) pair for make_agent / _evaluator_for."""
    mode, N, k, l = bot.get("mode"), bot.get("N"), bot.get("k"), bot.get("l")
    if mode == "mcts":
        spec = ("fixed", N)
    elif mode == "branching":
        spec = ("branching", k, l)
    else:                                        # "hybrid" and "nn" both search the hybrid budget
        spec = ("hybrid", k, l)
    nn_ckpt = bot.get("nn_ckpt") if mode == "nn" else None
    return spec, nn_ckpt


def _evaluator(ckpt: Optional[str], cache: dict):
    if not ckpt:
        return None
    if ckpt not in cache:
        from ..machine_learning.benchmark import _evaluator_for
        cache[ckpt] = _evaluator_for(ckpt)
    return cache[ckpt]


def bot_plays_seed(seed: int, starting_player: Optional[int], spec, evaluator) -> int:
    """Deterministic bot-vs-bot game on this deal. Returns the winning seat."""
    agents = [make_agent(spec, evaluator=evaluator), make_agent(spec, evaluator=evaluator)]
    winner, _rewards, _term = play_game(agents, np.random.default_rng(seed), starting_player=starting_player)
    return winner


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Bot replays each recorded human deal; compare who wins.")
    p.add_argument("--data", nargs="+", default=["datasets/human"], help="dir(s) of recorded human games")
    p.add_argument("--include-assisted", action="store_true", help="include games where the Hint was used")
    p.add_argument("--limit", type=int, default=None, help="cap games processed")
    args = p.parse_args(argv)

    ev_cache: dict = {}
    seen: dict = {}                              # (seed, starter, spec, ckpt) -> bot winner (dedupe/cache)
    rows, you_better, bot_better, agree, skipped = [], 0, 0, 0, 0
    n = 0
    for g in replay_all(args.data, None):
        gen = g.gen or {}
        if not gen.get("human"):
            continue
        if gen.get("hint_used") and not args.include_assisted:
            skipped += 1
            continue
        human_seat = gen.get("primary_seat")
        bot_cfg = gen.get("bot") or {}
        if human_seat is None or not bot_cfg:
            skipped += 1
            continue
        spec, ckpt = _spec_and_ckpt(bot_cfg)
        key = (g.seed, g.starting_player, spec, ckpt)
        if key not in seen:
            seen[key] = bot_plays_seed(g.seed, g.starting_player, spec, _evaluator(ckpt, ev_cache))
        bot_winner = seen[key]

        you_won = g.won(human_seat)
        bot_won = bot_winner == human_seat
        verdict = ("you>bot" if you_won and not bot_won else
                   "bot>you" if bot_won and not you_won else "agree")
        you_better += verdict == "you>bot"
        bot_better += verdict == "bot>you"
        agree += verdict == "agree"
        rows.append((g.seed, human_seat, you_won, bot_won, verdict, bool(gen.get("hint_used"))))
        n += 1
        if args.limit and n >= args.limit:
            break

    print(f"\n{'seed':>12} {'seat':>4} {'you':>4} {'bot':>4}  verdict   {'assisted':>8}")
    for seed, seat, yw, bw, verdict, assisted in rows:
        print(f"{seed:>12} {'P'+str(seat):>4} {'W' if yw else 'L':>4} {'W' if bw else 'L':>4}  "
              f"{verdict:<8}  {'yes' if assisted else '':>8}")
    print(f"\n{n} games compared ({skipped} skipped: assisted/hotseat/scenario/incomplete meta).")
    if n:
        print(f"  you WON where the bot LOST : {you_better:>4}  ({you_better/n:.0%})  -- you outplayed it")
        print(f"  you LOST where the bot WON : {bot_better:>4}  ({bot_better/n:.0%})  -- it outplayed you")
        print(f"  agreed (both won/both lost): {agree:>4}  ({agree/n:.0%})")


if __name__ == "__main__":
    main()
