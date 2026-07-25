"""For each recorded human game, replay the SAME deal with the bot in the human's seat and compare outcomes.

    python -m imposterkings.data_analysis.bot_on_seed --data datasets/human

For every seed you played, a standard bot (hybrid k=20 l=3 by default) plays that deal in your seat against
an identical bot opponent. The bot-vs-bot replay is fully deterministic given the deal seed (the search RNG
is seeded from the numpy seed -- rng.py), so ONE playout per seed is definitive; identical seeds are computed
once. It holds the DEAL fixed (same starting cards) but not the trajectory -- the opponent responds to the
bot's moves, not yours -- so it answers "from these cards in your seat, can a bot win?" versus "did you?".

Reports the two disagreements you care about:
  - you WON where the bot LOST   (you beat a deal the bot couldn't),
  - you LOST where the bot WON   (the bot could win it, you didn't),
plus the games where you agreed. Assisted games (Hint panel used) are excluded by default. The bot's NN
eval head defaults to whatever you faced (gen['bot']); --nn overrides it.
"""
from __future__ import annotations

import argparse
from typing import Optional

import numpy as np

from ..arena import play_game
from .budget_scaling import make_agent
from .replay import replay_all


def _evaluator(ckpt: Optional[str], cache: dict):
    if not ckpt:
        return None                                  # no NN head -> plain (rollout) hybrid MCTS
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
    p.add_argument("--k", type=int, default=20, help="hybrid budget k for the bot (standard = 20)")
    p.add_argument("--l", type=int, default=3, help="hybrid budget l for the bot (standard = 3)")
    p.add_argument("--nn", default=None,
                   help="checkpoint for the bot's NN eval head (default: whatever you faced, per gen['bot'])")
    p.add_argument("--include-assisted", action="store_true", help="include games where the Hint was used")
    p.add_argument("--limit", type=int, default=None, help="cap games processed")
    args = p.parse_args(argv)

    spec = ("hybrid", args.k, args.l)
    ev_cache: dict = {}
    seen: dict = {}                                  # (seed, starter, ckpt) -> bot winner (deterministic cache)
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
        if human_seat is None:
            skipped += 1
            continue
        ckpt = args.nn or (gen.get("bot") or {}).get("nn_ckpt")
        key = (g.seed, g.starting_player, ckpt)
        if key not in seen:
            seen[key] = bot_plays_seed(g.seed, g.starting_player, spec, _evaluator(ckpt, ev_cache))
        bot_winner = seen[key]

        you_won, bot_won = g.won(human_seat), bot_winner == human_seat
        went = "first" if human_seat == g.starting_player else "second"
        verdict = ("you>bot" if you_won and not bot_won else
                   "bot>you" if bot_won and not you_won else "agree")
        you_better += verdict == "you>bot"
        bot_better += verdict == "bot>you"
        agree += verdict == "agree"
        rows.append((g.seed, went, you_won, bot_won, verdict))
        n += 1
        if args.limit and n >= args.limit:
            break

    order = {"you>bot": 0, "bot>you": 1, "agree": 2}     # disagreements first, then agreements
    rows.sort(key=lambda r: (order[r[4]], r[0]))
    nn_lbl = args.nn or "plain hybrid (as faced)"
    print(f"\nBOT-ON-SEED   bot = hybrid-k{args.k}-l{args.l}, NN head = {nn_lbl}   ({n} games, {skipped} skipped)")
    print(f"  {'seed':>12}  {'you moved':>9}  {'you':>3}  {'bot':>3}  verdict")
    print(f"  {'-'*12}  {'-'*9}  {'-'*3}  {'-'*3}  {'-'*8}")
    for seed, went, yw, bw, verdict in rows:
        tag = "  <- you beat what the bot lost" if verdict == "you>bot" else \
              "  <- bot could win it, you lost" if verdict == "bot>you" else ""
        print(f"  {seed:>12}  {went:>9}  {'W' if yw else 'L':>3}  {'W' if bw else 'L':>3}  {verdict:<8}{tag}")
    if n:
        print(f"\n  you WON where the bot LOST : {you_better:>3}  ({you_better/n:.0%})")
        print(f"  you LOST where the bot WON : {bot_better:>3}  ({bot_better/n:.0%})")
        print(f"  agreed (both won / lost)   : {agree:>3}  ({agree/n:.0%})")
        print("\n  (replay any seed to study it:  python -m imposterkings.ui.app --seed <seed>)")


if __name__ == "__main__":
    main()
