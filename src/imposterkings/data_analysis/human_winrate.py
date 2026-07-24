"""Human win-rate over a recorded human-vs-AI corpus, split by turn order -- the human counterpart to the
self-play TURN ORDER table, to see whether the ~59/41 second-player edge holds for human play.

    python -m imposterkings.data_analysis.human_winrate --data datasets/human

Reads games written by ui.app (gen['human']=True, gen['primary_seat']=the human's seat). Splits the human's
win-rate into moving FIRST vs SECOND with a Wilson 95% CI, next to the bot self-play baseline (P1 ~41% /
P2 ~59%). Assisted games (Hint panel used) are excluded by default.
"""
from __future__ import annotations

import argparse

from .corpus_stats import wilson
from .replay import replay_all

BOT_BASELINE = {"first": 0.41, "second": 0.59}     # from the 7,000-game self-play turn-order study


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Human win-rate (P1/P2) over recorded human-vs-AI games.")
    p.add_argument("--data", nargs="+", default=["datasets/human"], help="dir(s) of recorded human games")
    p.add_argument("--include-assisted", action="store_true", help="include games where the Hint was used")
    p.add_argument("--by-seat", action="store_true",
                   help="also print the seat-independent turn-order table (first/second-mover win-rate, "
                        "regardless of whether the human or the bot occupied the seat)")
    args = p.parse_args(argv)

    cells = {"first": [0, 0], "second": [0, 0]}        # [games, wins] for the human
    seat_cells = {"first": [0, 0], "second": [0, 0]}   # [games, wins] by TURN ORDER, any occupant
    assisted = skipped = 0
    for g in replay_all(args.data, None):
        gen = g.gen or {}
        if not gen.get("human"):
            continue
        if gen.get("hint_used"):
            assisted += 1
            if not args.include_assisted:
                continue
        seat = gen.get("primary_seat")
        if seat is None:
            skipped += 1
            continue
        went = "first" if seat == g.starting_player else "second"
        cells[went][0] += 1
        cells[went][1] += int(g.won(seat))
        first_won = g.won(g.starting_player)           # winner by TURN ORDER (regardless of human/bot)
        seat_cells["first"][0] += 1
        seat_cells["first"][1] += int(first_won)
        seat_cells["second"][0] += 1
        seat_cells["second"][1] += int(not first_won)

    total_n = cells["first"][0] + cells["second"][0]
    total_w = cells["first"][1] + cells["second"][1]
    print(f"\nHUMAN WIN RATE  ({total_n} games; {assisted} assisted "
          f"{'included' if args.include_assisted else 'excluded'}, {skipped} skipped)")
    print(f"  {'went':>7} {'games':>6} {'wins':>5} {'winrate':>8}  {'95% CI':>15}   bot baseline")
    for went in ("first", "second"):
        n, w = cells[went]
        if not n:
            print(f"  {went:>7} {n:>6} {w:>5} {'--':>8}")
            continue
        pt, lo, hi = wilson(w, n)
        print(f"  {went:>7} {n:>6} {w:>5} {pt*100:>7.1f}%  [{lo*100:>4.1f}, {hi*100:>4.1f}]%   "
              f"P{'1' if went=='first' else '2'} ~{BOT_BASELINE[went]*100:.0f}%")
    if total_n:
        pt, lo, hi = wilson(total_w, total_n)
        print(f"  {'overall':>7} {total_n:>6} {total_w:>5} {pt*100:>7.1f}%  [{lo*100:>4.1f}, {hi*100:>4.1f}]%")
        print("\n(Human moves first in some deals and second in others; compare each seat to its own baseline "
              "-- a human 'better than the bot' shows as a P1 rate above ~41% or a P2 rate above ~59%.)")

    if args.by_seat and total_n:
        print(f"\nTURN ORDER (seat-independent -- winner by first/second mover, any occupant)")
        print(f"  {'moved':>7} {'games':>6} {'wins':>5} {'winrate':>8}  {'95% CI':>15}   bot baseline")
        for went in ("first", "second"):
            n, w = seat_cells[went]
            pt, lo, hi = wilson(w, n)
            print(f"  {went:>7} {n:>6} {w:>5} {pt*100:>7.1f}%  [{lo*100:>4.1f}, {hi*100:>4.1f}]%   "
                  f"P{'1' if went=='first' else '2'} ~{BOT_BASELINE[went]*100:.0f}%")


if __name__ == "__main__":
    main()
