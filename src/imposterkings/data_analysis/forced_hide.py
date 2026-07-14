"""Is the bot RIGHT never to hide the Assassin, or is it a search blind spot?

    python -m imposterkings.data_analysis.forced_hide --deals 150 --nn models/v3c_gen2/attn_d64_L2.pt

The observation (corpus_stats): across 4,000 setup decisions the search picked "hide the Assassin" as its
argmax **zero** times, and the netless rollout bot hides it in 0.4% of the hands that hold it. But the
threat it would dodge is real and large -- an Inquisitor rips the Assassin out of a hand in **20.3% of
games** -- and the corpus cannot say whether hiding is actually bad: it only ever happened 64 times, by
temperature accident, at 54.7% [42.6, 66.3] against a 59.5% baseline. That interval contains the baseline.
The data does not say hiding is bad. It says nobody ever tried it.

So this is an INTERVENTION, not another observation. We force the hide and play the game out.

THE DESIGN, and why each piece is there:

  * **Paired on the deal.** The same 18-card permutation is played twice -- once forced, once free. ~75% of
    outcome variance in this game is deal luck (the scaling study), so an unpaired A/B would drown a 5pp
    effect in noise. Here the deal is a controlled constant and only the hide differs.
  * **Mirrored on the starter.** ``deal(rng, starting_player=s)`` re-uses the same permutation with either
    seat moving first, so the +17.6pp second-player advantage cancels instead of leaking into the estimate
    (the Assassin-holder's seat and the coin-flip starter are independent draws, so without this the arms
    would not be balanced on tempo).
  * **Only the HIDE is forced.** The bot still chooses its own discard and every subsequent move, so what we
    measure is the hide and nothing else.
  * **The search still runs; only the played move is overridden** (the ``_TemperatureAgent`` trick). So every
    forced position is recorded WITH its real candidates/mean_q -- the corpus is trainable, and it also tells
    us what the bot *wanted* to do in a position it was not allowed to have.

Outcome measure: the win rate OF THE ASSASSIN-HOLDING SEAT, forced vs free, over the same deals.
"""
from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import cards, record
from ..actions import Action, ActionKind, StepKind
from ..arena import play_game
from ..record import DecisionRecord, GameRecord
from ..state import GameState
from .budget_scaling import make_agent, spec_label
from ..machine_learning.benchmark import _evaluator_for, parse_spec

ASSASSIN = cards.card_ids_for_name("Assassin")[0]      # 16 -- a singleton, so exactly one seat can hold it


class ForceHideAgent:
    """Play the wrapped agent's game, except: at SETUP_HIDE, hide ``card`` if it is legal to do so.

    The search STILL RUNS and ``last_result`` is still exposed, so the recorder captures the candidates the
    bot would have chosen from -- we learn what it wanted as well as what we made it do.
    """

    def __init__(self, inner, card: int) -> None:
        self.inner, self.card = inner, card
        self.last_result = None
        self.forced = 0

    @property
    def budget(self):
        return self.inner.budget

    @property
    def iterations(self):
        return self.inner.iterations

    def select_move(self, view, rng):
        move = self.inner.select_move(view, rng)         # run the real search first (-> last_result)
        self.last_result = self.inner.last_result
        if view.pending[-1].kind == StepKind.SETUP_HIDE:
            forced = Action(ActionKind.HIDE_CARD, card=self.card)
            if forced in view.legal_moves():
                self.forced += 1
                return forced
        return move


def assassin_seat(seed: int) -> Optional[int]:
    """Which seat was dealt the Assassin (None if it sat in the two leftover cards)."""
    st = GameState.deal(np.random.default_rng(seed))
    for s in (0, 1):
        if ASSASSIN in st.hands[s]:
            return s
    return None


def _one(seed: int, seat: int, sp: int, forced: bool, spec, ckpt: Optional[str]) -> Tuple[int, GameRecord]:
    """One game. ``seat`` holds the Assassin; ``sp`` starts; ``forced`` = that seat must hide the Assassin."""
    ev = _evaluator_for(ckpt) if ckpt else None
    agents = [make_agent(spec, evaluator=ev), make_agent(spec, evaluator=ev)]
    if forced:
        agents[seat] = ForceHideAgent(agents[seat], ASSASSIN)

    rec = GameRecord(deal_seed=seed, gen={
        "spec": spec_label(spec), "mode": spec[0], "k": spec[1], "l": spec[2] if len(spec) > 2 else None,
        "self_play": True, "temp_plies": 0, "value_ckpt": ckpt,
        # `paired_deals` tells token_dataset._guard_seed that the repeated deal_seed is DELIBERATE: four
        # games share one deal (2 arms x 2 starters), and grouping them under one game_id is exactly right --
        # they share an identical 18-card permutation, so splitting them across train/val would leak the
        # validation hands into training. The guard still rejects a seed repeated ACROSS corpora.
        "paired_deals": 4,
        "experiment": "assassin_hide", "arm": "forced" if forced else "free",
        "assassin_seat": seat, "forced_starting_player": sp,
    })

    def collect(s, view, move, agent, state):
        rec.decisions.append(DecisionRecord.build(s, view, move, agent))

    # Same play_rng across the two arms: they start identical and only diverge at the forced hide, which is
    # a mild common-random-numbers variance reduction on top of the deal pairing.
    winner, rewards, _ = play_game(agents, np.random.default_rng(seed), on_decision=collect,
                                   starting_player=sp, play_rng=np.random.default_rng([seed, sp]))
    rec.winner = winner
    rec.rewards = list(rewards)
    rec.starting_player = sp
    for d in rec.decisions:
        d.z = rewards[d.seat]
    return int(winner == seat), rec        # 1 iff the Assassin-holder won


def _chunk(seeds: List[int], spec, ckpt) -> Dict:
    import torch
    torch.set_num_threads(1)
    rows, recs = [], []
    for seed in seeds:
        seat = assassin_seat(seed)
        if seat is None:
            continue
        for sp in (0, 1):                                  # mirror the starter on the SAME permutation
            for forced in (True, False):
                won, rec = _one(seed, seat, sp, forced, spec, ckpt)
                rows.append({"seed": seed, "seat": seat, "starter": sp,
                             "arm": "forced" if forced else "free", "holder_won": won})
                recs.append(rec)
    return {"rows": rows, "recs": recs}


def _existing(out_dir: str):
    """Deals already on disk, and the next free shard index -- so a long run RESUMES instead of redoing
    work or clobbering it. (The 100-deal pilot cost 35 min; a 500-deal run must not be all-or-nothing.)"""
    import glob
    import json
    done, shards = set(), sorted(glob.glob(os.path.join(out_dir, "games_*.jsonl")))
    rows = []
    for f in shards:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                done.add(r["deal_seed"])
                g = r["gen"]
                rows.append({"seed": r["deal_seed"], "seat": g["assassin_seat"],
                             "starter": r["starting_player"], "arm": g["arm"],
                             "holder_won": int(r["winner"] == g["assassin_seat"])})
    return done, len(shards), rows


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Causal test: does hiding the Assassin at setup help or hurt?")
    p.add_argument("--deals", type=int, default=150,
                   help="TOTAL deals wanted (already-recorded ones count); each is 4 games")
    p.add_argument("--base-seed", type=int, default=50_000_000)
    p.add_argument("--spec", default="hybrid-k20-l3")
    p.add_argument("--nn", default=os.path.join("models", "v3c_gen2", "attn_d64_L2.pt"),
                   help="value/policy net for BOTH seats in BOTH arms (self-play; the only difference is "
                        "the forced hide)")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--chunk", type=int, default=5)
    p.add_argument("--out-dir", default=os.path.join("datasets", "forced_assassin_hide"))
    p.add_argument("--fresh", action="store_true", help="ignore what is on disk (default: RESUME)")
    args = p.parse_args(argv)

    spec = parse_spec(args.spec)
    done, shard0, old_rows = (set(), 0, []) if args.fresh else _existing(args.out_dir)
    if done:
        print(f"resuming: {len(done)} deals ({len(old_rows)} games) already recorded in {args.out_dir}/")

    # The Assassin sits in one of the two hands 16/18 of the time; the rest of the seeds are skipped.
    # Scan from base_seed so the seed->deal mapping is stable across runs and the new deals EXTEND the old.
    seeds, s = [], args.base_seed
    while len(seeds) + len(done) < args.deals:
        if s not in done and assassin_seat(s) is not None:
            seeds.append(s)
        s += 1

    if not seeds:
        print("  nothing to do -- the target is already on disk")
        report(old_rows)
        return

    print(f"forced-hide experiment: {len(seeds)} NEW deals -> {len(done) + len(seeds)} total\n"
          f"  {len(seeds) * 4} new games = {len(seeds)} deals x 2 starters x 2 arms (forced | free), "
          f"self-play {spec_label(spec)}, net={args.nn}")

    chunks = [seeds[i:i + args.chunk] for i in range(0, len(seeds), args.chunk)]
    t0 = time.perf_counter()
    from joblib import Parallel, delayed
    from tqdm import tqdm
    os.makedirs(args.out_dir, exist_ok=True)
    jobs = [delayed(_chunk)(c, spec, args.nn) for c in chunks]

    rows = list(old_rows)
    n_new = 0
    # Write each shard AS IT LANDS, not at the end: a crash 90 minutes in should cost one chunk, not the run.
    for i, o in enumerate(tqdm(Parallel(n_jobs=args.workers, return_as="generator")(jobs),
                               total=len(jobs), desc="chunks", unit="chunk")):
        record.write_jsonl(os.path.join(args.out_dir, f"games_{shard0 + i:05d}.jsonl"), o["recs"])
        rows.extend(o["rows"])
        n_new += len(o["recs"])
    print(f"\n  recorded {n_new} new games -> {args.out_dir}/  ({len(rows)} games total)")

    report(rows, time.perf_counter() - t0)


def report(rows: List[dict], wall: float = 0.0) -> Dict[str, float]:
    """Paired: for each (deal, starter) the SAME deal was played forced and free. Compare within the pair."""
    by: Dict[Tuple[int, int], Dict[str, int]] = defaultdict(dict)
    for r in rows:
        by[(r["seed"], r["starter"])][r["arm"]] = r["holder_won"]
    pairs = [v for v in by.values() if "forced" in v and "free" in v]

    nf = sum(v["forced"] for v in pairs)
    nl = sum(v["free"] for v in pairs)
    n = len(pairs)
    # The paired statistic: only the pairs where the two arms DISAGREE carry information (McNemar).
    win_only = sum(1 for v in pairs if v["forced"] == 1 and v["free"] == 0)
    lose_only = sum(1 for v in pairs if v["forced"] == 0 and v["free"] == 1)
    disagree = win_only + lose_only
    diff = (nf - nl) / n if n else 0.0
    se = (np.sqrt(disagree) / n) if disagree else 0.0             # McNemar's SE on the paired difference
    z = diff / se if se > 0 else 0.0

    print(f"\n{'=' * 78}\nDOES HIDING THE ASSASSIN HELP?   {n} paired deals x starters "
          f"({2 * n} games)\n{'=' * 78}")
    print(f"  Assassin-holder wins, FORCED to hide it : {nf}/{n} = {nf / n * 100:.1f}%")
    print(f"  Assassin-holder wins, FREE to choose    : {nl}/{n} = {nl / n * 100:.1f}%")
    print(f"  paired difference                       : {diff * 100:+.1f}pp  (SE {se * 100:.1f}, z {z:+.2f})")
    print(f"  pairs where the arms disagreed          : {disagree}/{n}  "
          f"(forced-only wins {win_only}, free-only wins {lose_only})")
    if abs(z) < 1.96:
        print("\n  => NOT SIGNIFICANT. Forcing the hide neither helps nor hurts measurably. The search's "
              "refusal\n     to hide the Assassin is not obviously WRONG -- but neither is it vindicated; "
              "on this evidence\n     the choice is close to free, and the bot's 0/4000 conviction is far "
              "stronger than the data\n     warrants.")
    elif diff > 0:
        print("\n  => HIDING THE ASSASSIN *WINS*. The bot has a genuine BLIND SPOT: it never once chose a "
              "move that\n     measurably improves its win rate. Likely cause -- the cost (no reaction to a "
              "king flip) is\n     immediate and visible to a shallow rollout, while the benefit (dodging "
              "the Inquisitor, which\n     steals the Assassin in 20.3% of games) is contingent and deep.")
    else:
        print("\n  => HIDING THE ASSASSIN *LOSES*. The search was right, and for a legible reason: a hidden "
              "Assassin\n     cannot react to the opponent's king flip -- which is exactly what it exists "
              "to punish, and what\n     ends 29.6% of all games.")
    if wall:
        print(f"\n  wall {wall:.0f}s")
    return {"n": n, "forced": nf / n if n else 0, "free": nl / n if n else 0, "diff": diff, "z": z}


if __name__ == "__main__":
    main()
