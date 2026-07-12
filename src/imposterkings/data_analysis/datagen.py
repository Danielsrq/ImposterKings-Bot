"""Self-play dataset generation: play MCTS-vs-MCTS games and log replayable `GameRecord`s to JSONL.

    python -m imposterkings.data_analysis.datagen --games 2000 --k 20 --l 3 --workers 10
    python -m imposterkings.data_analysis.datagen --games 100 --k 20 --temp-plies 6   # explore openings

Each game is one JSONL line (see `record.GameRecord`): a replayable header (`deal_seed` + gen meta) plus a
`DecisionRecord` per ply carrying the MCTS candidate stats (visits / mean_q / visit_share → the policy and
value targets) and the back-filled terminal reward `z`. Because the log stores the deal seed and the full
ordered action list, every game reconstructs exactly via `GameState.deal(default_rng(deal_seed))` + replay
(`state.apply` is deterministic) — so the same file is both the training corpus AND the `ui.review --replay`
source. See `DATASET.md`. Chunked joblib parallelism: each worker writes its own shard file.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from typing import Dict, List, Tuple

import numpy as np

from ..arena import play_game
from ..record import DecisionRecord, GameRecord, write_jsonl
from .budget_scaling import make_agent, spec_label

Spec = Tuple


class _TemperatureAgent:
    """Wrap an MCTS agent so its first ``temp_plies`` real decisions PLAY a move sampled from the visit
    distribution (temperature ``temp``) instead of the argmax -- for opening state-coverage. The true
    search still runs and is recorded (``last_result``); only the *played* move is randomized."""

    def __init__(self, inner, temp_plies: int, temp: float = 1.0) -> None:
        self.inner = inner
        self.temp_plies = temp_plies
        self.temp = temp
        self._n = 0
        self.last_result = None

    # on_decision / cost hooks read these off the agent
    @property
    def budget(self):
        return self.inner.budget

    @property
    def iterations(self):
        return self.inner.iterations

    def select_move(self, view, rng):
        move = self.inner.select_move(view, rng)     # runs the search -> inner.last_result
        self.last_result = res = self.inner.last_result
        if res is not None:                          # a real (searched) decision
            if self._n < self.temp_plies and len(res.stats) > 1:
                w = np.array([s.visits for s in res.stats], dtype=float)
                if self.temp != 1.0:
                    w = w ** (1.0 / self.temp)
                move = res.stats[int(rng.choice(len(res.stats), p=w / w.sum()))].move
            self._n += 1
        return move


def _gen_meta(spec: Spec, temp_plies: int, base_seed: int, value_ckpt=None) -> Dict:
    return {"spec": spec_label(spec), "mode": spec[0], "k": spec[1],
            "l": spec[2] if len(spec) > 2 else None,
            "temp_plies": temp_plies, "self_play": True, "base_seed": base_seed,
            "value_ckpt": value_ckpt}


def collect_game(spec: Spec, seed: int, temp_plies: int, base_seed: int,
                 evaluator=None, value_ckpt=None) -> GameRecord:
    """Play one self-play game from deal ``seed``; return a replayable, target-carrying GameRecord.

    ``evaluator`` (from a checkpoint) turns the self-play agents into NN-MCTS (PUCT); ``value_ckpt`` is
    its path, recorded in the gen meta for provenance."""
    def mk():
        a = make_agent(spec, evaluator=evaluator)
        return _TemperatureAgent(a, temp_plies) if temp_plies > 0 else a

    rec = GameRecord(gen=_gen_meta(spec, temp_plies, base_seed, value_ckpt), deal_seed=seed)

    def collect(seat, view, move, agent, state):
        rec.decisions.append(DecisionRecord.build(seat, view, move, agent))

    winner, rewards, term = play_game([mk(), mk()], np.random.default_rng(seed), on_decision=collect)
    rec.winner = winner
    rec.rewards = list(rewards)
    rec.starting_player = term.starting_player
    for d in rec.decisions:
        d.z = rewards[d.seat]
    return rec


def _existing_state(out_dir: str) -> Tuple[int, int, Dict]:
    """Resume bookkeeping for a corpus dir: (games already generated, next free shard index, the newest
    shard's ``gen`` header). Counts records by JSONL lines; shard index parsed from ``games_NNNNN.jsonl``."""
    files = sorted(glob.glob(os.path.join(out_dir, "games_*.jsonl")))
    if not files:
        return 0, 0, {}
    n = 0
    for f in files:
        with open(f, encoding="utf-8") as fh:
            n += sum(1 for line in fh if line.strip())
    last_idx = max(int(re.search(r"games_(\d+)\.jsonl$", f).group(1)) for f in files)
    with open(files[-1], encoding="utf-8") as fh:
        first = json.loads(fh.readline())
    return n, last_idx + 1, first.get("gen", {})


def _chunk_task(spec: Spec, seeds: List[int], out_dir: str, shard_idx: int,
                temp_plies: int, base_seed: int, value_ckpt=None) -> Dict:
    t0 = time.perf_counter()
    ev = None
    if value_ckpt:                                    # NN-MCTS self-play: build the evaluator once/worker
        import torch
        torch.set_num_threads(1)                      # attention evals run torch; avoid oversubscription
        from ..machine_learning.benchmark import _evaluator_for
        ev = _evaluator_for(value_ckpt)               # checkpoint-type dispatch: MLP or attention
    recs = [collect_game(spec, s, temp_plies, base_seed, evaluator=ev, value_ckpt=value_ckpt)
            for s in seeds]
    path = os.path.join(out_dir, f"games_{shard_idx:05d}.jsonl")
    write_jsonl(path, recs)
    return {"count": len(recs), "seconds": time.perf_counter() - t0, "path": path}


def run(spec: Spec, games: int, workers: int, chunk: int, base_seed: int,
        temp_plies: int, out_dir: str, value_ckpt=None,
        seed_offset: int = 0, shard_offset: int = 0) -> List[Dict]:
    """``seed_offset``/``shard_offset`` support resumable corpora: seeds continue at
    ``base_seed + seed_offset`` and shard files at ``games_{shard_offset:05d}.jsonl`` (the recorded ``gen``
    meta keeps the ORIGINAL base_seed so later resumes can verify against it)."""
    from joblib import Parallel, delayed
    from tqdm import tqdm

    seeds = [base_seed + seed_offset + i for i in range(games)]
    chunks = [seeds[i:i + chunk] for i in range(0, games, chunk)]
    jobs = [delayed(_chunk_task)(spec, ch, out_dir, shard_offset + idx, temp_plies, base_seed, value_ckpt)
            for idx, ch in enumerate(chunks)]
    return list(tqdm(Parallel(n_jobs=workers, return_as="generator")(jobs),
                     total=len(chunks), desc="shards", unit="shard"))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Self-play dataset generation -> replayable JSONL shards.")
    p.add_argument("--games", type=int, default=2000)
    p.add_argument("--mode", choices=["hybrid", "branching", "fixed"], default="hybrid")
    p.add_argument("--k", type=int, default=20, help="budget multiplier (hybrid/branching) or N (fixed)")
    p.add_argument("--l", type=int, default=3)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--chunk", type=int, default=25, help="games per shard / parallel job")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--temp-plies", type=int, default=0,
                   help="sample the played move for the first N decisions/agent (0 = greedy self-play)")
    p.add_argument("--out-dir", default=os.path.join("datasets", "selfplay_k20l3"))
    p.add_argument("--force", action="store_true", help="write into a non-empty --out-dir")
    p.add_argument("--resume", action="store_true",
                   help="continue an existing corpus: seeds resume after the games already generated, "
                        "shards continue at the next free index (flags must match the corpus)")
    p.add_argument("--value-checkpoint", default=None,
                   help="NN checkpoint -> NN-MCTS (PUCT) self-play instead of rollout MCTS")
    args = p.parse_args(argv)

    spec: Spec = ("fixed", args.k) if args.mode == "fixed" else (args.mode, args.k, args.l)
    seed_offset = shard_offset = 0
    if args.resume:
        n0, shard_offset, prev = _existing_state(args.out_dir)
        if n0 == 0:
            p.error(f"--resume: no existing shards found in {args.out_dir}")
        cur = _gen_meta(spec, args.temp_plies, args.base_seed, args.value_checkpoint)
        for key in ("spec", "temp_plies", "base_seed", "value_ckpt"):
            if prev.get(key) != cur[key]:
                p.error(f"--resume mismatch on {key!r}: corpus has {prev.get(key)!r}, "
                        f"flags give {cur[key]!r} -- refusing to mix engines/seeds in one corpus")
        seed_offset = n0
        print(f"resume: {n0} games exist -> seeds continue at {args.base_seed + n0}, "
              f"shards at games_{shard_offset:05d}.jsonl")
    elif os.path.isdir(args.out_dir) and any(os.scandir(args.out_dir)) and not args.force:
        p.error(f"{args.out_dir} exists and is non-empty; use --resume to continue it, "
                f"--force to mix, or a different --out-dir")
    os.makedirs(args.out_dir, exist_ok=True)

    n_shards = (args.games + args.chunk - 1) // args.chunk
    engine = f"NN-MCTS(PUCT) <- {args.value_checkpoint}" if args.value_checkpoint else "rollout MCTS"
    print(f"datagen  spec={spec_label(spec)}  engine={engine}  games={args.games}  "
          f"temp_plies={args.temp_plies}  chunk={args.chunk} -> {n_shards} shards  "
          f"workers={args.workers}  base_seed={args.base_seed}\n  -> {args.out_dir}")

    t0 = time.perf_counter()
    results = run(spec, args.games, args.workers, args.chunk, args.base_seed,
                  args.temp_plies, args.out_dir, value_ckpt=args.value_checkpoint,
                  seed_offset=seed_offset, shard_offset=shard_offset)
    wall = time.perf_counter() - t0

    games = sum(r["count"] for r in results)
    compute = sum(r["seconds"] for r in results)
    print(f"\nwrote {games} games in {len(results)} shards -> {args.out_dir}")
    print(f"timing: {compute:.0f}s compute (single-core) / {wall:.0f}s wall "
          f"(speedup {compute / wall:.1f}x on {args.workers} workers) = {compute / games:.2f}s/game")


if __name__ == "__main__":
    main()
