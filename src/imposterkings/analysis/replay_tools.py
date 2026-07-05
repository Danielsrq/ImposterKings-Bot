"""Convert between a multi-game dataset shard and single-game replay files (same `GameRecord` format).

    python -m imposterkings.analysis.replay_tools split datasets/selfplay_k20l3/games_00000.jsonl \
        --out-dir replays                       # dataset  -> one game_<seed>.jsonl per game
    python -m imposterkings.analysis.replay_tools bundle replays/game_*.jsonl --out corpus.jsonl
                                                # replay files -> one dataset shard

A "replay file" and a "dataset shard" are the same thing: a JSONL of `record.GameRecord` lines. So
`ui.review --replay <shard> --game N` already reads any dataset shard directly; these helpers just
slice one game out (for sharing/inspection) or reassemble games into a corpus. Raw lines are copied
byte-for-byte (lossless); games are named by `deal_seed`.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional


def _lines(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [ln if ln.endswith("\n") else ln + "\n" for ln in f if ln.strip()]


def split(dataset: str, out_dir: str, game: Optional[int] = None) -> List[str]:
    """Write each game in ``dataset`` as its own ``game_<deal_seed>.jsonl`` (or only game ``game``)."""
    os.makedirs(out_dir, exist_ok=True)
    out: List[str] = []
    for i, ln in enumerate(_lines(dataset)):
        if game is not None and i != game:
            continue
        tag = json.loads(ln).get("deal_seed", i)
        path = os.path.join(out_dir, f"game_{tag}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(ln)
        out.append(path)
    return out


def bundle(files: List[str], out_path: str) -> int:
    """Concatenate per-game replay files into one dataset shard; returns the game count."""
    if any(os.path.abspath(f) == os.path.abspath(out_path) for f in files):
        raise ValueError("--out must not be one of the input files")
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for fp in files:
            for ln in _lines(fp):
                out.write(ln)
                n += 1
    return n


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Slice/assemble self-play GameRecord JSONL (dataset<->replay).")
    sub = p.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("split", help="dataset shard -> one replay file per game")
    ps.add_argument("dataset")
    ps.add_argument("--out-dir", required=True)
    ps.add_argument("--game", type=int, default=None, help="only extract this game index")
    pb = sub.add_parser("bundle", help="replay files -> one dataset shard")
    pb.add_argument("files", nargs="+")
    pb.add_argument("--out", required=True)
    args = p.parse_args(argv)

    if args.cmd == "split":
        paths = split(args.dataset, args.out_dir, args.game)
        print(f"wrote {len(paths)} replay file(s) -> {args.out_dir}")
    else:
        n = bundle(args.files, args.out)
        print(f"bundled {n} game(s) -> {args.out}")


if __name__ == "__main__":
    main()
