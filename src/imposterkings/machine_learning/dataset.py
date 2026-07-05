"""Replay a self-play JSONL corpus into training tensors (state,action -> q). Numpy only.

    python -m imposterkings.machine_learning.dataset --data datasets/selfplay_k20l3 --out datasets/tensors/k20l3.npz

For every *searched* decision (forced plies carry no candidates), each candidate action becomes one row:
``X`` = featurize(info_set, action), ``y`` = the MCTS ``mean_q`` of that action (the target), ``w`` =
``visit_share`` (loss weight -- reliable q's dominate), ``z`` = the mover's terminal reward (kept for
later blend experiments), plus ``game_id`` (group split), ``decision_id`` (group ranking eval), and
``is_chosen``. Deterministic: each game reconstructs from ``deal_seed`` + the action log.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, Optional

import numpy as np

from ..record import dict_to_action, read_jsonl
from ..state import GameState
from .features import FEATURE_DIM, encode, feature_names


def build(data_dir: str, out_path: str, limit: Optional[int] = None) -> Dict:
    files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    try:
        from tqdm import tqdm
        files = tqdm(files, desc="shards", unit="shard")
    except Exception:
        pass

    X, y, w, z, gid, did, chosen = [], [], [], [], [], [], []
    n_games = dec_id = 0
    for f in files:
        for rec in read_jsonl(f):
            g = rec["deal_seed"]
            rewards = rec["rewards"]
            st = GameState.deal(np.random.default_rng(g))
            for d in rec["decisions"]:
                cands = d.get("candidates") or []
                if cands:                                  # a real (searched) decision
                    view = st.information_set(d["seat"])
                    for c in cands:
                        X.append(encode(view, dict_to_action(c["move"])))
                        y.append(c["mean_q"]); w.append(c["visit_share"]); z.append(rewards[d["seat"]])
                        gid.append(g); did.append(dec_id); chosen.append(int(c["move"] == d["chosen"]))
                    dec_id += 1
                st = st.apply(dict_to_action(d["chosen"]))
            n_games += 1
            if limit and n_games >= limit:
                break
        if limit and n_games >= limit:
            break

    arrs = dict(
        X=np.asarray(X, np.float32), y=np.asarray(y, np.float32), w=np.asarray(w, np.float32),
        z=np.asarray(z, np.float32), game_id=np.asarray(gid, np.int64),
        decision_id=np.asarray(did, np.int64), is_chosen=np.asarray(chosen, np.int8),
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(out_path, **arrs)
    meta = {"feature_dim": FEATURE_DIM, "target": "q", "n_rows": int(arrs["X"].shape[0]),
            "n_games": n_games, "n_decisions": dec_id, "source": data_dir,
            "feature_names": feature_names()}
    with open(out_path + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return {k: meta[k] for k in ("n_rows", "n_games", "n_decisions", "feature_dim")}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build (state,action)->q training tensors from a self-play corpus.")
    p.add_argument("--data", default=os.path.join("datasets", "selfplay_k20l3"))
    p.add_argument("--out", default=os.path.join("datasets", "tensors", "k20l3.npz"))
    p.add_argument("--limit", type=int, default=None, help="only process the first N games (smoke)")
    args = p.parse_args(argv)

    print(f"building tensors from {args.data} -> {args.out}" + (f" (limit {args.limit})" if args.limit else ""))
    stats = build(args.data, args.out, args.limit)
    print(f"  {stats['n_rows']} rows ({stats['n_decisions']} decisions, {stats['n_games']} games), "
          f"D={stats['feature_dim']}  ->  {args.out}(+.meta.json)")


if __name__ == "__main__":
    main()
