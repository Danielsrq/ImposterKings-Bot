"""Replay a self-play JSONL corpus into TOKEN training tensors for the attention model. Numpy only.

    python -m imposterkings.machine_learning.token_dataset \
        --data datasets/selfplay_k20l3 --out datasets/tensors/k20l3_tokens.npz

The token sibling of ``dataset.py``: the *same* replay, rows and labels, but each row's features are the
variable-length token set from ``features.tokenize`` (card/board/phase/action) instead of the flat [216]
vector. Card-token count varies per row (8-16), so cards are stored **ragged** -- all rows' card tokens
concatenated into one ``cards`` array indexed by CSR ``card_offsets`` (row i = cards[off[i]:off[i+1]]).
Deterministic: each game reconstructs from ``deal_seed`` + the action log. No engine changes, no
re-collection -- this is pure post-processing of the existing corpus.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np

from ..record import dict_to_action, read_jsonl
from ..state import GameState
from .features import ACTION_DIM, BOARD_DIM, CARD_DIM, PHASE_DIM, tokenize


def build(data_dir: str, out_path: str, limit: Optional[int] = None) -> Dict:
    files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    try:
        from tqdm import tqdm
        files = tqdm(files, desc="shards", unit="shard")
    except Exception:
        pass

    cards_chunks = []                                  # [Nᵢ,44] per row, concatenated at the end
    offsets = [0]
    board, phase, action = [], [], []
    y, w, z, gid, did, chosen = [], [], [], [], [], []
    n_games = n_skipped = dec_id = total_tokens = 0
    for f in files:
        for rec in read_jsonl(f):
            g = rec["deal_seed"]
            rewards = rec["rewards"]
            st = GameState.deal(np.random.default_rng(g))
            mark = (len(cards_chunks), len(offsets), len(board), total_tokens, dec_id)
            desynced = False
            for d in rec["decisions"]:
                a = dict_to_action(d["chosen"])
                if a not in st.legal_moves():
                    desynced = True                    # corpus recorded under different rules -> drop game
                    break
                cands = d.get("candidates") or []
                if cands:                              # a real (searched) decision
                    view = st.information_set(d["seat"])
                    for c in cands:
                        t = tokenize(view, dict_to_action(c["move"]))
                        cards_chunks.append(t.cards)
                        total_tokens += t.cards.shape[0]
                        offsets.append(total_tokens)
                        board.append(t.board); phase.append(t.phase); action.append(t.action)
                        y.append(c["mean_q"]); w.append(c["visit_share"]); z.append(rewards[d["seat"]])
                        gid.append(g); did.append(dec_id); chosen.append(int(c["move"] == d["chosen"]))
                    dec_id += 1
                st = st.apply(a)
            if desynced:                               # roll this game's rows back entirely
                c0, o0, r0, t0, dec_id = mark
                del cards_chunks[c0:]; del offsets[o0:]
                for lst in (board, phase, action, y, w, z, gid, did, chosen):
                    del lst[r0:]
                total_tokens = t0
                n_skipped += 1
                continue
            n_games += 1
            if limit and n_games >= limit:
                break
        if limit and n_games >= limit:
            break
    if n_skipped:
        print(f"  WARNING: skipped {n_skipped} game(s) whose recorded actions are no longer legal "
              f"(corpus predates an engine rules change) -- rows rolled back, remaining games kept")

    cards = (np.concatenate(cards_chunks, axis=0).astype(np.float32) if cards_chunks
             else np.zeros((0, CARD_DIM), np.float32))
    arrs = dict(
        cards=cards, card_offsets=np.asarray(offsets, np.int64),
        board=np.asarray(board, np.float32).reshape(-1, BOARD_DIM),
        phase=np.asarray(phase, np.float32).reshape(-1, PHASE_DIM),
        action=np.asarray(action, np.float32).reshape(-1, ACTION_DIM),
        y=np.asarray(y, np.float32), w=np.asarray(w, np.float32), z=np.asarray(z, np.float32),
        game_id=np.asarray(gid, np.int64), decision_id=np.asarray(did, np.int64),
        is_chosen=np.asarray(chosen, np.int8),
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(out_path, **arrs)
    meta = {"card_dim": CARD_DIM, "board_dim": BOARD_DIM, "phase_dim": PHASE_DIM,
            "action_dim": ACTION_DIM, "target": "q", "n_rows": int(arrs["y"].shape[0]),
            "n_games": n_games, "n_skipped_desynced": n_skipped, "n_decisions": dec_id,
            "total_card_tokens": int(cards.shape[0]), "source": data_dir}
    with open(out_path + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return {k: meta[k] for k in ("n_rows", "n_games", "n_decisions", "total_card_tokens")}


class TokenRows:
    """A loaded token dataset: ragged card tokens (CSR offsets) + fixed board/phase/action + labels."""

    def __init__(self, arrs: Dict[str, np.ndarray]):
        self.cards = arrs["cards"]                      # [T, 44]
        self.card_offsets = arrs["card_offsets"]        # [n_rows+1]
        self.board, self.phase, self.action = arrs["board"], arrs["phase"], arrs["action"]
        self.y, self.w, self.z = arrs["y"], arrs["w"], arrs["z"]
        self.game_id, self.decision_id, self.is_chosen = (
            arrs["game_id"], arrs["decision_id"], arrs["is_chosen"])

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def tokens(self, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Row i's token set: (cards[Nᵢ,44], board[14], phase[53], action[23])."""
        a, b = int(self.card_offsets[i]), int(self.card_offsets[i + 1])
        return self.cards[a:b], self.board[i], self.phase[i], self.action[i]


def load(path: str) -> TokenRows:
    with np.load(path) as npz:                          # materialize once (avoid lazy per-access reads)
        return TokenRows({k: npz[k] for k in npz.files})


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build token training tensors from a self-play corpus.")
    p.add_argument("--data", default=os.path.join("datasets", "selfplay_k20l3"))
    p.add_argument("--out", default=os.path.join("datasets", "tensors", "k20l3_tokens.npz"))
    p.add_argument("--limit", type=int, default=None, help="only process the first N games (smoke)")
    args = p.parse_args(argv)

    print(f"building token tensors from {args.data} -> {args.out}"
          + (f" (limit {args.limit})" if args.limit else ""))
    s = build(args.data, args.out, args.limit)
    print(f"  {s['n_rows']} rows ({s['n_decisions']} decisions, {s['n_games']} games), "
          f"{s['total_card_tokens']} card tokens  ->  {args.out}(+.meta.json)")


if __name__ == "__main__":
    main()
