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
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..record import dict_to_action, read_jsonl
from ..state import GameState
from .features import ACTION_DIM, BOARD_DIM, CARD_DIM, PHASE_DIM, tokenize


def _guard_seed(seed: int, seen: set, shard: str) -> None:
    """``game_id == deal_seed`` and train/val split BY GAME -- so two corpora sharing a seed would fuse two
    DIFFERENT games under one id and leak rows across the split. Fail loudly rather than corrupt silently."""
    if seed in seen:
        raise ValueError(
            f"duplicate deal_seed {seed} (again in {shard}). Pooled corpora must use DISJOINT --base-seed "
            f"ranges: game_id == deal_seed, and the train/val split is by game_id, so a collision fuses two "
            f"different games and leaks them across the split.")
    seen.add(seed)


def _shards(data_dir) -> List[str]:
    """Every shard across one or MORE corpus dirs (pooling gen-1 + k50 + mixed without copying shards).

    Pooling is only sound when the corpora use DISJOINT deal-seed ranges: ``game_id == deal_seed`` and
    ``train_tokens`` splits train/val by game_id, so colliding seeds would fuse different games under one
    id (and leak across the split). ``build`` asserts this."""
    dirs = [data_dir] if isinstance(data_dir, str) else list(data_dir)
    return sorted(f for d in dirs for f in glob.glob(os.path.join(d, "*.jsonl")))


def build(data_dir, out_path: str, limit: Optional[int] = None, feat: str = "v1") -> Dict:
    """``data_dir`` is one corpus dir or a LIST of them (pooled). ``feat="v2"`` builds FIXED-shape rows via
    ``features2.tokenize`` (cards [n,18,46] + kings [n,2,4], no ragged offsets); "v1" is the original
    variable-length path, byte-identical."""
    if feat == "v2":
        return _build_v2(data_dir, out_path, limit)
    files = _shards(data_dir)
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
    seen: set = set()
    for f in files:
        for rec in read_jsonl(f):
            g = rec["deal_seed"]
            _guard_seed(g, seen, f)
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
            "total_card_tokens": int(cards.shape[0]),
            "source": data_dir if isinstance(data_dir, str) else list(data_dir)}
    with open(out_path + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return {k: meta[k] for k in ("n_rows", "n_games", "n_decisions", "total_card_tokens")}


def _build_v2(data_dir, out_path: str, limit: Optional[int]) -> Dict:
    """The v2 (fixed-18) build: same replay/skip-on-desync logic, fixed-shape output arrays."""
    from .features2 import tokenize as tokenize2
    files = _shards(data_dir)
    try:
        from tqdm import tqdm
        files = tqdm(files, desc="shards", unit="shard")
    except Exception:
        pass

    cards, kings, board, phase, action = [], [], [], [], []
    y, w, z, gid, did, chosen = [], [], [], [], [], []
    n_games = n_skipped = dec_id = 0
    seen: set = set()
    for f in files:
        for rec in read_jsonl(f):
            g = rec["deal_seed"]
            _guard_seed(g, seen, f)
            rewards = rec["rewards"]
            st = GameState.deal(np.random.default_rng(g))
            mark = (len(cards), dec_id)
            desynced = False
            for d in rec["decisions"]:
                a = dict_to_action(d["chosen"])
                if a not in st.legal_moves():
                    desynced = True
                    break
                cands = d.get("candidates") or []
                if cands:
                    view = st.information_set(d["seat"])
                    for c in cands:
                        t = tokenize2(view, dict_to_action(c["move"]))
                        cards.append(t.cards); kings.append(t.kings)
                        board.append(t.board); phase.append(t.phase); action.append(t.action)
                        y.append(c["mean_q"]); w.append(c["visit_share"]); z.append(rewards[d["seat"]])
                        gid.append(g); did.append(dec_id); chosen.append(int(c["move"] == d["chosen"]))
                    dec_id += 1
                st = st.apply(a)
            if desynced:
                r0, dec_id = mark
                for lst in (cards, kings, board, phase, action, y, w, z, gid, did, chosen):
                    del lst[r0:]
                n_skipped += 1
                continue
            n_games += 1
            if limit and n_games >= limit:
                break
        if limit and n_games >= limit:
            break
    if n_skipped:
        print(f"  WARNING: skipped {n_skipped} desynced game(s) (corpus predates a rules change)")

    from .features2 import ACTION_DIM as AD2, BOARD_DIM as BD2, CARD_DIM as CD2, PHASE_DIM as PD2
    arrs = dict(
        cards=(np.stack(cards) if cards else np.zeros((0, 18, CD2), np.float32)).astype(np.float32),
        kings=(np.stack(kings) if kings else np.zeros((0, 2, 4), np.float32)).astype(np.float32),
        board=np.asarray(board, np.float32).reshape(-1, BD2),
        phase=np.asarray(phase, np.float32).reshape(-1, PD2),
        action=np.asarray(action, np.float32).reshape(-1, AD2),
        y=np.asarray(y, np.float32), w=np.asarray(w, np.float32), z=np.asarray(z, np.float32),
        game_id=np.asarray(gid, np.int64), decision_id=np.asarray(did, np.int64),
        is_chosen=np.asarray(chosen, np.int8),
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(out_path, **arrs)
    meta = {"feat": "v2", "card_dim": CD2, "board_dim": BD2, "phase_dim": PD2, "action_dim": AD2,
            "target": "q", "n_rows": int(arrs["y"].shape[0]), "n_games": n_games,
            "n_skipped_desynced": n_skipped, "n_decisions": dec_id,
            "total_card_tokens": int(arrs["y"].shape[0]) * 18,
            "source": data_dir if isinstance(data_dir, str) else list(data_dir)}
    with open(out_path + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return {k: meta[k] for k in ("n_rows", "n_games", "n_decisions", "total_card_tokens")}


class TokenRows:
    """A loaded token dataset. v1: ragged card tokens (CSR offsets); v2 (``feat == "v2"``): fixed-shape
    ``cards [n,18,46]`` + ``kings [n,2,4]`` (detected by the presence of the ``kings`` array)."""

    def __init__(self, arrs: Dict[str, np.ndarray]):
        self.feat = "v2" if "kings" in arrs else "v1"
        self.cards = arrs["cards"]                      # v1: [T, 44] ragged; v2: [n, 18, 46]
        self.kings = arrs.get("kings")                  # v2 only: [n, 2, 4]
        self.card_offsets = arrs.get("card_offsets")    # v1 only: [n_rows+1]
        self.board, self.phase, self.action = arrs["board"], arrs["phase"], arrs["action"]
        self.y, self.w, self.z = arrs["y"], arrs["w"], arrs["z"]
        self.game_id, self.decision_id, self.is_chosen = (
            arrs["game_id"], arrs["decision_id"], arrs["is_chosen"])

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def tokens(self, i: int) -> Tuple[np.ndarray, ...]:
        """Row i's token set. v1: (cards[Nᵢ,44], board, phase, action); v2: (cards[18,46], kings[2,4],
        board, phase, action)."""
        if self.feat == "v2":
            return self.cards[i], self.kings[i], self.board[i], self.phase[i], self.action[i]
        a, b = int(self.card_offsets[i]), int(self.card_offsets[i + 1])
        return self.cards[a:b], self.board[i], self.phase[i], self.action[i]


def load(path: str) -> TokenRows:
    with np.load(path) as npz:                          # materialize once (avoid lazy per-access reads)
        return TokenRows({k: npz[k] for k in npz.files})


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build token training tensors from a self-play corpus.")
    p.add_argument("--data", nargs="+", default=[os.path.join("datasets", "selfplay_k20l3")],
                   help="one or MORE corpus dirs; several are POOLED into one dataset (they must use "
                        "disjoint --base-seed ranges -- game_id == deal_seed and the split is by game)")
    p.add_argument("--out", default=os.path.join("datasets", "tensors", "k20l3_tokens.npz"))
    p.add_argument("--limit", type=int, default=None, help="only process the first N games (smoke)")
    p.add_argument("--feat", default="v1", choices=["v1", "v2"],
                   help="featurization version (v2 = fixed-18 instance tokens + zone posteriors)")
    args = p.parse_args(argv)

    src = args.data if len(args.data) > 1 else args.data[0]
    print(f"building token tensors ({args.feat}) from {src} -> {args.out}"
          + (f" (limit {args.limit})" if args.limit else ""))
    s = build(src, args.out, args.limit, feat=args.feat)
    print(f"  {s['n_rows']} rows ({s['n_decisions']} decisions, {s['n_games']} games), "
          f"{s['total_card_tokens']} card tokens  ->  {args.out}(+.meta.json)")


if __name__ == "__main__":
    main()
