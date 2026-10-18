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


def _guard_seed(seed: int, seen: set, shard: str, paired: bool = False) -> None:
    """``game_id == deal_seed`` and train/val split BY GAME -- so two corpora sharing a seed would fuse two
    DIFFERENT games under one id and leak rows across the split. Fail loudly rather than corrupt silently.

    ``paired`` corpora (``gen.paired_deals``) repeat a seed ON PURPOSE: ``forced_hide`` plays the SAME deal
    four times (forced|free x starter 0|1) and the pairing is the whole design. There ``game_id ==
    deal_seed`` is exactly right -- it keeps all four games of a deal on the SAME side of the split, which
    is what we want, because they share an identical 18-card permutation and splitting them apart would leak
    the val hands into training. So the repeat is permitted; what stays forbidden is a repeat ACROSS corpora.
    """
    if seed in seen and not paired:
        raise ValueError(
            f"duplicate deal_seed {seed} (again in {shard}). Pooled corpora must use DISJOINT --base-seed "
            f"ranges: game_id == deal_seed, and the train/val split is by game_id, so a collision fuses two "
            f"different games and leaks them across the split. (A corpus that repeats a seed BY DESIGN must "
            f"say so with gen.paired_deals -- see forced_hide.py.)")
    seen.add(seed)


def _parse_data(data) -> Tuple[List[str], Dict[str, int], Dict[str, Dict[str, str]]]:
    """Split each ``--data`` entry into a dir plus an optional per-corpus suffix: a game CAP ``dir:N`` OR a
    record PREDICATE ``dir:field=value``.

    ``--limit`` is a GLOBAL first-N cap -- useless for mix-weighting. ``dir:N`` caps ONE corpus, so a
    shallow-search or weaker-labelled corpus can be down-weighted in the pool (``gen1_v3c:1000``) without
    copying shards. ``dir:field=value`` keeps only records whose ``gen.field`` matches -- e.g.
    ``forced_assassin_hide:arm=forced`` selects the forced arm alone (the free arm is ordinary self-play).
    Split on the LAST colon: a suffix of all digits is a cap, one containing ``=`` is a predicate, anything
    else (a Windows drive letter ``C:\\...``) leaves the entry a plain dir.
    """
    entries = [data] if isinstance(data, str) else list(data)
    dirs: List[str] = []
    caps: Dict[str, int] = {}
    preds: Dict[str, Dict[str, str]] = {}
    for e in entries:
        head, sep, tail = e.rpartition(":")
        if sep and tail.isdigit():
            dirs.append(head)
            caps[head] = int(tail)
        elif sep and "=" in tail:
            k, _, val = tail.partition("=")
            dirs.append(head)
            preds.setdefault(head, {})[k] = val
        else:
            dirs.append(e)
    return dirs, caps, preds


def _match(rec: dict, pred: Dict[str, str]) -> bool:
    """True if the record passes a corpus predicate (``gen.field == value`` for every constraint)."""
    if not pred:
        return True
    g = rec.get("gen") or {}
    return all(str(g.get(k)) == v for k, v in pred.items())


def _shards(dirs) -> List[Tuple[str, str]]:
    """``(dir, shard)`` pairs across one or MORE corpus dirs (pooling without copying shards).

    The dir is carried alongside each shard so the build loop can enforce a per-corpus cap. Pooling is only
    sound when the corpora use DISJOINT deal-seed ranges: ``game_id == deal_seed`` and ``train_tokens``
    splits train/val by game_id, so colliding seeds would fuse different games under one id (and leak across
    the split). ``_guard_seed`` asserts this."""
    dl = [dirs] if isinstance(dirs, str) else list(dirs)
    return sorted(((d, f) for d in dl for f in glob.glob(os.path.join(d, "*.jsonl"))), key=lambda p: p[1])


def build(data_dir, out_path: str, limit: Optional[int] = None, feat: str = "v1") -> Dict:
    """``data_dir`` is one corpus dir or a LIST of them (pooled). ``feat="v2"`` builds FIXED-shape rows via
    ``features2.tokenize`` (cards [n,18,46] + kings [n,2,4], no ragged offsets); "v1" is the original
    variable-length path, byte-identical."""
    if feat == "v2":
        return _build_v2(data_dir, out_path, limit)
    if feat == "v3":
        return _build_v3(data_dir, out_path, limit)
    dirs, caps, preds = _parse_data(data_dir)
    files = _shards(dirs)
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
    per_dir: Dict[str, int] = {}                       # games kept PER SOURCE dir, for the dir:N cap
    for src, f in files:                               # 'src' not 'd': 'd' is the decision var below
        if caps.get(src) and per_dir.get(src, 0) >= caps[src]:
            continue                                   # this corpus's cap is met -> skip the rest of it
        for rec in read_jsonl(f):
            if caps.get(src) and per_dir.get(src, 0) >= caps[src]:
                break                                  # cap reached mid-shard
            if not _match(rec, preds.get(src, {})):
                continue                               # corpus predicate (e.g. arm=forced): skip non-matching
            g = rec["deal_seed"]
            _guard_seed(g, seen, f, paired=bool((rec.get("gen") or {}).get("paired_deals")))
            rewards = rec["rewards"]
            st = GameState.deal(np.random.default_rng(g),      # honour the RECORDED starter: a no-op for a
                                rec.get("starting_player"))   # normal deal, but a mirrored corpus
                                                              # forces it and would otherwise desync
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
            per_dir[src] = per_dir.get(src, 0) + 1      # the cap counts USABLE games (desynced don't)
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
            "total_card_tokens": int(cards.shape[0]), "source": dirs,
            "caps": caps, "preds": preds, "games_per_source": dict(per_dir)}      # self-documenting mix
    with open(out_path + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return {k: meta[k] for k in ("n_rows", "n_games", "n_decisions", "total_card_tokens")}


def _build_v2(data_dir, out_path: str, limit: Optional[int]) -> Dict:
    """The v2 (fixed-18) build: same replay/skip-on-desync logic, fixed-shape output arrays."""
    from .features2 import tokenize as tokenize2
    dirs, caps, preds = _parse_data(data_dir)
    files = _shards(dirs)
    try:
        from tqdm import tqdm
        files = tqdm(files, desc="shards", unit="shard")
    except Exception:
        pass

    cards, kings, board, phase, action = [], [], [], [], []
    y, w, z, gid, did, chosen = [], [], [], [], [], []
    n_games = n_skipped = dec_id = 0
    seen: set = set()
    per_dir: Dict[str, int] = {}                       # games kept PER SOURCE dir, for the dir:N cap
    for src, f in files:                               # 'src' not 'd': 'd' is the decision var below
        if caps.get(src) and per_dir.get(src, 0) >= caps[src]:
            continue                                   # this corpus's cap is met -> skip the rest of it
        for rec in read_jsonl(f):
            if caps.get(src) and per_dir.get(src, 0) >= caps[src]:
                break                                  # cap reached mid-shard
            if not _match(rec, preds.get(src, {})):
                continue                               # corpus predicate (e.g. arm=forced): skip non-matching
            g = rec["deal_seed"]
            _guard_seed(g, seen, f, paired=bool((rec.get("gen") or {}).get("paired_deals")))
            rewards = rec["rewards"]
            st = GameState.deal(np.random.default_rng(g),      # honour the RECORDED starter: a no-op for a
                                rec.get("starting_player"))   # normal deal, but a mirrored corpus
                                                              # forces it and would otherwise desync
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
            per_dir[src] = per_dir.get(src, 0) + 1      # the cap counts USABLE games (desynced don't)
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
            "total_card_tokens": int(arrs["y"].shape[0]) * 18, "source": dirs,
            "caps": caps, "preds": preds, "games_per_source": dict(per_dir)}      # self-documenting mix
    with open(out_path + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return {k: meta[k] for k in ("n_rows", "n_games", "n_decisions", "total_card_tokens")}


def _build_v3(data_dir, out_path: str, limit: Optional[int]) -> Dict:
    """The v3 (late-fusion, Study 11) build: ONE row per DECISION -- the candidate-independent state once
    (``features3.tokenize_state3``) + its candidates as pointer records padded to ``MAX_CANDS`` (kind = -1
    on padding), with per-candidate labels ``y`` (mean_q) / ``w`` (visit share). Same replay, caps,
    predicates and desync skip as v2, so the same games (and ``game_split``) come out. Each decision's
    recorded candidates are checked against ``legal_moves()``; mismatches are counted, not hidden."""
    from . import features3 as F3
    dirs, caps, preds = _parse_data(data_dir)
    files = _shards(dirs)
    try:
        from tqdm import tqdm
        files = tqdm(files, desc="shards", unit="shard")
    except Exception:
        pass

    K = F3.MAX_CANDS
    cols: Dict[str, list] = {k: [] for k in ("cards", "card_depth", "hands", "kings", "decision", "src_ptr",
                                             "cand_kind", "cand_ptr_a", "cand_ptr_b", "cand_name",
                                             "cand_number", "y", "w", "z", "chosen", "game_id")}
    n_games = n_skipped = n_cand_mismatch = 0
    seen: set = set()
    per_dir: Dict[str, int] = {}
    for src, f in files:
        if caps.get(src) and per_dir.get(src, 0) >= caps[src]:
            continue
        for rec in read_jsonl(f):
            if caps.get(src) and per_dir.get(src, 0) >= caps[src]:
                break
            if not _match(rec, preds.get(src, {})):
                continue
            g = rec["deal_seed"]
            _guard_seed(g, seen, f, paired=bool((rec.get("gen") or {}).get("paired_deals")))
            rewards = rec["rewards"]
            st = GameState.deal(np.random.default_rng(g), rec.get("starting_player"))
            mark = len(cols["game_id"])
            desynced = False
            for d in rec["decisions"]:
                a = dict_to_action(d["chosen"])
                legal = st.legal_moves()
                if a not in legal:
                    desynced = True
                    break
                cands = d.get("candidates") or []
                if cands:
                    moves = [dict_to_action(c["move"]) for c in cands]
                    if set(moves) != set(legal):
                        n_cand_mismatch += 1
                    view = st.information_set(d["seat"])
                    s = F3.tokenize_state3(view, legal_moves=legal)
                    c = F3.encode_candidates(view, s, moves)
                    n = len(moves)

                    def pad(x, fill, dtype):
                        out = np.full(K, fill, dtype)
                        out[:n] = x
                        return out

                    cols["cards"].append(s.cards); cols["card_depth"].append(s.card_depth)
                    cols["hands"].append(s.hands); cols["kings"].append(s.kings)
                    cols["decision"].append(s.decision); cols["src_ptr"].append(s.src_ptr)
                    cols["cand_kind"].append(pad(c.kind, -1, np.int8))
                    cols["cand_ptr_a"].append(pad(c.ptr_a, -1, np.int8))
                    cols["cand_ptr_b"].append(pad(c.ptr_b, -1, np.int8))
                    cols["cand_name"].append(pad(c.name, -1, np.int8))
                    cols["cand_number"].append(pad(c.number, -1, np.int8))
                    cols["y"].append(pad([cc["mean_q"] for cc in cands], 0.0, np.float32))
                    cols["w"].append(pad([cc["visit_share"] for cc in cands], 0.0, np.float32))
                    cols["z"].append(rewards[d["seat"]])
                    cols["chosen"].append(moves.index(a))
                    cols["game_id"].append(g)
                st = st.apply(a)
            if desynced:
                for lst in cols.values():
                    del lst[mark:]
                n_skipped += 1
                continue
            n_games += 1
            per_dir[src] = per_dir.get(src, 0) + 1
            if limit and n_games >= limit:
                break
        if limit and n_games >= limit:
            break
    if n_skipped:
        print(f"  WARNING: skipped {n_skipped} desynced game(s) (corpus predates a rules change)")
    if n_cand_mismatch:
        print(f"  NOTE: {n_cand_mismatch} decision(s) whose recorded candidates != legal_moves()")

    D = len(cols["game_id"])
    arrs = dict(
        cards=np.asarray(cols["cards"], np.float32).reshape(D, 18, F3.CARD_DIM),
        card_depth=np.asarray(cols["card_depth"], np.int8).reshape(D, 18),
        hands=np.asarray(cols["hands"], np.float32).reshape(D, 2, F3.HAND_DIM),
        kings=np.asarray(cols["kings"], np.float32).reshape(D, 2, F3.KING_DIM),
        decision=np.asarray(cols["decision"], np.float32).reshape(D, F3.DECISION_DIM),
        src_ptr=np.asarray(cols["src_ptr"], np.int8),
        **{k: np.asarray(cols[k], np.int8).reshape(D, K) for k in
           ("cand_kind", "cand_ptr_a", "cand_ptr_b", "cand_name", "cand_number")},
        y=np.asarray(cols["y"], np.float32).reshape(D, K), w=np.asarray(cols["w"], np.float32).reshape(D, K),
        z=np.asarray(cols["z"], np.float32), chosen=np.asarray(cols["chosen"], np.int8),
        game_id=np.asarray(cols["game_id"], np.int64), decision_id=np.arange(D, dtype=np.int64),
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, out_path)
    n_rows = int((arrs["cand_kind"] >= 0).sum())
    meta = {"feat": "v3", "card_dim": F3.CARD_DIM, "hand_dim": F3.HAND_DIM, "king_dim": F3.KING_DIM,
            "decision_dim": F3.DECISION_DIM, "max_cands": K, "target": "q", "layout": "per-decision",
            "n_rows": n_rows, "n_games": n_games, "n_skipped_desynced": n_skipped, "n_decisions": D,
            "n_cand_mismatch": n_cand_mismatch, "total_card_tokens": D * 18, "source": dirs,
            "caps": caps, "preds": preds, "games_per_source": dict(per_dir)}
    with open(out_path + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return {k: meta[k] for k in ("n_rows", "n_games", "n_decisions", "total_card_tokens")}


def load_v3(path: str) -> Dict[str, np.ndarray]:
    """A v3 (per-decision) dataset as a dict of arrays; ``cand_kind < 0`` marks padding."""
    with np.load(path) as npz:
        return {k: npz[k] for k in npz.files}


def build_belief(data_dir, out_path: str, limit: Optional[int] = None) -> Dict:
    """Belief-net dataset (Study 7): ONE state-only row per searched decision, labelled with the TRUE zone
    of every card (``features2.true_zones``, read off the omniscient replay). No candidate loop, no q/w/z --
    belief is ``P(zones | s)``, not f(s,a). Always v2 (needs the fixed 18 card-instance tokens). Reuses the
    same replay / desync-skip / seed-guard / per-corpus cap+predicate as the q builds."""
    from .features2 import tokenize_state, true_zones
    dirs, caps, preds = _parse_data(data_dir)
    files = _shards(dirs)
    try:
        from tqdm import tqdm
        files = tqdm(files, desc="shards", unit="shard")
    except Exception:
        pass

    cards, kings, board, phase, action = [], [], [], [], []
    zlab, smask, gid, did = [], [], [], []
    n_games = n_skipped = dec_id = 0
    seen: set = set()
    per_dir: Dict[str, int] = {}
    for src, f in files:
        if caps.get(src) and per_dir.get(src, 0) >= caps[src]:
            continue
        for rec in read_jsonl(f):
            if caps.get(src) and per_dir.get(src, 0) >= caps[src]:
                break
            if not _match(rec, preds.get(src, {})):
                continue                               # corpus predicate (e.g. arm=forced): skip non-matching
            g = rec["deal_seed"]
            _guard_seed(g, seen, f, paired=bool((rec.get("gen") or {}).get("paired_deals")))
            st = GameState.deal(np.random.default_rng(g), rec.get("starting_player"))
            mark = (len(cards), dec_id)
            desynced = False
            for d in rec["decisions"]:
                a = dict_to_action(d["chosen"])
                if a not in st.legal_moves():
                    desynced = True
                    break
                if d.get("candidates"):                # one belief row per genuine (searched) decision
                    view = st.information_set(d["seat"])
                    t = tokenize_state(view)
                    zl, sm = true_zones(view, st)
                    cards.append(t.cards); kings.append(t.kings)
                    board.append(t.board); phase.append(t.phase); action.append(t.action)
                    zlab.append(zl); smask.append(sm)
                    gid.append(g); did.append(dec_id)
                    dec_id += 1
                st = st.apply(a)
            if desynced:
                r0, dec_id = mark
                for lst in (cards, kings, board, phase, action, zlab, smask, gid, did):
                    del lst[r0:]
                n_skipped += 1
                continue
            n_games += 1
            per_dir[src] = per_dir.get(src, 0) + 1
            if limit and n_games >= limit:
                break
        if limit and n_games >= limit:
            break
    if n_skipped:
        print(f"  WARNING: skipped {n_skipped} desynced game(s) (corpus predates a rules change)")

    from .features2 import ACTION_DIM as AD2, BOARD_DIM as BD2, CARD_DIM as CD2, PHASE_DIM as PD2
    n_rows = len(gid)
    arrs = dict(
        cards=(np.stack(cards) if cards else np.zeros((0, 18, CD2), np.float32)).astype(np.float32),
        kings=(np.stack(kings) if kings else np.zeros((0, 2, 4), np.float32)).astype(np.float32),
        board=np.asarray(board, np.float32).reshape(-1, BD2),
        phase=np.asarray(phase, np.float32).reshape(-1, PD2),
        action=np.asarray(action, np.float32).reshape(-1, AD2),
        zone_label=(np.stack(zlab) if zlab else np.zeros((0, 18), np.int64)).astype(np.int64),
        seen_mask=(np.stack(smask) if smask else np.zeros((0, 18), np.bool_)).astype(np.bool_),
        game_id=np.asarray(gid, np.int64), decision_id=np.asarray(did, np.int64),
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(out_path, **arrs)
    meta = {"feat": "v2", "card_dim": CD2, "board_dim": BD2, "phase_dim": PD2, "action_dim": AD2,
            "target": "belief", "n_zones": 12, "n_rows": n_rows, "n_games": n_games,
            "n_skipped_desynced": n_skipped, "n_decisions": dec_id, "total_card_tokens": n_rows * 18,
            "source": dirs, "caps": caps, "preds": preds, "games_per_source": dict(per_dir)}
    with open(out_path + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return {k: meta[k] for k in ("n_rows", "n_games", "n_decisions", "total_card_tokens")}


class TokenRows:
    """A loaded token dataset. v1: ragged card tokens (CSR offsets); v2 (``feat == "v2"``): fixed-shape
    ``cards [n,18,46]`` + ``kings [n,2,4]`` (detected by the presence of the ``kings`` array)."""

    def __init__(self, arrs: Dict[str, np.ndarray]):
        self.feat = "v2" if "kings" in arrs else "v1"
        self.target = "belief" if "zone_label" in arrs else "q"     # belief datasets carry zone labels
        self.cards = arrs["cards"]                      # v1: [T, 44] ragged; v2: [n, 18, 46]
        self.kings = arrs.get("kings")                  # v2 only: [n, 2, 4]
        self.card_offsets = arrs.get("card_offsets")    # v1 only: [n_rows+1]
        self.board, self.phase, self.action = arrs["board"], arrs["phase"], arrs["action"]
        self.y, self.w, self.z = arrs.get("y"), arrs.get("w"), arrs.get("z")     # q datasets only
        self.zone_label, self.seen_mask = arrs.get("zone_label"), arrs.get("seen_mask")  # belief only
        self.game_id, self.decision_id = arrs["game_id"], arrs["decision_id"]
        self.is_chosen = arrs.get("is_chosen")

    def __len__(self) -> int:
        return int(self.game_id.shape[0])

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
                        "disjoint --base-seed ranges -- game_id == deal_seed and the split is by game). "
                        "Append ':N' to cap a corpus at N games for mix-weighting, e.g. "
                        "datasets/selfplay_gen1_v3c:1000")
    p.add_argument("--out", default=os.path.join("datasets", "tensors", "k20l3_tokens.npz"))
    p.add_argument("--limit", type=int, default=None, help="only process the first N games (smoke)")
    p.add_argument("--feat", default="v1", choices=["v1", "v2", "v3"],
                   help="featurization version (v2 = fixed-18 instance tokens + zone posteriors; v3 = the "
                        "late-fusion per-decision layout, Study 11)")
    p.add_argument("--target", default="q", choices=["q", "belief"],
                   help="q = action-in value labels (mean_q); belief = per-card true-zone labels, one "
                        "state-only row per decision (Study 7, always v2)")
    args = p.parse_args(argv)

    src = args.data if len(args.data) > 1 else args.data[0]
    if args.target == "belief":
        print(f"building BELIEF tensors (v2) from {src} -> {args.out}"
              + (f" (limit {args.limit})" if args.limit else ""))
        s = build_belief(src, args.out, args.limit)
    else:
        print(f"building token tensors ({args.feat}) from {src} -> {args.out}"
              + (f" (limit {args.limit})" if args.limit else ""))
        s = build(src, args.out, args.limit, feat=args.feat)
    print(f"  {s['n_rows']} rows ({s['n_decisions']} decisions, {s['n_games']} games), "
          f"{s['total_card_tokens']} card tokens  ->  {args.out}(+.meta.json)")


if __name__ == "__main__":
    main()
