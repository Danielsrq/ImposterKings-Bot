"""Build the LF attention viewer (one self-contained HTML page) from RECORDED games -- a post-process step.

    python -m imposterkings.machine_learning.lf_viewer datasets/human --out reports/viewer/human.html --open
    python -m imposterkings.machine_learning.lf_viewer datasets/human/session_20260724_154246.jsonl --last 5 \\
        --out reports/viewer/last5.html

Reads game-record JSONL (``record.GameRecord``: the pygame app's ``datasets/human`` sessions, or any self-play
corpus), from files or folders. Every game is replayed from its seed and recorded moves; at each decision LF-B
scores the legal actions from the deciding seat's own information set, exactly as ``lf_attention_export``
does for live self-play. The search columns (visit share, root value) come from the record: the bot's own
search at the time. A human's moves had no search, so they show the net's view only.

The q, policy and attention are LF-B's opinion of the position, not necessarily what the recorded bot used
(an app game may have been played against vanilla MCTS or another net). Games recorded under older rules
whose moves are no longer legal are skipped and counted.
"""
from __future__ import annotations

import argparse
import glob
import os
import webbrowser
from typing import List, Optional

import numpy as np

from ..record import dict_to_action, read_jsonl
from ..state import GameState
from .lf_attention_export import decision_step, finish_game, make_blob, move_entry, write_html


def record_files(paths: List[str]) -> List[str]:
    """Files as given; folders expand to every ``*.jsonl`` beneath them (sorted, so the order is stable)."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True))
        elif os.path.isfile(p):
            out.append(p)
        else:
            raise FileNotFoundError(p)
    return out


def replay_game(model, rec: dict, source: str) -> Optional[dict]:
    """One recorded game -> the viewer's game dict, or None if it no longer replays (older rules, or unfinished)."""
    if "deal" in rec:
        raise ValueError(f"{source}: explicit-deal records (the Rust generator's) need record.initial_state")
    st = GameState.deal(np.random.default_rng(rec["deal_seed"]), rec.get("starting_player"))
    steps: List[dict] = []
    moves: List[dict] = []
    for d in rec["decisions"]:
        move = dict_to_action(d["chosen"])
        if move not in st.legal_moves():
            return None
        seat = d["seat"]
        view = st.information_set(seat)
        cands = d.get("candidates") or []
        visits = value = None
        if cands:                                          # the bot searched here; a human ply has none
            visits = {dict_to_action(c["move"]): c["visit_share"] for c in cands}
            total = sum(c["visits"] for c in cands)
            value = sum(c["visits"] * c["mean_q"] for c in cands) / total if total else None
        steps.append(decision_step(model, view, move, st, seat, len(moves), visits, value))
        moves.append(move_entry(view, move, seat, len(moves)))
        st = st.apply(move)
    if not st.is_terminal():
        return None
    gen = rec.get("gen") or {}
    who = None
    if gen.get("human") and gen.get("primary_seat") is not None:
        hs = int(gen["primary_seat"])
        who = {hs: "you", 1 - hs: "bot"}
    return finish_game(steps, moves, st, seed=rec["deal_seed"], source=source, who=who,
                       opponent=gen.get("opponent_spec"), recorded_at=gen.get("recorded_at"))


def build(paths: List[str], out: str, lf: str, last: Optional[int] = None) -> dict:
    from . import latefusion_model as LF
    model, _ = LF.load(lf)
    recs = [(rec, f"{os.path.basename(f)}:{i + 1}") for f in record_files(paths) for i, rec in enumerate(read_jsonl(f))]
    if last:
        recs = recs[-last:]
    games, skipped = [], []
    for rec, src in recs:
        g = replay_game(model, rec, src)
        if g is None:
            skipped.append(src)
        else:
            games.append(g)
    if not games:
        raise SystemExit(f"no replayable games in {paths} ({len(skipped)} skipped)")
    opps = sorted({g["opponent"] for g in games if g.get("opponent")})
    human = any(g.get("who") for g in games)
    intro = ("Recorded games replayed through Study 11's late-fusion net (LF-B). At every decision, pick a legal action "
             "to see where its candidate token looked: cross-attention over the 23 state tokens, computed from the deciding "
             "player's own view. q, policy and attention are LF-B's opinion; the search column and the graph's filled points "
             "are the recorded bot's own search" + (f" ({', '.join(opps)})" if opps else "") + "."
             + (" Your moves had no search, so they show the net's view only (hollow points)." if human else ""))
    write_html(make_blob(games, lf, ", ".join(opps) or "recorded", title="LF-B on recorded games", intro=intro), out)
    return {"games": len(games), "skipped": skipped, "decisions": sum(len(g["steps"]) for g in games)}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build the LF attention viewer page from recorded game JSONL.")
    p.add_argument("paths", nargs="+", help="game-record .jsonl files and/or folders of them")
    p.add_argument("--out", required=True, help="the .html page to write")
    p.add_argument("--lf", default=os.path.join("models", "study11", "LF-B.pt"))
    p.add_argument("--last", type=int, default=None, help="only the most recent N games (in file order)")
    p.add_argument("--open", action="store_true", help="open the page in the default browser when done")
    a = p.parse_args(argv)
    r = build(a.paths, a.out, a.lf, a.last)
    print(f"{r['games']} games, {r['decisions']} decisions -> {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")
    if r["skipped"]:
        print(f"  skipped {len(r['skipped'])} game(s) that no longer replay (recorded under older rules, or "
              f"unfinished): {', '.join(r['skipped'][:8])}{' ...' if len(r['skipped']) > 8 else ''}")
    if a.open:
        webbrowser.open("file://" + os.path.abspath(a.out))


if __name__ == "__main__":
    main()
