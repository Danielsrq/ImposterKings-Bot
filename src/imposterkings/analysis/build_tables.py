"""Build exploration tables from a self-play JSONL corpus and print the DATASET.md query cookbook.

    python -m imposterkings.analysis.build_tables --data datasets/selfplay_k20l3 --report

Replays every game (deterministic: deal_seed + action log) and emits three CSVs to ``--out-dir``:
  * ``card_locations.csv`` -- one row per (game, card) at the **post-setup** position (after both players
    hide+discard): where each of the 18 cards sits (hand/hidden/setup_discard/leftover). Ground truth.
  * ``ply_events.csv``     -- one row per decision: phase, action fields, leading/source/against card,
    mover z, chosen mcts_q, root value, legal count.
  * ``games.csv``          -- one row per game: outcome, length, hidden cards, king flips, ability counts.

``--report`` then answers the exploration queries in pure Python (no pandas needed): hidden/discard-card
distributions, card-location correlations (e.g. P(Queen in hand | Assassin hidden)), and King's-Hand usage.
The CSVs stay pandas-ready for ad-hoc analysis once pandas is installed.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np

from .. import cards
from ..actions import StepKind
from ..record import dict_to_action, read_jsonl
from ..state import GameState


def _name(cid: Optional[int]) -> Optional[str]:
    return cards.card_name(cid) if cid is not None else None


def _iter_records(data_dir: str, limit: Optional[int] = None):
    n = 0
    for f in sorted(glob.glob(os.path.join(data_dir, "*.jsonl"))):
        for r in read_jsonl(f):
            yield r
            n += 1
            if limit and n >= limit:
                return


CARD_LOC_COLS = ["game_id", "card_id", "name", "value", "ability", "init_location"]
PLY_COLS = ["game_id", "ply", "seat", "phase", "action_kind", "action_card", "guess_name", "target",
            "number", "leading_card", "source_card", "against_card", "mover_z", "mcts_q", "root_value",
            "legal_count"]
GAME_COLS = ["game_id", "winner", "starting_player", "num_decisions", "hidden0", "hidden1",
             "kings0", "kings1", "n_kingshand", "n_assassin_reveal"]


def build(data_dir: str, out_dir: str, limit: Optional[int] = None) -> Dict[str, int]:
    os.makedirs(out_dir, exist_ok=True)
    cardloc: List[Dict] = []
    plyev: List[Dict] = []
    games: List[Dict] = []
    failures = 0

    for r in _iter_records(data_dir, limit):
        gid = r["deal_seed"]
        st = GameState.deal(np.random.default_rng(gid))
        post_setup = None
        try:
            for i, d in enumerate(r["decisions"]):
                if post_setup is None and st.phase == StepKind.MAIN:
                    post_setup = st
                lead = st.leading
                pend = st.pending[-1] if st.pending else None
                cands = d.get("candidates") or []
                tot = sum(c["visits"] for c in cands)
                root_v = sum(c["visits"] * c["mean_q"] for c in cands) / tot if tot else ""
                mq = next((c["mean_q"] for c in cands if c["move"] == d["chosen"]), "")
                ch = d["chosen"]
                plyev.append({
                    "game_id": gid, "ply": i, "seat": d["seat"], "phase": d["phase"],
                    "action_kind": ch["kind"], "action_card": _name(ch.get("card")),
                    "guess_name": ch.get("name"), "target": ch.get("target"), "number": ch.get("number"),
                    "leading_card": _name(lead.card) if lead else None,
                    "source_card": _name(pend.source) if pend and pend.source is not None else None,
                    "against_card": _name(pend.against) if pend and pend.against is not None else None,
                    "mover_z": d.get("z"), "mcts_q": mq, "root_value": root_v,
                    "legal_count": len(st.legal_moves()),
                })
                st = st.apply(dict_to_action(ch))
        except Exception:
            failures += 1
            continue
        gt = post_setup if post_setup is not None else st          # post-setup ground truth
        loc: Dict[int, str] = {}
        for c in gt.hands[0]:
            loc[c] = "hand0"
        for c in gt.hands[1]:
            loc[c] = "hand1"
        for seat in (0, 1):
            if gt.hidden[seat] is not None:
                loc[gt.hidden[seat]] = f"hidden{seat}"
            if gt.setup_discard[seat] is not None:
                loc[gt.setup_discard[seat]] = f"setup_discard{seat}"
        loc[gt.leftover_faceup] = "leftover_faceup"
        loc[gt.leftover_facedown] = "leftover_facedown"
        for cid, l in loc.items():
            cardloc.append({"game_id": gid, "card_id": cid, "name": cards.card_name(cid),
                            "value": cards.card_value(cid), "ability": cards.card_ability(cid).name,
                            "init_location": l})
        games.append({
            "game_id": gid, "winner": r["winner"], "starting_player": r["starting_player"],
            "num_decisions": len(r["decisions"]),
            "hidden0": _name(gt.hidden[0]), "hidden1": _name(gt.hidden[1]),
            "kings0": st.kings[0], "kings1": st.kings[1],          # final flips from terminal state
            "n_kingshand": sum(1 for d in r["decisions"] if d["chosen"]["kind"] == "REVEAL_KINGSHAND"),
            "n_assassin_reveal": sum(1 for d in r["decisions"] if d["chosen"]["kind"] == "REVEAL_ASSASSIN"),
        })

    for name, cols, rows in [("card_locations.csv", CARD_LOC_COLS, cardloc),
                             ("ply_events.csv", PLY_COLS, plyev), ("games.csv", GAME_COLS, games)]:
        with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    return {"games": len(games), "card_rows": len(cardloc), "ply_rows": len(plyev), "failures": failures}


# --- query cookbook (pure Python) -----------------------------------------------------

def _read(path: str) -> List[Dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _top(counter: Counter, total: int) -> str:
    return "  ".join(f"{n}:{c}({c/total:.0%})" for n, c in counter.most_common())


def report(out_dir: str) -> None:
    cl = _read(os.path.join(out_dir, "card_locations.csv"))
    pe = _read(os.path.join(out_dir, "ply_events.csv"))
    games = _read(os.path.join(out_dir, "games.csv"))
    n_games = len(games)
    HID = ("hidden0", "hidden1")
    HAND = ("hand0", "hand1")

    print(f"\n=== exploration report ({n_games} games) ===")

    # (a) hidden / setup-discard distributions
    hidden = Counter(r["name"] for r in cl if r["init_location"] in HID)
    disc = Counter(r["name"] for r in cl if r["init_location"] in ("setup_discard0", "setup_discard1"))
    print("\n(a) what the bot HIDES at setup (per-card, over 2 hidden/game):")
    print("   ", _top(hidden, sum(hidden.values())))
    print("(a) what the bot DISCARDS at setup:")
    print("   ", _top(disc, sum(disc.values())))

    # (a2) card-location correlations, per game
    bygame = defaultdict(dict)
    for r in cl:
        bygame[r["game_id"]][r["name"]] = r["init_location"]
    def P(cond, event):
        sub = [m for m in bygame.values() if cond(m)]
        if not sub:
            return float("nan"), 0
        return sum(1 for m in sub if event(m)) / len(sub), len(sub)
    q_in_hand = lambda m: m.get("Queen") in HAND
    a_hidden = lambda m: m.get("Assassin") in HID
    base_q, _ = P(lambda m: True, q_in_hand)
    cond_q, nA = P(a_hidden, q_in_hand)
    print(f"\n(a2) P(Queen in hand)            = {base_q:.2f}")
    print(f"(a2) P(Queen in hand | Assassin hidden) = {cond_q:.2f}   (over {nA} games w/ Assassin hidden)")
    # a couple more correlations
    kh_hidden = lambda m: m.get("KingsHand") in HID
    base_kh, _ = P(lambda m: True, kh_hidden)
    print(f"(a2) P(King's Hand hidden)       = {base_kh:.2f}   "
          f"| given Assassin hidden = {P(a_hidden, kh_hidden)[0]:.2f}")

    # (b) King's-Hand usage
    kh = [r for r in pe if r["action_kind"] == "REVEAL_KINGSHAND"]
    n_kh_games = sum(1 for g in games if int(g["n_kingshand"]) > 0)
    print(f"\n(b) King's Hand reveal used in {n_kh_games}/{n_games} games ({n_kh_games/n_games:.0%}); "
          f"{len(kh)} reveals total")
    by_phase = Counter(r["phase"] for r in kh)
    print("(b) blocked context (phase):", dict(by_phase),
          "  [REACTION_KINGSHAND=ability, REACTION_KH_VS_ASSASSIN=assassin flip]")
    blocked = Counter((r["source_card"] or r["leading_card"] or "?") for r in kh)
    print("(b) card whose ability/flip was blocked:", _top(blocked, len(kh)) if kh else "(none)")

    # assassin usage for context
    asr = [r for r in pe if r["action_kind"] == "REVEAL_ASSASSIN"]
    print(f"\n(context) Assassin reveal used in "
          f"{sum(1 for g in games if int(g['n_assassin_reveal'])>0)}/{n_games} games; {len(asr)} reveals")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build exploration tables from a self-play corpus + queries.")
    p.add_argument("--data", default=os.path.join("datasets", "selfplay_k20l3"))
    p.add_argument("--out-dir", default=os.path.join("results", "tables_k20l3"))
    p.add_argument("--limit", type=int, default=None, help="only process the first N games (smoke)")
    p.add_argument("--report", action="store_true", help="print the query cookbook after building")
    args = p.parse_args(argv)

    print(f"building tables from {args.data} -> {args.out_dir}"
          + (f" (limit {args.limit})" if args.limit else ""))
    stats = build(args.data, args.out_dir, args.limit)
    print(f"  {stats['games']} games | {stats['card_rows']} card rows | {stats['ply_rows']} ply rows"
          + (f" | {stats['failures']} replay failures" if stats["failures"] else ""))
    if args.report:
        report(args.out_dir)


if __name__ == "__main__":
    main()
