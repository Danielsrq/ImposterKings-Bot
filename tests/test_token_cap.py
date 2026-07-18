"""The `dir:N` per-corpus cap in token_dataset.

`--limit` is a GLOBAL first-N cap; `dir:N` caps ONE corpus so a shallow-search or weaker-labelled corpus can
be down-weighted in the pool without copying shards. Two things must hold, and both are easy to get wrong:

  1. The cap is PER SOURCE DIR -- the counter must not bleed across corpora (the inner decision loop rebinds
     `d`, so the source dir is carried as `src`; a shadow there would count decisions, not games).
  2. `dir:N` must not swallow a Windows drive letter -- `C:\\...` splits on the LAST colon and the suffix is
     not all-digits, so it is NOT a cap.

These use synthetic empty-decision records: a game with no decisions replays trivially (no desync) and counts
as one game producing zero rows -- exactly enough to test the counting without needing valid play.
"""
from __future__ import annotations

import json
import os

from imposterkings.machine_learning import token_dataset as TD


def test_parse_data_splits_cap_but_not_a_windows_drive_letter():
    dirs, caps, preds = TD._parse_data(["datasets/gen1:1000", "datasets/k50"])
    assert dirs == ["datasets/gen1", "datasets/k50"]
    assert caps == {"datasets/gen1": 1000}              # only the ':N' entry is capped
    assert preds == {}                                  # no record predicates given

    # a Windows absolute path must survive intact -- the drive-letter colon is NOT a cap
    dirs, caps, preds = TD._parse_data([r"C:\data\gen1"])
    assert dirs == [r"C:\data\gen1"] and caps == {} and preds == {}
    # ...and a capped absolute path splits on the LAST colon only
    dirs, caps, preds = TD._parse_data([r"C:\data\gen1:500"])
    assert dirs == [r"C:\data\gen1"] and caps == {r"C:\data\gen1": 500}

    # a 'field=value' suffix is a record PREDICATE, not a cap (e.g. select the forced arm)
    dirs, caps, preds = TD._parse_data(["datasets/forced_assassin_hide:arm=forced"])
    assert dirs == ["datasets/forced_assassin_hide"] and caps == {}
    assert preds == {"datasets/forced_assassin_hide": {"arm": "forced"}}


def _corpus(path: str, seeds, per_shard: int = 25) -> None:
    os.makedirs(path, exist_ok=True)
    recs = [{"deal_seed": s, "starting_player": 0, "winner": 0, "rewards": [1.0, -1.0],
             "gen": {"self_play": True}, "decisions": []} for s in seeds]
    for i in range(0, len(recs), per_shard):
        with open(os.path.join(path, f"games_{i // per_shard:05d}.jsonl"), "w", encoding="utf-8") as fh:
            for r in recs[i:i + per_shard]:
                fh.write(json.dumps(r) + "\n")


def test_cap_takes_exactly_N_from_the_capped_corpus_and_all_of_the_rest(tmp_path):
    a = str(tmp_path / "A"); b = str(tmp_path / "B")
    _corpus(a, range(0, 100))                  # 100 games
    _corpus(b, range(1000, 1100))              # 100 games, DISJOINT seeds (no _guard_seed collision)

    stats = TD.build([f"{a}:30", b], str(tmp_path / "out.npz"), feat="v2")
    assert stats["n_games"] == 130             # 30 capped from A + all 100 of B

    meta = json.load(open(str(tmp_path / "out.npz.meta.json"), encoding="utf-8"))
    assert meta["games_per_source"] == {a: 30, b: 100}      # the counter did NOT bleed across dirs
    assert meta["caps"] == {a: 30}


def test_cap_larger_than_the_corpus_is_a_no_op(tmp_path):
    a = str(tmp_path / "A")
    _corpus(a, range(0, 40))                    # only 40 games
    stats = TD.build([f"{a}:999"], str(tmp_path / "out.npz"), feat="v2")
    assert stats["n_games"] == 40              # cap above the corpus size -> take everything


def test_an_uncapped_build_still_works(tmp_path):
    a = str(tmp_path / "A")
    _corpus(a, range(0, 50))
    stats = TD.build(a, str(tmp_path / "out.npz"), feat="v2")
    assert stats["n_games"] == 50
    meta = json.load(open(str(tmp_path / "out.npz.meta.json"), encoding="utf-8"))
    assert meta["caps"] == {}                  # no caps recorded when none given


def _armed_corpus(path: str, seeds, arm_of) -> None:
    """A corpus whose records carry ``gen.arm`` -- like forced_hide's forced|free arms."""
    os.makedirs(path, exist_ok=True)
    recs = [{"deal_seed": s, "starting_player": 0, "winner": 0, "rewards": [1.0, -1.0],
             "gen": {"self_play": True, "arm": arm_of(s)}, "decisions": []} for s in seeds]
    with open(os.path.join(path, "games_00000.jsonl"), "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")


def test_arm_predicate_keeps_only_matching_records(tmp_path):
    a = str(tmp_path / "F")
    _armed_corpus(a, range(0, 20), arm_of=lambda s: "forced" if s % 2 == 0 else "free")

    stats = TD.build([f"{a}:arm=forced"], str(tmp_path / "out.npz"), feat="v2")
    assert stats["n_games"] == 10             # only the 10 forced-arm records survive

    meta = json.load(open(str(tmp_path / "out.npz.meta.json"), encoding="utf-8"))
    assert meta["preds"] == {a: {"arm": "forced"}}    # the predicate is self-documented in the meta
    assert meta["caps"] == {}
