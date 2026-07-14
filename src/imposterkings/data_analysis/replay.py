"""Replay a recorded corpus back through the engine, one ply at a time.

A ``GameRecord`` stores only ``(deal_seed, [chosen actions])`` -- that is enough to reconstruct the game
exactly, because ``GameState.deal(default_rng(seed))`` is deterministic and ``apply`` has no rng. But the
*interesting* facts are not in the log: what a King's Hand countered, whether a guess was correct, whether
the Warlord's royalty bonus actually fired -- all of those live in the ENGINE STATE at the moment of the
action, not in the recorded action dict. So we replay, and hand each ply back with the state that produced it.

Four near-duplicate replay loops already exist (``token_dataset.py``, ``dataset.py``, ``build_tables.py``,
and datagen's own). This is the one they should all have shared.

**The canary is not optional.** ``GameState.apply`` does NOT validate legality -- ``abilities.resolve``
simply dispatches on ``pending[-1].kind`` -- so an action recorded under a different ruleset can *silently
corrupt* the replay rather than raise. (It happens to raise ~99% of the time, which is worse: it looks safe.)
The gen-0 corpus desyncs on 449/2000 games after the Mystic-muting fix, and ``build_tables.py`` swallowed
446 of them in a bare ``except`` -- so its published tables were quietly built from 1,554 games, not 2,000.
Here a desynced game is dropped, COUNTED, and reported.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Union

import numpy as np

from ..actions import Action, StepKind
from ..record import dict_to_action, read_jsonl
from ..state import GameState


def shards(data: Union[str, Sequence[str]]) -> List[str]:
    """Every ``*.jsonl`` shard across one or more corpus dirs."""
    dirs = [data] if isinstance(data, str) else list(data)
    return sorted(f for d in dirs for f in glob.glob(os.path.join(d, "*.jsonl")))


def iter_records(data: Union[str, Sequence[str]], limit: Optional[int] = None,
                 progress: bool = True) -> Iterator[dict]:
    """Yield raw game records across the given corpus dirs, oldest shard first."""
    files = shards(data)
    if progress:
        try:
            from tqdm import tqdm
            files = tqdm(files, desc="shards", unit="shard")
        except Exception:
            pass
    n = 0
    for f in files:
        for rec in read_jsonl(f):
            yield rec
            n += 1
            if limit and n >= limit:
                return


@dataclass(frozen=True)
class Ply:
    """One decision, with the state that PRECEDES it.

    ``state`` is pre-apply on purpose: every engine predicate worth asking (``royalty_present()``,
    ``active_royalty()``, ``effective_hand_value()``, "does the defender hold the guessed name") is a
    question about the position the mover faced, not the one they created.
    """
    index: int
    seat: int
    state: GameState                       # BEFORE the action
    action: Action
    decision: dict                         # the raw record (phase, chosen, candidates, sims, z, ...)

    @property
    def step(self):
        """The pending step being answered -- carries ``source``/``against``/``guess``/``picked``."""
        return self.state.pending[-1]

    @property
    def phase(self) -> StepKind:
        return self.state.phase

    @property
    def candidates(self) -> List[dict]:
        """The search's candidate moves. EMPTY for a forced ply -- and also for the unlabelled opponent
        in an asymmetric corpus (datagen passes ``agent=None``), so absence is not evidence of forcedness."""
        return self.decision.get("candidates") or []

    def preferred(self) -> Optional[dict]:
        """The move the search actually PREFERRED (argmax visit share).

        Not the same as the move played: datagen ran with ``temp_plies: 6``, so the first six searched plies
        -- which includes both setup decisions -- were SAMPLED from the visit distribution. ``chosen`` is a
        draw; this is the policy.
        """
        c = self.candidates
        return max(c, key=lambda x: x["visit_share"]) if c else None


@dataclass
class Game:
    """A replayed game. ``desynced`` games have a truncated ``plies`` list and must be discarded."""
    rec: dict
    plies: List[Ply] = field(default_factory=list)
    dealt: Optional[GameState] = None      # the position straight out of deal(), before any setup choice
    terminal: Optional[GameState] = None   # None iff desynced
    desynced: bool = False
    desync_at: Optional[int] = None

    @property
    def seed(self) -> int:
        return self.rec["deal_seed"]

    @property
    def winner(self) -> int:
        return self.rec["winner"]

    @property
    def rewards(self) -> List[float]:
        return self.rec["rewards"]

    def won(self, seat: int) -> bool:
        return self.rewards[seat] > 0

    @property
    def starting_player(self) -> int:
        return self.rec["starting_player"]

    def second_player(self) -> int:
        return 1 - self.starting_player

    @property
    def gen(self) -> dict:
        return self.rec.get("gen") or {}

    def agent_of(self, seat: int) -> str:
        """Which agent played this seat.

        Self-play corpora have one agent on both seats. Asymmetric ones record ``primary_seat`` (the
        labelled agent) and ``opponent_spec``; the seat alternates with the deal seed, so it is NOT the
        same thing as ``starting_player`` -- conflating them would silently blend two different bots.
        """
        g = self.gen
        if g.get("self_play", True):
            return g.get("spec", "?")
        return g["spec"] if seat == g["primary_seat"] else (g.get("opponent_spec") or "?")

    def hands_dealt(self) -> List[tuple]:
        return [self.dealt.hands[0], self.dealt.hands[1]]

    def setup_choice(self, seat: int, kind: StepKind) -> Optional[int]:
        """The card this seat hid / discarded at setup (both are recorded plies, for BOTH seats)."""
        for p in self.plies[:4]:
            if p.seat == seat and p.phase == kind:
                return p.action.card
        return None


def replay(rec: dict) -> Game:
    """Re-derive a game from its seed and its recorded moves, keeping the pre-apply state at every ply."""
    # Pass the RECORDED starter rather than letting deal() re-flip its coin. For every ordinary corpus this
    # is a no-op (the recorded starter IS that coin flip -- verified on 2,000 seeds), but a corpus that
    # MIRRORS a deal by forcing the starter (forced_hide.py) is otherwise unreplayable for half its games:
    # deal() draws the permutation first and the coin second, so the hands match and only the starter drifts.
    st = GameState.deal(np.random.default_rng(rec["deal_seed"]), starting_player=rec.get("starting_player"))
    g = Game(rec=rec, dealt=st)
    for i, d in enumerate(rec["decisions"]):
        a = dict_to_action(d["chosen"])
        if a not in st.legal_moves():          # THE canary: apply() would not catch this
            g.desynced, g.desync_at = True, i
            return g
        g.plies.append(Ply(index=i, seat=d["seat"], state=st, action=a, decision=d))
        st = st.apply(a)
    g.terminal = st
    return g


def replay_all(data: Union[str, Sequence[str]], limit: Optional[int] = None,
               progress: bool = True) -> Iterator[Game]:
    """Replay a corpus, dropping desynced games and reporting how many. Never silently swallows."""
    n = dropped = 0
    for rec in iter_records(data, limit, progress):
        g = replay(rec)
        n += 1
        if g.desynced:
            dropped += 1
            continue
        yield g
    if dropped:
        print(f"  WARNING: dropped {dropped}/{n} game(s) ({dropped / n:.1%}) whose recorded actions are no "
              f"longer legal -- the corpus predates an engine rules change. Every statistic below is "
              f"computed on the remaining {n - dropped}, and is biased toward the games the change did not "
              f"touch.")


def desync_report(data: Union[str, Sequence[str]], limit: Optional[int] = None) -> Dict[str, int]:
    """Replay-only integrity check: how many games no longer reproduce, and where they break."""
    from collections import Counter
    n = dropped = 0
    where: Counter = Counter()
    for rec in iter_records(data, limit, progress=False):
        g = replay(rec)
        n += 1
        if g.desynced:
            dropped += 1
            where[rec["decisions"][g.desync_at]["phase"]] += 1
    return {"games": n, "desynced": dropped, "by_phase": dict(where)}
