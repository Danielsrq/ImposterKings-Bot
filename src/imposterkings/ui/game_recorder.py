"""Append human-vs-AI games played in ``ui.app`` to a JSONL corpus the replay/corpus_stats pipeline reads.

Reuses ``record.py``'s ``GameRecord``/``DecisionRecord`` serialization verbatim, so a recorded human game
replays through ``data_analysis.replay`` exactly like a self-play game (``deal_seed`` + ``starting_player``
+ the ordered ``chosen`` actions reproduce it). One JSON line per finished game, appended to a per-session
file under the (already gitignored) datasets dir.

What we DON'T record, because it can't be replayed from a seed alone:
- hotseat games (no bot -- both seats are the human),
- scenario / custom-start games (built from an explicit initial state, not a deal).

The bot's identity (engine mode + budget + checkpoint) rides in ``gen["bot"]`` so the ``bot_on_seed``
utility can rebuild the exact opponent and replay the same deal. ``gen["hint_used"]`` flags games where the
Hint panel was visible while a move was committed (or was ever opened), so those can be filtered as assisted.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import List, Optional

from ..record import DecisionRecord, GameRecord


class _Res:
    """Adapter so ``DecisionRecord.build`` (which reads ``agent.last_result``) can take a bare SearchResult
    (the bot's search) or ``None`` (a human ply -> empty candidates)."""

    __slots__ = ("last_result",)

    def __init__(self, result):
        self.last_result = result


class GameRecorder:
    def __init__(self, out_dir: str = os.path.join("datasets", "human"), enabled: bool = True):
        self.out_dir = out_dir
        self.enabled = enabled
        self.path: Optional[str] = None            # the session file, created on the first finished game
        self._reset()

    def _reset(self) -> None:
        self._armed = False
        self._done = False
        self._seed: Optional[int] = None
        self._human_seat: Optional[int] = None
        self._bot: Optional[dict] = None
        self._decisions: List[DecisionRecord] = []
        self._hint_used = False
        self._assists = {"hint_panel": False, "analysis_drawer": False, "bot_reasoning": False}

    # -- lifecycle ---------------------------------------------------------------------------------------

    def start(self, *, seed, human_seat: int, bot: dict, hotseat: bool = False,
              scenario: bool = False) -> None:
        """Begin recording a new game. Disarms for hotseat / scenario / a missing seed / when disabled."""
        self._reset()
        self._armed = self.enabled and not hotseat and not scenario and seed is not None
        if self._armed:
            self._seed = int(seed)
            self._human_seat = human_seat
            self._bot = dict(bot)

    def log_ply(self, seat: int, view, move, result) -> None:
        """One committed ply. ``result`` = the bot's SearchResult on a bot ply (captures candidates), else
        ``None`` (human plies stay policy-free, which ``replay.Ply.candidates`` tolerates)."""
        if self._armed and not self._done:
            self._decisions.append(DecisionRecord.build(seat, view, move, _Res(result)))

    def snapshot_assists(self, hint: bool, analysis: bool, reasoning: bool) -> None:
        """OR the assist panels' visibility (at the moment of a move) into the sticky per-game flags."""
        if not self._armed:
            return
        if hint:
            self._hint_used = True
            self._assists["hint_panel"] = True
        if analysis:
            self._assists["analysis_drawer"] = True
        if reasoning:
            self._assists["bot_reasoning"] = True

    def mark_hint(self) -> None:
        """Catch a Hint panel that was opened then closed again before a move was made."""
        if self._armed:
            self._hint_used = True
            self._assists["hint_panel"] = True

    def finalize(self, state) -> None:
        """Append the game if it's armed, terminal, and not yet written (idempotent -- safe to call every
        frame, on New Game, and at quit)."""
        if not self._armed or self._done or not state.is_terminal():
            return
        rewards = list(state.result())
        rec = GameRecord(
            gen={
                "human": True, "self_play": False,
                "primary_seat": self._human_seat,          # the human's seat (for per-agent labelling)
                "spec": "human",
                "opponent_spec": self._bot.get("label"),
                "bot": self._bot,                          # full opponent config -> bot_on_seed rebuilds it
                "value_ckpt": self._bot.get("nn_ckpt"),
                "hint_used": self._hint_used,
                "assists": dict(self._assists),
                "recorded_at": datetime.now().isoformat(timespec="seconds"),
            },
            deal_seed=self._seed,
            starting_player=state.starting_player,
            decisions=self._decisions,
            winner=state.winner,
            rewards=rewards,
        )
        for d in rec.decisions:
            d.z = rewards[d.seat]
        self._append(rec)
        self._done = True

    # -- io ----------------------------------------------------------------------------------------------

    def _append(self, rec: GameRecord) -> None:
        if self.path is None:
            os.makedirs(self.out_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.path = os.path.join(self.out_dir, f"session_{stamp}.jsonl")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec)) + "\n")
        print(f"recorded game (seed {rec.deal_seed}, winner P{rec.winner}"
              f"{', hint used' if rec.gen['hint_used'] else ''}) -> {self.path}")
