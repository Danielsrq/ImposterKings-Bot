"""Decision/game records and JSONL persistence -- the seam for the future NN pipeline.

Scaffolded in Phase 3 (no training yet): self-play with the MCTS agent can emit, per decision, the
candidate visit/Q statistics (the policy target) plus the back-filled terminal reward ``z`` -- the
AlphaZero-style label set a later plan will encode and learn from. Mirrors bigtwo's record.py shape.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np

from .actions import Action
from .explain import format_action
from .infoset import InformationSet
from .state import GameState


def action_to_dict(a: Action) -> dict:
    d = {"kind": a.kind.name}
    for k in ("card", "target", "number", "name"):
        v = getattr(a, k)
        if v is not None:
            d[k] = v
    return d


@dataclass
class Candidate:
    move: dict
    move_str: str
    visits: int
    mean_q: float
    visit_share: float


@dataclass
class DecisionRecord:
    seat: int
    phase: str
    chosen: dict
    candidates: List[Candidate]
    sims: int
    elapsed_ms: float
    z: Optional[float] = None  # back-filled terminal reward for the deciding seat

    @classmethod
    def build(cls, seat: int, view: InformationSet, move: Action, agent) -> "DecisionRecord":
        result = getattr(agent, "last_result", None)
        candidates: List[Candidate] = []
        sims, elapsed_ms = 0, 0.0
        if result is not None:
            sims, elapsed_ms = result.iterations, result.elapsed * 1000.0
            candidates = [
                Candidate(action_to_dict(s.move), format_action(s.move), s.visits, s.mean_q, s.visit_share)
                for s in result.stats
            ]
        return cls(seat=seat, phase=view.pending[-1].kind.name, chosen=action_to_dict(move),
                   candidates=candidates, sims=sims, elapsed_ms=elapsed_ms)


@dataclass
class GameRecord:
    decisions: List[DecisionRecord] = field(default_factory=list)
    winner: Optional[int] = None
    rewards: Optional[List[float]] = None


def play_and_record(agents, rng: np.random.Generator,
                    starting_player: Optional[int] = None) -> GameRecord:
    """Play one game, capturing a DecisionRecord per ply with the terminal reward back-filled."""
    record = GameRecord()
    state = GameState.deal(rng, starting_player=starting_player)
    while not state.is_terminal():
        seat = state.to_play
        view = state.information_set(seat)
        move = agents[seat].select_move(view, rng)
        record.decisions.append(DecisionRecord.build(seat, view, move, agents[seat]))
        state = state.apply(move)
    record.winner = state.winner
    record.rewards = state.result()
    for d in record.decisions:
        d.z = record.rewards[d.seat]
    return record


def write_jsonl(path: str, records: List[GameRecord]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")
