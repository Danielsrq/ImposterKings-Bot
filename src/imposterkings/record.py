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

from .actions import Action, ActionKind
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


def dict_to_action(d: dict) -> Action:
    """Inverse of :func:`action_to_dict` -- rebuild an Action from its JSON dict (round-trips exactly)."""
    return Action(kind=ActionKind[d["kind"]], card=d.get("card"), target=d.get("target"),
                  number=d.get("number"), name=d.get("name"))


@dataclass
class Candidate:
    move: dict
    move_str: str
    visits: int
    mean_q: float
    visit_share: float
    solved: Optional[float] = None      # godmode: EXACT value (mover perspective, +/-1); None for MCTS
    win: Optional[bool] = None          # godmode: is this move a proven win for the mover


@dataclass
class DecisionRecord:
    seat: int
    phase: str
    chosen: dict
    candidates: List[Candidate]
    sims: int
    elapsed_ms: float
    z: Optional[float] = None  # back-filled terminal reward for the deciding seat
    eval_type: str = "mcts"             # "mcts" | "godmode" -- how candidates were valued
    solved_value: Optional[float] = None   # godmode: the position's EXACT value (mover perspective)

    @classmethod
    def build(cls, seat: int, view: InformationSet, move: Action, agent) -> "DecisionRecord":
        result = getattr(agent, "last_result", None)
        candidates: List[Candidate] = []
        sims, elapsed_ms = 0, 0.0
        eval_type, solved_value = "mcts", None
        if result is not None:
            sims, elapsed_ms = result.iterations, result.elapsed * 1000.0
            eval_type = getattr(result, "eval_type", "mcts")
            god = eval_type == "godmode"
            solved_value = result.root_value() if god else None
            candidates = [
                Candidate(action_to_dict(s.move), format_action(s.move), s.visits, s.mean_q, s.visit_share,
                          solved=(s.mean_q if god else None), win=(getattr(s, "win", None) if god else None))
                for s in result.stats
            ]
        return cls(seat=seat, phase=view.pending[-1].kind.name, chosen=action_to_dict(move),
                   candidates=candidates, sims=sims, elapsed_ms=elapsed_ms,
                   eval_type=eval_type, solved_value=solved_value)


@dataclass
class GameRecord:
    # replayable header: (deal_seed, ordered decisions[].chosen) fully reconstructs the game, since
    # GameState.deal(default_rng(deal_seed)) + apply(actions) is deterministic (state.apply has no rng).
    schema_version: int = 2             # v2: DecisionRecord gains eval_type + solved_value (godmode)
    gen: Optional[dict] = None          # generator meta: spec label, mode/k/l, temp_plies, base_seed
    deal_seed: Optional[int] = None
    starting_player: Optional[int] = None
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
        agent = agents[seat]
        if getattr(agent, "perfect_info", False):      # godmode gets the true state; others stay view-only
            move = agent.select_move(view, rng, state=state)
        else:
            move = agent.select_move(view, rng)
        record.decisions.append(DecisionRecord.build(seat, view, move, agent))
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


def read_jsonl(path: str) -> List[dict]:
    """Load game records back as plain dicts (one JSON object per line)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
