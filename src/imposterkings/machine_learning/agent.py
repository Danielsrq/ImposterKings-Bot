"""The reusable NN eval-move-picker (``NNPolicy``) and an ``Agent``-protocol wrapper (``NNAgent``).

``NNPolicy`` is the shared component: given a trained model it scores/orders the legal moves by predicted
``q`` and returns the best one -- used by ``NNAgent`` (self-play / benchmarking) AND by ``ui.app`` to seat
a checkpoint as a live bot. ``NNAgent`` just adapts it to ``arena``'s ``select_move(view, rng)`` seam, so no
change to the arena or app move loop is needed.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from ..actions import Action
from ..infoset import InformationSet
from . import features
from .checkpoint import load as load_checkpoint


class NNPolicy:
    """Pick / score legal moves by the model's predicted action-value ``q`` (no search)."""

    def __init__(self, model, device: str = "cpu") -> None:
        self.model = model.eval()
        self.device = device

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "NNPolicy":
        model, _ = load_checkpoint(path, device)
        return cls(model, device)

    def evaluate(self, view: InformationSet) -> List[Tuple[Action, float]]:
        """Every legal move with its predicted q (one batched forward pass)."""
        moves = view.legal_moves()
        x = torch.from_numpy(np.stack([features.encode(view, m) for m in moves])).to(self.device)
        with torch.no_grad():
            q = self.model(x).squeeze(-1).cpu().numpy()
        return list(zip(moves, (float(v) for v in q)))

    def best_move(self, view: InformationSet) -> Action:
        moves = view.legal_moves()
        if len(moves) == 1:                     # forced -> no forward pass
            return moves[0]
        return max(self.evaluate(view), key=lambda mq: mq[1])[0]


class NNAgent:
    """A greedy NN policy as an agent: play the highest-predicted-q legal move. Duck-types ``Agent``."""

    def __init__(self, policy: NNPolicy, name: str = "nn") -> None:
        self.policy = policy
        self.name = name
        self.last_result = None                 # interface parity with MCTSAgent

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu", name: str = "nn") -> "NNAgent":
        return cls(NNPolicy.from_checkpoint(path, device), name)

    def select_move(self, view: InformationSet, rng: np.random.Generator) -> Action:
        return self.policy.best_move(view)
