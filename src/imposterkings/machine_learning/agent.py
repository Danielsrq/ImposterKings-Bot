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
    """Pick / score legal moves by the model's predicted action-value ``q`` (no search).

    Handles BOTH model families. The MLP flattens the view into one 216-dim vector (``features.encode``);
    the attention net tokenizes it (``features2`` / ``features``) and scores one (state, action) pair per
    row. Both are driven from the ``InformationSet`` -- never a ``GameState`` -- so greedy play sees exactly
    what the mover sees and cannot leak the opponent's hand. (``attention_model.evaluator_from_model``
    takes a *state* because MCTS calls it on an already-determinized node; that seam is wrong for greedy.)
    """

    def __init__(self, model, device: str = "cpu") -> None:
        self.model = model.eval()
        self.device = device
        self.attn = hasattr(model, "cfg")                   # AttnConfig -> the attention net
        self.feat = getattr(getattr(model, "cfg", None), "feat", None)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "NNPolicy":
        blob = torch.load(path, map_location=device, weights_only=False)
        if blob.get("model_type") == "attention":
            from .attention_model import load as load_attn
            model, _ = load_attn(path, device)
            return cls(model, device)
        model, _ = load_checkpoint(path, device)
        return cls(model, device)

    def _q(self, view: InformationSet, moves: List[Action]) -> np.ndarray:
        if not self.attn:
            x = torch.from_numpy(np.stack([features.encode(view, m) for m in moves])).to(self.device)
            with torch.no_grad():
                return self.model(x).squeeze(-1).cpu().numpy()

        if self.feat == "v2":
            from .attention_model import collate2
            from .features2 import tokenize_state, with_action
            st = tokenize_state(view, legal_moves=moves)     # featurize the position ONCE...
            b = collate2([with_action(st, m) for m in moves])   # ...then stamp each candidate action
            with torch.no_grad():
                return self.model(b["cards"], b["board"], b["phase"], b["action"],
                                  kings=b["kings"])[0].cpu().numpy()

        from .attention_model import collate
        from .features import tokenize_state, with_action
        st = tokenize_state(view, legal_moves=moves)
        b = collate([with_action(st, m) for m in moves])
        with torch.no_grad():
            return self.model(b["cards"], b["board"], b["phase"], b["action"],
                              b["card_mask"])[0].cpu().numpy()

    def evaluate(self, view: InformationSet) -> List[Tuple[Action, float]]:
        """Every legal move with its predicted q (one batched forward pass)."""
        moves = view.legal_moves()
        return list(zip(moves, (float(v) for v in self._q(view, moves))))

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
