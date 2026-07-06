"""Build an ISMCTS evaluator from a trained checkpoint: leaf -> (per-seat value, policy prior).

The returned callable is what `mcts.SearchConfig.evaluator` expects. It runs the tiny MLP in **numpy**
(torch is used only to load the checkpoint), so the search stays torch-free and avoids per-call torch
dispatch overhead across the millions of leaf evaluations in self-play.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from ..actions import Action
from ..rules import NUM_PLAYERS
from . import features
from .checkpoint import load as _load


def build_evaluator(checkpoint: str, device: str = "cpu") -> Callable:
    """`state -> ([per-seat value], {move: prior})`. Value = max_a Q (mover-relative), prior = softmax(Q)."""
    model, _ = _load(checkpoint, device)
    layers = [(lin.weight.detach().cpu().numpy().astype(np.float32),
               lin.bias.detach().cpu().numpy().astype(np.float32))
              for lin in model.net if hasattr(lin, "weight")]        # nn.Linear layers, in order

    def _forward(x: np.ndarray) -> np.ndarray:                       # ReLU between layers, tanh at the end
        for i, (w, b) in enumerate(layers):
            x = x @ w.T + b
            x = np.maximum(x, 0.0) if i < len(layers) - 1 else np.tanh(x)
        return x[:, 0]

    def evaluate(state) -> Tuple[List[float], Dict[Action, float]]:
        mover = state.to_play
        view = state.information_set(mover)
        moves = state.legal_moves()
        q = _forward(np.stack([features.encode(view, m) for m in moves]).astype(np.float32))
        v = float(q.max())
        value = [0.0] * NUM_PLAYERS
        value[mover] = v
        value[1 - mover] = -v                                        # zero-sum leaf value, like result()
        e = np.exp(q - q.max())
        priors = {m: float(p) for m, p in zip(moves, e / e.sum())}
        return value, priors

    return evaluate
