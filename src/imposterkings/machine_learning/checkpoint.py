"""Save/load a trained MLP as a self-describing checkpoint (torch). One format for train, agent, and UI."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from .mlp import MLP


def save(path: str, model: MLP, meta: Optional[Dict] = None) -> None:
    torch.save({"arch": list(model.hidden_dims), "feature_dim": model.in_dim, "target": "q",
                "state_dict": model.state_dict(), "meta": meta or {}}, path)


def load(path: str, device: str = "cpu") -> Tuple[MLP, Dict]:
    b = torch.load(path, map_location=device, weights_only=False)
    model = MLP(b["feature_dim"], b["arch"])
    model.load_state_dict(b["state_dict"])
    model.eval()
    return model, b.get("meta", {})
