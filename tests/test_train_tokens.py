"""train_tokens smoke: the attention model trains over token rows (loss computes, decreases, and a
checkpoint round-trips). Requires the self-play corpus; skips otherwise. Not a quality claim."""
import glob
import os

import numpy as np
import pytest
import torch

from imposterkings.machine_learning import token_dataset as TD
from imposterkings.machine_learning import train_tokens as TT
from imposterkings.machine_learning.attention_model import (
    AttentionModel, AttnConfig, collate, load, save)
from imposterkings.machine_learning.features import Tokens

DATA = os.path.join("datasets", "selfplay_k20l3")
pytestmark = pytest.mark.skipif(not glob.glob(os.path.join(DATA, "*.jsonl")),
                                reason="self-play corpus not present")


def _rows(tmp_path, limit):
    out = str(tmp_path / f"tok{limit}.npz")
    TD.build(DATA, out, limit=limit)
    return TD.load(out)


def test_smoke_train(tmp_path):
    rows = _rows(tmp_path, 10)
    hp = {"epochs": 2, "batch": 64, "lr": 1e-3, "patience": 5, "val_frac": 0.2, "seed": 0}
    r = TT.run(rows, AttnConfig(d_model=32, n_heads=4, ffn_hidden=64), hp)
    assert np.isfinite(r["train_mse"]) and np.isfinite(r["val_mse"])
    assert r["seconds"] > 0 and r["epochs"] <= 2 and r["rows"] == len(rows)
    for k in ("top1_bestq", "top1_chosen", "recall2", "spearman"):
        assert k in r


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    m = AttentionModel(AttnConfig(d_model=32)).eval()
    p = str(tmp_path / "m.pt")
    save(p, m, meta={"tag": 7})
    m2, meta = load(p)
    assert meta["tag"] == 7 and m2.cfg == m.cfg
    rows = _rows(tmp_path, 2)
    c, b, ph, a = rows.tokens(0)
    packed = collate([Tokens(c, b, ph, a, [])])
    args = (packed["cards"], packed["board"], packed["phase"], packed["action"], packed["card_mask"])
    with torch.no_grad():
        q1, _ = m(*args)
        q2, _ = m2(*args)
    assert torch.allclose(q1, q2)                       # weights preserved


def test_loss_decreases(tmp_path):
    rows = _rows(tmp_path, 4)
    torch.manual_seed(0)
    idx = np.arange(min(64, len(rows)))
    packed, y, w = TT.collate_fn([TT.TokenTorchDataset(rows, idx)[i] for i in range(len(idx))])
    model = AttentionModel(AttnConfig(d_model=32))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    def wmse():
        with torch.no_grad():
            return float((w * (TT._forward(model, packed) - y) ** 2).sum() / w.sum().clamp_min(1e-8))

    start = wmse()
    for _ in range(60):
        opt.zero_grad()
        loss = (w * (TT._forward(model, packed) - y) ** 2).sum() / w.sum().clamp_min(1e-8)
        loss.backward(); opt.step()
    assert wmse() < start                               # the model can fit a tiny batch
