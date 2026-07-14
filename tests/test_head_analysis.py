"""Causal per-head attention entropy.

The whole point of this module is that the OBVIOUS measurement is wrong. Averaging the attention map over
every query row makes every head look diffuse (87-92% of uniform) -- because at L=2 the final layer's card
rows are computed and then DISCARDED, and they dominate the average. Restricted to the rows whose output
actually reaches the q-head, the same heads are far sharper (down to 26%).

So the tests here pin the ROW SELECTION, not the arithmetic: if someone "simplifies" this back to a mean
over all rows, the headline finding of Study 5 (that d96's extra heads specialize rather than dilute)
silently becomes unmeasurable.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from imposterkings.machine_learning import attention_model as AM  # noqa: E402
from imposterkings.machine_learning.head_analysis import analyse, seq_labels  # noqa: E402


class _Rows:
    """The minimum TokenRows surface analyse() touches (v2: fixed 18 cards + kings)."""

    def __init__(self, n=64):
        rng = np.random.default_rng(0)
        self.feat = "v2"
        self.n = n
        self.cards = rng.standard_normal((n, 18, 46)).astype(np.float32)
        self.kings = rng.standard_normal((n, 2, 4)).astype(np.float32)
        self.board = rng.standard_normal((n, 4)).astype(np.float32)
        self.phase = rng.standard_normal((n, 15)).astype(np.float32)
        self.action = rng.standard_normal((n, 51)).astype(np.float32)
        self.y = rng.uniform(-1, 1, n).astype(np.float32)
        self.w = np.ones(n, np.float32)
        self.game_id = np.arange(n) // 4
        self.decision_id = np.arange(n)
        self.is_chosen = np.zeros(n, np.int64)

    def __len__(self):
        return self.n

    def tokens(self, i):
        return (self.cards[i], self.kings[i], self.board[i], self.phase[i], self.action[i])


def _ckpt(tmp_path, heads, d_model):
    torch.manual_seed(0)
    m = AM.AttentionModel(AM.AttnConfig(d_model=d_model, n_layers=2, n_heads=heads, feat="v2"))
    p = tmp_path / f"attn_d{d_model}_L2.pt"
    AM.save(str(p), m)
    return str(p)


def test_the_sequence_labels_match_the_v2_layout():
    """S=24: [CLS, 18 cards, 2 kings, board, phase, action]. A mislabeled axis would mis-attribute every
    head's attention -- the 'this head watches the Assassin' claim rests entirely on this ordering."""
    lab = seq_labels("v2")
    assert len(lab) == 24
    assert lab[0] == "CLS" and lab[-1] == "ACTION"
    assert lab[17] == "Assassin#16" and lab[18] == "Fool#17"       # cards occupy 1..18
    assert lab[19:22] == ["king_mine", "king_theirs", "board"]


def test_entropy_is_measured_on_the_CAUSAL_rows_only(tmp_path):
    """The load-bearing test. The final layer contributes ONLY through its CLS row; earlier layers through
    the card rows the next layer reads. Measuring over all rows instead is the mistake this module exists
    to correct, and it changes the answer by tens of points."""
    rows, idx = _Rows(), np.arange(64)
    path = _ckpt(tmp_path, heads=4, d_model=32)
    r = analyse(path, rows, idx, batch=32)

    assert len(r["entropy"]) == 2                              # one per layer
    assert all(e.shape == (4,) for e in r["entropy"])          # one per head
    assert all(0.0 <= v <= 1.0 for e in r["entropy"] for v in e)

    # Recompute the FINAL layer by hand from the raw map, restricted to the CLS row, and demand a match.
    model, _ = AM.load(path)
    model.eval()
    from imposterkings.machine_learning.train_tokens import collate_fn
    batch = collate_fn([(*rows.tokens(i), float(rows.y[i]), float(rows.w[i])) for i in range(64)])[0]
    with torch.no_grad():
        _, attns = model.forward_layers(batch["cards"], batch["board"], batch["phase"], batch["action"],
                                        batch["card_mask"], kings=batch.get("kings"))
    A = attns[-1].numpy()
    S = A.shape[-1]
    cls_row = A[:, :, 0:1, :]                                  # <- the ONLY row the q-head reads
    want = (-(cls_row * np.log(cls_row + 1e-12)).sum(-1) / math.log(S)).mean(axis=(0, 2))
    assert np.abs(r["entropy"][-1] - want).max() < 1e-5

    all_rows = (-(A * np.log(A + 1e-12)).sum(-1) / math.log(S)).mean(axis=(0, 2))
    assert not np.allclose(all_rows, want, atol=1e-3), \
        "if averaging over ALL rows gives the same answer, the causal selection is not doing anything"


def test_attention_mass_is_a_distribution_over_the_sequence(tmp_path):
    """'What each head reads' must be a real probability distribution over the 24 tokens, or the reported
    percentages ('board: 71%') are meaningless."""
    rows, idx = _Rows(), np.arange(64)
    r = analyse(_ckpt(tmp_path, heads=4, d_model=32), rows, idx, batch=32)
    mass = r["mass"]
    assert mass.shape == (4, 24)                               # [heads, sequence]
    assert np.allclose(mass.sum(axis=1), 1.0, atol=1e-4)       # softmax rows -> each head sums to 1
    assert (mass >= 0).all()
    assert len(r["labels"]) == mass.shape[1]                   # one label per column


def test_an_untrained_net_is_near_uniform(tmp_path):
    """The reference point every number is read against: an untrained head attends to everything equally
    (~99% of uniform). Specialization is movement DOWN from here, so if this drifts, every entropy figure
    in the study log loses its meaning."""
    rows, idx = _Rows(), np.arange(64)
    r = analyse(_ckpt(tmp_path, heads=4, d_model=32), rows, idx, batch=32)
    for e in r["entropy"]:
        assert e.min() > 0.90, "an untrained head should be ~uniform; a low value means the metric is broken"
