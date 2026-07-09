"""Shape/forward smoke tests for the attention model: token embedding -> CLS attention -> q + the
per-head N x N attention and CLS->card importance readout. Untrained (random init) -- we check plumbing,
masking, bounds, parameterizability and determinism, not learned behavior."""
import numpy as np
import pytest
import torch

from imposterkings.infoset import InformationSet
from imposterkings.machine_learning import features as F
from imposterkings.machine_learning.attention_model import (
    AttentionModel, AttnConfig, cls_importance, collate)
from imposterkings.state import GameState


def _views(specs=((0, 2), (1, 5), (2, 8), (3, 11))):
    """The acting-seat InfoSet after `k` random plies, for each (seed, k) -- gives varied mid-game N."""
    out = []
    for seed, k in specs:
        s = GameState.deal(np.random.default_rng(seed))
        rng = np.random.default_rng(seed + 100)
        last = InformationSet.from_state(s, s.to_play)
        for _ in range(k):
            if s.is_terminal():
                break
            v = InformationSet.from_state(s, s.to_play)
            last = v
            lm = v.legal_moves()
            s = s.apply(lm[int(rng.integers(len(lm)))])
        out.append(last)
    return out


def _batch(views, action=None):
    return collate([F.tokenize(v, action) for v in views])


def test_forward_shapes_and_bounded():
    torch.manual_seed(0)
    b = _batch(_views())
    model = AttentionModel(AttnConfig()).eval()
    with torch.no_grad():
        q, attn = model(b["cards"], b["board"], b["phase"], b["action"], b["card_mask"])
    B, nmax = b["cards"].shape[:2]
    assert q.shape == (B,)
    assert (q >= -1).all() and (q <= 1).all()                      # Tanh-bounded
    assert attn.shape == (B, 4, nmax + 4, nmax + 4)                # S = N + 4, 4 heads


def test_attention_rows_are_distributions():
    torch.manual_seed(0)
    b = _batch(_views())
    model = AttentionModel(AttnConfig()).eval()
    with torch.no_grad():
        _, attn = model(b["cards"], b["board"], b["phase"], b["action"], b["card_mask"])
    assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-5)


def test_mask_zeros_padded_keys():
    """Deterministic masking check: manually pad 3 card slots -> they must receive ~0 attention."""
    torch.manual_seed(0)
    tok = F.tokenize(_views([(0, 6)])[0])
    n = tok.cards.shape[0]
    nmax = n + 3
    cards = torch.zeros(1, nmax, F.CARD_DIM)
    cards[0, :n] = torch.from_numpy(tok.cards)
    mask = torch.zeros(1, nmax, dtype=torch.bool)
    mask[0, :n] = True
    board = torch.from_numpy(tok.board)[None]
    phase = torch.from_numpy(tok.phase)[None]
    action = torch.from_numpy(tok.action)[None]
    model = AttentionModel(AttnConfig()).eval()
    with torch.no_grad():
        q, attn = model(cards, board, phase, action, mask)
    assert attn[0, :, :, 1 + n:1 + nmax].abs().max() < 1e-6         # padded key columns ~ 0
    assert torch.isfinite(q).all()


def test_cls_importance_labels_aligned():
    torch.manual_seed(0)
    views = _views()
    toks = [F.tokenize(v) for v in views]
    b = collate(toks)
    model = AttentionModel(AttnConfig()).eval()
    with torch.no_grad():
        _, attn = model(b["cards"], b["board"], b["phase"], b["action"], b["card_mask"])
    imp = cls_importance(attn, b["labels"])
    assert len(imp) == len(views)
    for pairs, t in zip(imp, toks):
        assert len(pairs) == len(t.labels)                         # one weight per real card token
        assert {lb for lb, _ in pairs} == set(t.labels)            # labels aligned, none from pad slots


def test_parameterizable():
    torch.manual_seed(0)
    b = _batch(_views())
    model = AttentionModel(AttnConfig(d_model=128, n_layers=2, n_heads=8, ffn_hidden=256)).eval()
    with torch.no_grad():
        q, attn = model(b["cards"], b["board"], b["phase"], b["action"], b["card_mask"])
    assert attn.shape[1] == 8                                      # n_heads tracks config
    assert q.shape == (b["cards"].shape[0],)
    assert model.param_count() > 0


def test_bad_head_config_raises():
    with pytest.raises(ValueError):
        AttnConfig(d_model=64, n_heads=5)                          # 64 % 5 != 0


def test_variable_n_forward_is_finite():
    b = _batch(_views(((0, 0), (1, 6), (2, 12))))                  # mixed depths -> variable N + padding
    model = AttentionModel(AttnConfig()).eval()
    with torch.no_grad():
        q, attn = model(b["cards"], b["board"], b["phase"], b["action"], b["card_mask"])
    assert torch.isfinite(q).all() and torch.isfinite(attn).all()


def test_determinism():
    torch.manual_seed(0)
    model = AttentionModel(AttnConfig()).eval()
    b = _batch(_views())
    with torch.no_grad():
        q1, a1 = model(b["cards"], b["board"], b["phase"], b["action"], b["card_mask"])
        q2, a2 = model(b["cards"], b["board"], b["phase"], b["action"], b["card_mask"])
    assert torch.equal(q1, q2) and torch.equal(a1, a2)
