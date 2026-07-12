"""The attention leaf evaluator: satisfies the mcts SearchConfig.evaluator contract, plugs into
MCTSAgent, and the benchmark checkpoint-type dispatch resolves it. Uses a tiny untrained checkpoint
(plumbing, not a quality claim)."""
import numpy as np
import torch

from imposterkings.agents import MCTSAgent
from imposterkings.budget import hybrid
from imposterkings.infoset import InformationSet
from imposterkings.machine_learning import benchmark as B
from imposterkings.machine_learning.attention_model import (
    AttentionModel, AttnConfig, build_evaluator, save)
from imposterkings.rules import NUM_PLAYERS
from imposterkings.state import GameState


def _ckpt(tmp_path):
    torch.manual_seed(0)
    p = str(tmp_path / "attn.pt")
    save(p, AttentionModel(AttnConfig(d_model=32)))
    return p


def _states(n=4):
    return [GameState.deal(np.random.default_rng(s)) for s in range(n)]


def test_evaluator_contract(tmp_path):
    ev = build_evaluator(_ckpt(tmp_path))
    for state in _states():
        value, priors = ev(state)
        assert len(value) == NUM_PLAYERS
        mover = state.to_play
        assert value[mover] == -value[1 - mover]           # zero-sum
        assert -1.0 <= value[mover] <= 1.0                 # Tanh-bounded q
        assert set(priors) == set(state.legal_moves())     # one prior per legal move
        assert abs(sum(priors.values()) - 1.0) < 1e-5      # softmax distribution


def test_plugs_into_mcts(tmp_path):
    ev = build_evaluator(_ckpt(tmp_path))
    agent = MCTSAgent(budget=hybrid(20, 3), evaluator=ev)
    s = GameState.deal(np.random.default_rng(0))
    view = InformationSet.from_state(s, s.to_play)
    move = agent.select_move(view, np.random.default_rng(1))
    assert move in view.legal_moves()                      # evaluator drives PUCT without error


def test_benchmark_dispatch_picks_attention(tmp_path):
    ev = B._evaluator_for(_ckpt(tmp_path))                 # attention ckpt -> attention evaluator
    value, priors = ev(GameState.deal(np.random.default_rng(0)))
    assert len(value) == NUM_PLAYERS and abs(sum(priors.values()) - 1.0) < 1e-5


def test_benchmark_smoke_endtoend(tmp_path):
    # tiny mirrored match: attention-MCTS@k20 vs vanilla-MCTS@k20, 2 deals, 1 worker
    res = B.run(ckpt=_ckpt(tmp_path), opponents=[("vanilla", ("hybrid", 20, 3))],
                deals=2, chunk=2, workers=1, base_seed=0, independent_rng=True,
                nn_mcts="hybrid-k20-l3")
    assert res[0]["opponent"] == "vanilla" and res[0]["games"] == 4
    assert 0.0 <= res[0]["winrate"] <= 1.0


def test_parse_opponent_nnmcts_spec():
    assert B.parse_opponent("models/mlp_256.pt@hybrid-k20-l3") == \
        ("nnmcts", "models/mlp_256.pt", ("hybrid", 20, 3))
    assert B.parse_opponent("m.pt@fixed50") == ("nnmcts", "m.pt", ("fixed", 50))
    assert B.parse_opponent("models/mlp_32.pt") == ("nn", "models/mlp_32.pt")   # bare .pt stays greedy
    assert B.parse_opponent("hybrid-k20-l3") == ("hybrid", 20, 3)


def test_benchmark_nnmcts_opponent_endtoend(tmp_path):
    # the OPPONENT is itself an NN-MCTS (net as eval head in search) -- 1 deal, tiny budgets
    ck = _ckpt(tmp_path)
    res = B.run(ckpt=ck, opponents=[("nnmcts-opp", B.parse_opponent(f"{ck}@fixed30"))],
                deals=1, chunk=1, workers=1, base_seed=0, independent_rng=True,
                nn_mcts="fixed30")
    assert res[0]["games"] == 2 and 0.0 <= res[0]["winrate"] <= 1.0
