"""Godmode solver: exactness (against a plain brute-force minimax on small end-game positions -- which also
proves the transposition key never mis-folds two different positions), terminal handling, determinism, and
the unsolved-on-budget path. Engine-only, no torch.
"""
from __future__ import annotations

import numpy as np
import pytest

from imposterkings.godmode import Budget, solve, state_key
from imposterkings.state import GameState


def _play_until(seed, keep_cards):
    """Random-play a deal until few cards remain (a small end-game tree brute force can exhaust), or
    terminal. Deterministic per seed."""
    s = GameState.deal(np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 1)
    for _ in range(400):
        if s.is_terminal() or sum(len(h) for h in s.hands) <= keep_cards:
            break
        legal = s.legal_moves()
        s = s.apply(legal[rng.integers(len(legal))])
    return s


def _brute(s, cap=[0]):
    """Plain minimax in P0 perspective -- no TT, no pruning. The ground truth to check `solve` against."""
    if s.winner is not None:
        return 1.0 if s.winner == 0 else -1.0
    moves = s.legal_moves()
    if not moves:
        return 1.0 if (1 - s.to_play) == 0 else -1.0
    cap[0] += 1
    vals = [_brute(s.apply(m), cap) for m in moves]
    return max(vals) if s.to_play == 0 else min(vals)


def test_terminal_solve_is_exact():
    s = _play_until(3, keep_cards=0)          # random-play all the way to a terminal
    assert s.is_terminal()
    want = 1.0 if s.winner == 0 else -1.0
    r = solve(s, Budget())
    assert r.value == want and r.winner == s.winner


def test_matches_bruteforce_on_endgames():
    """`solve` (AND-OR + transposition table) must equal exhaustive minimax on small positions. If the TT key
    ever folded two DIFFERENT positions, this would diverge -- so it doubles as the key-safety test."""
    checked = 0
    for seed in range(40):
        s = _play_until(seed, keep_cards=3)
        if s.is_terminal():
            continue
        cap = [0]
        truth = _brute(s, cap)
        if cap[0] > 60000:                    # too big to brute-force in a unit test; skip
            continue
        r = solve(s, Budget())
        assert r.value == truth, f"seed {seed}: solve {r.value} != brute {truth}"
        checked += 1
    assert checked >= 5                       # actually exercised a handful of non-trivial endgames


def test_determinism():
    s = _play_until(7, keep_cards=4)
    a = solve(s, Budget()).value
    b = solve(s, Budget()).value
    assert a == b and a in (1.0, -1.0)


def test_best_move_is_legal_and_optimal():
    s = _play_until(9, keep_cards=4)
    if s.is_terminal():
        pytest.skip("terminal")
    r = solve(s, Budget(), root_cut=False)
    assert r.best_move in s.legal_moves()
    assert r.per_move[r.best_move] == r.value           # the chosen move achieves the solved value
    maximizing = s.to_play == 0
    assert r.value == (max(r.per_move.values()) if maximizing else min(r.per_move.values()))


def test_budget_exceeded_reports_unsolved():
    s = GameState.deal(np.random.default_rng(70_000_000))   # a heavy full-game root
    r = solve(s, Budget(max_nodes=500))
    assert r.value is None and r.best_move is None and r.budget.solved is False


def test_state_key_is_hashable_and_stable():
    s = _play_until(3, keep_cards=5)
    k = state_key(s)
    assert hash(k) == hash(state_key(s))                # deterministic, hashable
    # history is excluded: it is never populated, so it can't distinguish states here anyway
    assert s.history == ()


# --- Phase 3: agent, arena composability, recording -------------------------------------------------

def _decision_state():
    """A small non-terminal end-game with a real choice (>1 legal move) -- godmode solves it instantly."""
    for seed in range(60):
        s = _play_until(seed, keep_cards=3)
        if not s.is_terminal() and len(s.legal_moves()) > 1:
            return s
    raise RuntimeError("no end-game decision state found")


def test_godmode_agent_returns_a_legal_optimal_move():
    from imposterkings.godmode import GodmodeAgent
    s = _decision_state()
    agent = GodmodeAgent()
    mv = agent.select_move(s.information_set(s.to_play), np.random.default_rng(0), state=s)
    assert mv in s.legal_moves()
    lr = agent.last_result
    assert lr.eval_type == "godmode" and lr.solved_value in (1.0, -1.0)
    assert max(st.mean_q for st in lr.stats) == lr.solved_value    # best move achieves the solved value


def test_godmode_agent_needs_the_true_state():
    from imposterkings.godmode import GodmodeAgent
    s = _decision_state()
    with pytest.raises(ValueError):
        GodmodeAgent().select_move(s.information_set(s.to_play), np.random.default_rng(0))   # no state=


def test_arena_composes_mcts_seat_with_godmode_seat():
    from imposterkings.agents import RandomAgent
    from imposterkings.arena import play_game
    from imposterkings.godmode import GodmodeAgent
    agents = [RandomAgent(), GodmodeAgent(max_nodes=3000)]   # godmode falls back on unsolved early plies
    winner, rewards, term = play_game(agents, np.random.default_rng(11))
    assert winner in (0, 1) and term.is_terminal()          # the perfect_info branch passed it the true state


def test_record_captures_godmode_eval_and_roundtrips():
    import json
    from dataclasses import asdict
    from imposterkings.godmode import GodmodeAgent
    from imposterkings.record import DecisionRecord, GameRecord
    s = _decision_state()
    agent = GodmodeAgent()
    view = s.information_set(s.to_play)
    mv = agent.select_move(view, np.random.default_rng(0), state=s)
    dr = DecisionRecord.build(s.to_play, view, mv, agent)
    assert dr.eval_type == "godmode" and dr.solved_value in (1.0, -1.0)
    assert dr.candidates and all(c.solved is not None and c.win is not None for c in dr.candidates)
    rec = GameRecord(deal_seed=1, decisions=[dr])
    assert rec.schema_version == 2
    assert '"eval_type": "godmode"' in json.dumps(asdict(rec))    # serializes cleanly


def test_record_mcts_stays_mcts():
    from imposterkings.agents import MCTSAgent
    from imposterkings.record import DecisionRecord
    s = _decision_state()
    agent = MCTSAgent(iterations=20)
    view = s.information_set(s.to_play)
    mv = agent.select_move(view, np.random.default_rng(0))
    dr = DecisionRecord.build(s.to_play, view, mv, agent)
    assert dr.eval_type == "mcts" and dr.solved_value is None
    assert all(c.solved is None and c.win is None for c in dr.candidates)


# --- godmode-MCTS toggle (determinize-once perfect-info search) --------------------------------------

def test_ismcts_unchanged_when_godmode_off():
    """The toggle must not touch normal SO-ISMCTS: godmode off is view-only, deterministic per seed."""
    from imposterkings.agents import MCTSAgent
    s = _decision_state()
    view = s.information_set(s.to_play)
    a = MCTSAgent(iterations=80)
    assert a.perfect_info is False
    m1 = a.select_move(view, np.random.default_rng(0))
    m2 = a.select_move(view, np.random.default_rng(0))
    assert m1 == m2 and m1 in view.legal_moves()          # same seed -> same move (stream untouched)


def test_godmode_agent_searches_true_state():
    from imposterkings.agents import MCTSAgent
    s = _decision_state()
    g = MCTSAgent(iterations=80, godmode=True)
    assert g.perfect_info is True and g.name.endswith("+godmode")
    mv = g.select_move(s.information_set(s.to_play), np.random.default_rng(0), state=s)
    assert mv in s.legal_moves()
    # graceful fallback: godmode but no state given -> normal IS-MCTS, still legal
    mv2 = g.select_move(s.information_set(s.to_play), np.random.default_rng(0))
    assert mv2 in s.legal_moves()


def test_arena_composes_ismcts_and_godmode_seats():
    from imposterkings.agents import MCTSAgent
    from imposterkings.arena import play_game
    agents = [MCTSAgent(iterations=40, godmode=True), MCTSAgent(iterations=40, godmode=False)]
    winner, _, term = play_game(agents, np.random.default_rng(11))
    assert winner in (0, 1) and term.is_terminal()        # godmode seat got the true state via the arena branch


def test_godmode_mcts_mostly_matches_the_solver():
    """A strong godmode search should pick a solver-OPTIMAL move on end-games -- confirming it really is
    perfect-info-optimal, not IS-MCTS in disguise. Tolerate one stochastic miss."""
    from imposterkings.agents import MCTSAgent
    from imposterkings.godmode import Budget, solve
    checked = agreed = 0
    for seed in range(40):
        s = _play_until(seed, keep_cards=3)
        if s.is_terminal() or len(s.legal_moves()) < 2:
            continue
        r = solve(s, Budget(max_nodes=200_000), root_cut=False)
        if r.value is None:
            continue
        mover = s.to_play
        sign = 1.0 if mover == 0 else -1.0
        best = max(sign * v for v in r.per_move.values())
        mv = MCTSAgent(iterations=3000, godmode=True).select_move(
            s.information_set(mover), np.random.default_rng(seed), state=s)
        checked += 1
        agreed += int(sign * r.per_move[mv] == best)
        if checked >= 8:
            break
    assert checked >= 5 and agreed >= checked - 1
