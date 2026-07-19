"""Godmode: an EXACT perfect-information solver over the true ``GameState``.

Under perfect information ImposterKings is deterministic, sequential, zero-sum, and has NO draws -- every
terminal is a strict win for one player (``rules.terminal_rewards`` -> +/-1). So the game value of a position
is a boolean "does the player to move force a win", and we solve it with an **AND-OR search** (the win/loss
specialisation of alpha-beta): at a node the mover controls, the node is a WIN as soon as ONE child is a win
(cut the siblings); at the opponent's node it is a LOSS as soon as one child is a loss. Every cut is a PROOF,
so every value -- and every transposition-table entry -- is exact; no bounds, no heuristics, no depth limit.

Values are carried in a FIXED player-0 perspective (+1 = P0 wins, -1 = P1 wins) rather than negamax, because
``apply`` frequently does NOT flip ``to_play`` (multi-step turns, the Oathbound second play, on-play
sub-decisions all keep the same mover) -- so alternation cannot be assumed.

This module is pure Python and torch-free. ``--probe`` measures how long solves take (wall + CPU) so we can
decide whether the pure search is fast enough or needs the escalation ladder (value-net move ordering ->
search engineering -> parallel -> Rust). See the plan / ``attention_exploration.md``.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .actions import Action
from .state import GameState

_TIME_CHECK_EVERY = 4096          # checking perf_counter() every node is itself costly; sample it


class BudgetExceeded(Exception):
    """Raised when a solve exceeds its node or wall-time budget -> the position is reported UNSOLVED."""


@dataclass
class Budget:
    """A solve's stop condition + live counters. ``None`` limits mean 'no limit' (solve to the end)."""
    max_nodes: Optional[int] = None
    time_limit: Optional[float] = None            # wall-clock seconds
    nodes: int = 0                                # interior nodes expanded (TT misses)
    tt_hits: int = 0
    max_depth: int = 0
    solved: bool = True
    wall: float = 0.0
    cpu: float = 0.0
    _deadline: Optional[float] = None
    _wall0: float = 0.0
    _cpu0: float = 0.0

    def start(self) -> None:
        self._wall0 = time.perf_counter()
        self._cpu0 = time.process_time()
        self._deadline = (self._wall0 + self.time_limit) if self.time_limit else None

    def stop(self) -> None:
        self.wall = time.perf_counter() - self._wall0
        self.cpu = time.process_time() - self._cpu0

    def tick(self, depth: int) -> None:
        self.nodes += 1
        if depth > self.max_depth:
            self.max_depth = depth
        if self.max_nodes is not None and self.nodes > self.max_nodes:
            raise BudgetExceeded
        if self._deadline is not None and (self.nodes & (_TIME_CHECK_EVERY - 1)) == 0:
            if time.perf_counter() > self._deadline:
                raise BudgetExceeded


def state_key(s: GameState) -> tuple:
    """Canonical, hashable transposition key. EXCLUDES ``history`` (never read by the engine; never even
    populated) and ``hand_lacks``/``hand_has`` (guess-leaked belief -- pure information that cannot gate
    legal_moves/resolve/win under perfect info, so dropping them folds more transpositions). INCLUDES
    ``pending`` -- the resolution chain defines the current micro-decision."""
    return (s.hands, s.hidden, s.kings, s.antechambers, s.stack, s.discard, s.setup_discard,
            s.leftover_faceup, s.leftover_facedown, s.muted_values, s.turn_player, s.starting_player,
            s.pending, s.winner)


Order = Callable[[GameState, List[Action], int], List[Action]]   # (state, moves, depth) -> ordered moves


def make_nn_order(evaluator, max_depth: int = 8) -> Order:
    """Value-net move ordering (an accelerant, NEVER a correctness input): at shallow depth, try each node's
    mover-preferred move first so the AND-OR cut fires sooner. A forward runs PER ordered node, so it only
    pays off shallow (few, high-fan-out nodes) -- deeper nodes keep natural order. ``evaluator`` is
    ``attention_model.evaluator_from_model`` -> ``(value, {move: prior})``; priors are the mover's softmax-Q,
    so sorting by descending prior tries the mover's best move first at BOTH players' nodes."""
    def order(s: GameState, moves: List[Action], depth: int) -> List[Action]:
        if depth > max_depth or len(moves) <= 1:
            return moves
        try:
            _v, priors = evaluator(s)
        except Exception:
            return moves
        return sorted(moves, key=lambda m: -priors.get(m, 0.0))
    return order


def _solve(s: GameState, tt: Dict[tuple, float], budget: Budget, depth: int, order: Optional[Order]) -> float:
    """Return the EXACT value of ``s`` in P0's perspective (+1 P0 wins / -1 P1 wins)."""
    if s.winner is not None:
        return 1.0 if s.winner == 0 else -1.0
    moves = s.legal_moves()
    if not moves:                                 # no legal move -> the player to move loses (engine rule)
        return 1.0 if (1 - s.to_play) == 0 else -1.0

    k = state_key(s)
    cached = tt.get(k)
    if cached is not None:
        budget.tt_hits += 1
        return cached

    budget.tick(depth)
    ordered = order(s, moves, depth) if order is not None else moves
    if s.to_play == 0:                            # OR node for P0: WIN if ANY child wins
        v = -1.0
        for m in ordered:
            if _solve(s.apply(m), tt, budget, depth + 1, order) == 1.0:
                v = 1.0
                break                             # decisive child -> proven WIN, cut the rest
    else:                                         # opponent's node: P0 LOSES if ANY child is a P0-loss
        v = 1.0
        for m in ordered:
            if _solve(s.apply(m), tt, budget, depth + 1, order) == -1.0:
                v = -1.0
                break
    tt[k] = v
    return v


@dataclass
class SolveResult:
    value: Optional[float]                        # +1 P0 / -1 P1 (mover-independent, P0 perspective); None=unsolved
    best_move: Optional[Action]                   # a move achieving the mover's optimum
    per_move: Dict[Action, float] = field(default_factory=dict)   # root move -> value (P0 perspective)
    budget: Optional[Budget] = None

    @property
    def winner(self) -> Optional[int]:
        return None if self.value is None else (0 if self.value > 0 else 1)


def solve(state: GameState, budget: Optional[Budget] = None, tt: Optional[Dict[tuple, float]] = None,
          order: Optional[Order] = None, root_cut: bool = True) -> SolveResult:
    """Solve ``state`` exactly. ``root_cut`` stops as soon as the root's optimal value is proven (fastest, for
    a win/loss verdict); set it False to evaluate EVERY root move (per-move values, e.g. the hide-Assassin
    analysis). A shared ``tt`` can be reused across related solves. Returns ``value=None`` if the budget is
    exceeded."""
    budget = budget or Budget()
    tt = tt if tt is not None else {}
    budget.start()
    try:
        if state.winner is not None:
            budget.stop()
            v = 1.0 if state.winner == 0 else -1.0
            return SolveResult(value=v, best_move=None, budget=budget)
        moves = state.legal_moves()
        if not moves:
            budget.stop()
            v = 1.0 if (1 - state.to_play) == 0 else -1.0
            return SolveResult(value=v, best_move=None, budget=budget)

        maximizing = (state.to_play == 0)
        target = 1.0 if maximizing else -1.0      # the mover's best possible outcome (P0 perspective)
        ordered = order(state, moves, 0) if order is not None else moves
        per_move: Dict[Action, float] = {}
        best_val = -math.inf if maximizing else math.inf
        best_move = moves[0]
        for m in ordered:
            cv = _solve(state.apply(m), tt, budget, 1, order)
            per_move[m] = cv
            if (maximizing and cv > best_val) or (not maximizing and cv < best_val):
                best_val, best_move = cv, m
            if root_cut and cv == target:         # mover found its best possible -> verdict proven
                break
        budget.stop()
        return SolveResult(value=best_val, best_move=best_move, per_move=per_move, budget=budget)
    except BudgetExceeded:
        budget.solved = False
        budget.stop()
        return SolveResult(value=None, best_move=None, budget=budget)


# --- agent (composable with MCTS) + a SearchResult-shaped result ----------------------------------

@dataclass
class GodmodeStat:
    """One root move's solved outcome, mirroring ``mcts.MoveStat`` so the recorder/review read it unchanged.
    ``mean_q`` is the value in the MOVER's perspective (+1 the mover wins), ``win`` the boolean."""
    move: Action
    mean_q: float
    visits: int = 1
    avail: int = 1
    visit_share: float = 0.0
    win: bool = False


@dataclass
class GodmodeResult:
    """A ``mcts.SearchResult`` look-alike carrying an EXACT solve. ``root=None`` (no tree); the review's PV /
    icicle panels already guard on that, and its eval graph reads only ``root_value()``."""
    info: object
    best_move: Optional[Action]
    stats: List[GodmodeStat]
    iterations: int                               # nodes expanded (the solve's 'work')
    elapsed: float
    solved_value: Optional[float]                 # mover-perspective solved value (None if unsolved)
    eval_type: str = "godmode"
    root: object = None
    determinizations: int = 0

    def root_value(self) -> float:
        return self.solved_value if self.solved_value is not None else 0.0

    def policy_target(self):
        winners = [s for s in self.stats if s.win]
        pick = winners or self.stats
        return {s.move: 1.0 / len(pick) for s in pick} if pick else {}


class GodmodeAgent:
    """Perfect-information solver AS AN AGENT -- composable per seat (``agents=[MCTSAgent(...), GodmodeAgent()]``).
    The ``perfect_info`` flag tells ``arena.play_game`` to hand it the TRUE ``GameState`` (see the branch there);
    every other agent stays on the view-only ``select_move(view, rng)`` seam. On an unsolved position (budget
    exceeded) it falls back to ``fallback`` (a view-only agent) or the first legal move, and records nothing."""

    perfect_info = True
    name = "godmode"

    def __init__(self, time_limit: Optional[float] = None, max_nodes: Optional[int] = None,
                 evaluate_forced: bool = False, order: Optional[Order] = None, fallback=None) -> None:
        self.time_limit = time_limit
        self.max_nodes = max_nodes
        self.evaluate_forced = evaluate_forced
        self.order = order                        # optional value-net move ordering (accelerant)
        self.fallback = fallback                  # view-only agent for unsolved positions (else first legal)
        self.last_result: Optional[GodmodeResult] = None

    def select_move(self, view, rng, state: Optional[GameState] = None) -> Action:
        moves = view.legal_moves()
        if len(moves) == 1 and not self.evaluate_forced:
            self.last_result = None
            return moves[0]
        if state is None:                         # perfect_info agent MUST get the true state (arena passes it)
            raise ValueError("GodmodeAgent.select_move needs the true GameState (perfect_info=True)")
        r = solve(state, Budget(time_limit=self.time_limit, max_nodes=self.max_nodes),
                  order=self.order, root_cut=False)     # root_cut=False -> per-move values for the record
        if r.value is None:                       # unsolved within budget -> fall back, record nothing
            self.last_result = None
            return (self.fallback.select_move(view, rng) if self.fallback is not None else moves[0])

        mover = state.to_play
        sign = 1.0 if mover == 0 else -1.0        # P0 perspective -> mover perspective
        stats = [GodmodeStat(move=m, mean_q=sign * v, win=(sign * v > 0))
                 for m, v in r.per_move.items()]
        stats.sort(key=lambda s: -s.mean_q)
        self.last_result = GodmodeResult(info=view, best_move=r.best_move, stats=stats,
                                         iterations=r.budget.nodes, elapsed=r.budget.wall,
                                         solved_value=sign * r.value)
        return r.best_move


# --- feasibility probe + profiling ----------------------------------------------------------------

def _probe(n_deals: int, base_seed: int, time_limit: Optional[float], node_limit: Optional[int],
           profile: bool) -> None:
    import numpy as np

    def q(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0.0

    print(f"godmode probe: {n_deals} full-game deal roots  (time_limit={time_limit}s, node_limit={node_limit})")
    rows = []
    for i in range(n_deals):
        seed = base_seed + i
        s = GameState.deal(np.random.default_rng(seed))
        r = solve(s, Budget(max_nodes=node_limit, time_limit=time_limit))
        b = r.budget
        nps = b.nodes / b.wall if b.wall > 0 else 0.0
        rows.append(dict(seed=seed, solved=b.solved, value=r.value, nodes=b.nodes, tt_hits=b.tt_hits,
                         wall=b.wall, cpu=b.cpu, nps=nps, depth=b.max_depth,
                         winner=(None if r.value is None else r.winner)))
        tag = ("P0" if r.value == 1.0 else "P1") if b.solved else "UNSOLVED"
        print(f"  seed {seed}: {tag:>8}  nodes {b.nodes:>10,}  wall {b.wall:7.2f}s  cpu {b.cpu:7.2f}s  "
              f"{nps:>9,.0f} n/s  tt_hits {b.tt_hits:>9,}  depth {b.max_depth}")

    solved = [r for r in rows if r["solved"]]
    n_solved = len(solved)
    print(f"\n{'=' * 74}")
    print(f"solved {n_solved}/{n_deals} within budget")
    if solved:
        walls = [r["wall"] for r in solved]
        cpus = [r["cpu"] for r in solved]
        nodes = [r["nodes"] for r in solved]
        p1 = sum(1 for r in solved if r["winner"] == r_second(r["seed"]))
        print(f"  WALL  s: median {q(walls,.5):7.2f}  p90 {q(walls,.9):7.2f}  max {max(walls):7.2f}")
        print(f"  CPU   s: median {q(cpus,.5):7.2f}  p90 {q(cpus,.9):7.2f}  max {max(cpus):7.2f}")
        print(f"  nodes  : median {q(nodes,.5):>12,.0f}  p90 {q(nodes,.9):>12,.0f}  max {max(nodes):>12,.0f}")
        print(f"  mean nodes/sec {sum(nodes)/sum(walls):,.0f}" if sum(walls) else "")
        print(f"  second-player wins: {p1}/{n_solved} = {p1/n_solved*100:.1f}%   (structural PI bias, tiny n)")

    if profile and rows:
        import cProfile
        import pstats
        rep = max(solved or rows, key=lambda r: r["nodes"])   # the heaviest solved deal
        print(f"\ncProfile of the heaviest solved deal (seed {rep['seed']}, {rep['nodes']:,} nodes):")
        s = GameState.deal(np.random.default_rng(rep["seed"]))
        pr = cProfile.Profile()
        pr.enable()
        solve(s, Budget(max_nodes=node_limit, time_limit=time_limit))
        pr.disable()
        pstats.Stats(pr).sort_stats("tottime").print_stats(12)


def r_second(seed: int) -> Optional[int]:
    """The SECOND player (non-starter) for a coin-flip deal, so the probe can tally second-player wins.
    ``GameState.deal`` picks the starter from the same rng; we re-derive it deterministically."""
    import numpy as np
    st = GameState.deal(np.random.default_rng(seed))
    return 1 - st.starting_player


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Godmode perfect-information solver -- feasibility probe.")
    p.add_argument("--probe", action="store_true", help="run the timing/feasibility probe")
    p.add_argument("--deals", type=int, default=30)
    p.add_argument("--base-seed", type=int, default=70_000_000)
    p.add_argument("--time-limit", type=float, default=60.0, help="per-solve wall-clock cap (s); 0 = none")
    p.add_argument("--node-limit", type=int, default=0, help="per-solve node cap; 0 = none")
    p.add_argument("--profile", action="store_true", help="cProfile the heaviest solve")
    args = p.parse_args(argv)
    if args.probe:
        _probe(args.deals, args.base_seed, args.time_limit or None, args.node_limit or None, args.profile)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
