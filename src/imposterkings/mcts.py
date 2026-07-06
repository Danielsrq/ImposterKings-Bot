"""Single-Observer Information-Set MCTS (SO-ISMCTS), ported near-verbatim from bigtwo.

Reference: Cowling, Powley & Whitehouse, "Information Set Monte Carlo Tree Search" (2012).

One tree mixes the searcher's and the opponent's decision nodes. Because ImposterKings resolves a
turn as a sequence of micro-decisions (compound abilities) and reaction windows, consecutive nodes
may share a mover or flip movers -- both are handled with no special-casing, exactly as in bigtwo:
each iteration re-determinizes at the root and descends in lockstep with that concrete world, always
asking the concrete state for legal moves below the root (the InformationSet cannot enumerate the
opponent's). Near-perfect information makes the determinizations cheap and tightly clustered.

Per-node statistics: ``n`` visits, ``w`` reward credited to the seat that moved INTO the node,
``avail`` availability. Selection uses availability-based UCB1 so opponent nodes whose legality
varies per world are compared fairly. Invariant: ``avail >= n`` for every non-root node.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .actions import Action
from .infoset import InformationSet

DEFAULT_C = math.sqrt(2)


class Node:
    """A node in the ISMCTS tree. Holds statistics only; game state is carried separately."""
    __slots__ = ("parent", "incoming_move", "player_just_moved", "children", "n", "w", "avail", "priors")

    def __init__(self, parent: Optional["Node"], incoming_move: Optional[Action],
                 player_just_moved: Optional[int]) -> None:
        self.parent = parent
        self.incoming_move = incoming_move
        self.player_just_moved = player_just_moved  # seat that played incoming_move (None at root)
        self.children: Dict[Action, Node] = {}
        self.n = 0
        self.w = 0.0
        self.avail = 0
        self.priors: Optional[Dict[Action, float]] = None  # NN policy over this node's moves (PUCT only)

    def ucb(self, c: float) -> float:
        return self.w / self.n + c * math.sqrt(math.log(self.avail) / self.n)


@dataclass
class SearchConfig:
    rng: np.random.Generator
    iterations: int = 1000
    c: float = DEFAULT_C
    scaled: bool = True
    use_knowledge: bool = True     # honor guess-leaked opponent-hand facts when determinizing
    # AlphaZero mode: when an evaluator is attached, selection is PUCT (policy prior) and a non-terminal
    # leaf is valued by the net (no rollout). evaluator(state) -> (per-seat value vector, {move: prior}).
    evaluator: Optional["Callable"] = None
    c_puct: float = 1.5


@dataclass
class MoveStat:
    move: Action
    visits: int
    mean_q: float
    avail: int
    visit_share: float


@dataclass(frozen=True)
class PVStep:
    """One move along a principal variation (a most-visited path through the tree)."""
    move: Action
    player: int          # seat that played the move (Node.player_just_moved)
    visits: int
    mean_q: float        # value from that mover's perspective, in [-1, 1]


@dataclass
class SearchResult:
    info: InformationSet
    best_move: Action
    stats: List[MoveStat]            # sorted by visits, descending
    iterations: int
    elapsed: float                   # seconds
    determinizations: int = 0
    root: Optional["Node"] = field(default=None, repr=False)  # retained for PV extraction (in-memory)

    def policy_target(self) -> Dict[Action, float]:
        """Normalized visit counts -- the policy label the NN milestone will imitate."""
        total = sum(s.visits for s in self.stats)
        if total == 0:
            return {}
        return {s.move: s.visits / total for s in self.stats}

    def root_value(self) -> float:
        """Visit-weighted mean Q over the root's actions -- the search's value estimate of the
        position for the player to move (in [-1, 1]). Also the value target the NN milestone will use."""
        total = sum(s.visits for s in self.stats)
        if total == 0:
            return 0.0
        return sum(s.visits * s.mean_q for s in self.stats) / total

    def principal_variations(self, top: int = 2, depth: int = 6,
                             min_visits: int = 2) -> List[List[PVStep]]:
        """Top-``top`` lines: each starts at a most-visited root move and descends by most-visited
        child (chess-engine PV). Stops at ``depth``, at a leaf, or when a below-root node has
        ``visits < min_visits`` (ISMCTS deep-PV noise cutoff). Empty if the tree wasn't retained."""
        if self.root is None:
            return []
        lines: List[List[PVStep]] = []
        for stat in self.stats[:top]:
            node = self.root.children.get(stat.move)
            line: List[PVStep] = []
            while node is not None and len(line) < depth:
                if line and node.n < min_visits:
                    break  # below-root reliability cutoff
                line.append(PVStep(node.incoming_move, node.player_just_moved, node.n,
                                   node.w / node.n if node.n else 0.0))
                if not node.children:
                    break
                node = max(node.children.values(), key=lambda c: c.n)
            if line:
                lines.append(line)
        return lines


# --- the four phases -------------------------------------------------------------------

def _expand(node: Node, move: Action, mover: int) -> Node:
    child = Node(parent=node, incoming_move=move, player_just_moved=mover)
    node.children[move] = child
    return child


def _select(node: Node, state, config: SearchConfig
            ) -> Tuple[Node, object, List[Tuple[Node, frozenset]]]:
    """Descend (expanding once) in lockstep with the determinized ``state``."""
    visited: List[Tuple[Node, frozenset]] = []
    while not state.is_terminal():
        legal = state.legal_moves()             # from the concrete world, never info.legal_moves()
        if not legal:                           # non-terminal with no move: to_play loses -> score at leaf
            break                               # (only reachable at a determinized root sampled by others)
        legal_set = frozenset(legal)
        untried = [m for m in legal if m not in node.children]
        if untried:
            move = untried[int(config.rng.integers(len(untried)))]
            child = _expand(node, move, state.to_play)
            visited.append((node, legal_set))
            return child, state.apply(move), visited
        move = max(legal, key=lambda m: node.children[m].ucb(config.c))
        visited.append((node, legal_set))
        node = node.children[move]
        state = state.apply(move)
    return node, state, visited


def _select_puct(root: Node, state, config: SearchConfig):
    """Descend via PUCT (using stored NN priors) until an unexpanded/terminal leaf. No rollout.

    Returns (leaf, leaf_state). Re-checks legality each step (ISMCTS determinization). A node whose
    ``priors`` is None is an unevaluated leaf (the root on the first simulation); a selected move with no
    child yet is expanded and returned as the new leaf (the caller evaluates it and sets its priors)."""
    node = root
    while True:
        if state.is_terminal():
            return node, state
        legal = state.legal_moves()
        if not legal or node.priors is None:        # stuck (mover loses) or unevaluated leaf
            return node, state
        sqrt_n = math.sqrt(max(node.n, 1))
        default_p = 1.0 / len(legal)                # fallback for a determinization-exposed unseen move
        best_move, best_score = None, -math.inf
        for m in legal:
            child = node.children.get(m)
            p = node.priors.get(m, default_p)
            q = child.w / child.n if (child and child.n) else 0.0   # FPU 0 for unvisited
            n = child.n if child else 0
            score = q + config.c_puct * p * sqrt_n / (1 + n)
            if score > best_score:
                best_score, best_move = score, m
        child = node.children.get(best_move)
        if child is None:                           # expand -> the new node is the leaf
            child = _expand(node, best_move, state.to_play)
            return child, state.apply(best_move)
        node = child
        state = state.apply(best_move)


def _leaf_value(leaf: Node, state, config: SearchConfig) -> List[float]:
    """Value the leaf with the net (setting its priors), or exact result at terminal/stuck leaves."""
    if state.is_terminal():
        return state.result(scaled=config.scaled)
    if not state.legal_moves():                     # no move -> player to move loses
        return state.with_(winner=1 - state.to_play).result(scaled=config.scaled)
    value, priors = config.evaluator(state)
    leaf.priors = priors
    return value


def _rollout(state, config: SearchConfig) -> List[float]:
    """Uniform-random playout to a terminal state; return the per-seat reward vector."""
    while not state.is_terminal():
        legal = state.legal_moves()
        if not legal:                           # no legal move -> the player to move loses (game rule)
            return state.with_(winner=1 - state.to_play).result(scaled=config.scaled)
        state = state.apply(legal[int(config.rng.integers(len(legal)))])
    return state.result(scaled=config.scaled)


def _backpropagate(leaf: Node, reward: List[float],
                   visited: List[Tuple[Node, frozenset]]) -> None:
    node: Optional[Node] = leaf
    while node is not None:
        node.n += 1
        if node.player_just_moved is not None:
            node.w += reward[node.player_just_moved]
        node = node.parent
    for parent, legal_set in visited:
        for move, child in parent.children.items():
            if move in legal_set:
                child.avail += 1


# --- driver ----------------------------------------------------------------------------

def search(info: InformationSet, config: SearchConfig) -> SearchResult:
    """Run SO-ISMCTS from ``info`` and return per-root-move statistics."""
    start = time.perf_counter()
    root = Node(parent=None, incoming_move=None, player_just_moved=None)

    use_nn = config.evaluator is not None
    for _ in range(config.iterations):
        state = info.determinize(config.rng, use_knowledge=config.use_knowledge)  # fresh root determinization
        if use_nn:                                   # AlphaZero: PUCT + net value, no rollout
            leaf, leaf_state = _select_puct(root, state, config)
            _backpropagate(leaf, _leaf_value(leaf, leaf_state, config), [])
        else:                                        # classic: UCB + random rollout
            leaf, leaf_state, visited = _select(root, state, config)
            _backpropagate(leaf, _rollout(leaf_state, config), visited)

    total_visits = sum(child.n for child in root.children.values())
    stats = [
        MoveStat(move=move, visits=child.n,
                 mean_q=child.w / child.n if child.n else 0.0,
                 avail=child.avail,
                 visit_share=child.n / total_visits if total_visits else 0.0)
        for move, child in root.children.items()
    ]
    stats.sort(key=lambda s: s.visits, reverse=True)

    best_move = max(root.children,
                    key=lambda m: (root.children[m].n, root.children[m].w / root.children[m].n))

    return SearchResult(info=info, best_move=best_move, stats=stats,
                        iterations=config.iterations, elapsed=time.perf_counter() - start,
                        determinizations=config.iterations, root=root)
