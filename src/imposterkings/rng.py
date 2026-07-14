"""The two random draws MCTS actually makes, behind one interface -- so the backend is a switch, not a fork.

The search needs exactly two primitives: pick a uniform index, and shuffle a small list. numpy's Generator
serves both, but pays ~880 ns of Python/dispatch overhead PER CALL (its arithmetic is only ~6 ns/draw when
batched -- the loss is entirely per-call overhead, and MCTS draws one at a time, so it never amortizes).
The stdlib `random` module is 3.7-18x faster on those scalar draws.

Measured share of an N=500 rollout search: RNG is <= 6.3% of the time (~8,200 index picks + 500 shuffles),
so the honest ceiling is a few percent -- the engine, not the RNG, is where search time goes. Hence the
switch: `SearchConfig(rng_backend=...)` lets the A/B be run rather than argued about.

WHAT MUST NOT MOVE: `GameState.deal` keeps its numpy Generator. `deal_seed -> deal` is the reproducibility
contract behind every recorded corpus (five call sites re-derive a game from its seed alone), and the
desync canary in token_dataset DROPS games silently rather than raising -- so a changed deal stream would
quietly empty the datasets instead of failing loudly. The deal happens once per game; there is nothing to
win there and everything to lose. This module is for the SEARCH stream only, which nothing replays.
"""
from __future__ import annotations

import random
from typing import List, Protocol, Union

import numpy as np

BACKENDS = ("numpy", "stdlib")


class SearchRng(Protocol):
    """The whole surface the search needs."""

    def pick(self, n: int) -> int:
        """A uniform index in [0, n)."""

    def shuffle(self, xs: List) -> None:
        """Shuffle ``xs`` IN PLACE."""


class NumpyRng:
    """Arm A -- the status quo. Wraps a numpy Generator, preserving its exact stream."""

    __slots__ = ("g",)

    def __init__(self, g: np.random.Generator):
        self.g = g

    def pick(self, n: int) -> int:
        return int(self.g.integers(n))

    def shuffle(self, xs: List) -> None:
        # The historical expression from infoset.determinize: permutation + rebuild. Kept verbatim so arm A
        # is bit-for-bit the old behaviour (the A/B compares speed, and must not also change the sampling).
        xs[:] = [xs[i] for i in self.g.permutation(len(xs))]


class FastRng:
    """Arm B -- stdlib `random`. Same two primitives, no ndarray boxing on the way out."""

    __slots__ = ("r",)

    def __init__(self, seed: Union[int, random.Random, None] = None):
        self.r = seed if isinstance(seed, random.Random) else random.Random(seed)

    def pick(self, n: int) -> int:
        return self.r.randrange(n)

    def shuffle(self, xs: List) -> None:
        self.r.shuffle(xs)


def as_search_rng(rng, backend: str = "numpy") -> SearchRng:
    """Adapt whatever the caller passed into a SearchRng.

    Already adapted -> returned as-is. A raw numpy Generator (what agents/arena/tests still hand us) ->
    wrapped per ``backend``. For "stdlib" the Random is seeded FROM the Generator, so a run stays fully
    determined by the caller's numpy seed -- the same seed reproduces the same search under either backend's
    own stream, and no caller has to learn about a second RNG.
    """
    if hasattr(rng, "pick"):                       # already a SearchRng (or a test double)
        return rng
    if backend == "stdlib":
        return FastRng(int(rng.integers(2**63 - 1)))
    if backend == "numpy":
        return NumpyRng(rng)
    raise ValueError(f"unknown rng_backend {backend!r} (expected one of {BACKENDS})")
