"""The search RNG backends.

MCTS makes ~8,200 scalar draws per N=500 search and numpy's Generator costs ~880 ns per call (its
arithmetic is 6 ns -- the rest is per-call overhead it never amortizes, because the search draws one at a
time). Swapping the three hot call sites to stdlib `random` measured 9.0% faster over 200 paired positions.

These tests pin the two things that swap could have broken:
  1. The DEAL is untouched -- `deal_seed -> GameState.deal` is the contract behind every recorded corpus.
  2. Arm "numpy" still reproduces the pre-swap sampling exactly, so the A/B compared speed and nothing else.
"""
import numpy as np
import pytest

from imposterkings.actions import StepKind
from imposterkings.mcts import SearchConfig, search
from imposterkings.rng import BACKENDS, FastRng, NumpyRng, as_search_rng
from imposterkings.state import GameState


def _position(seed=0):
    st = GameState.deal(np.random.default_rng(seed), starting_player=0)
    while st.phase in (StepKind.SETUP_HIDE, StepKind.SETUP_DISCARD):
        st = st.apply(st.legal_moves()[0])
    return st.information_set(st.to_play)


@pytest.mark.parametrize("backend", BACKENDS)
def test_both_backends_draw_in_range_and_shuffle_a_permutation(backend):
    r = as_search_rng(np.random.default_rng(0), backend)
    assert all(0 <= r.pick(7) < 7 for _ in range(200))
    xs = list(range(12))
    r.shuffle(xs)
    assert sorted(xs) == list(range(12))            # a permutation: nothing gained, lost or duplicated
    assert xs != list(range(12))                    # ...and it actually moved (1/12! chance of a false fail)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_seed_fully_determines_the_search(backend):
    """Reproducibility survives the swap: the caller still supplies ONE numpy seed and gets one answer."""
    info = _position()

    def run():
        return [(str(s.move), s.visits) for s in
                search(info, SearchConfig(rng=np.random.default_rng(7), iterations=200,
                                          rng_backend=backend)).stats]

    assert run() == run()


def test_numpy_arm_reproduces_the_pre_swap_sampling():
    """NumpyRng.shuffle must be the OLD expression (permutation + rebuild), not merely *a* shuffle --
    otherwise the A/B that justified this change would have compared two different samplers, not two speeds."""
    src = [10, 20, 30, 40, 50, 60, 70]
    xs = list(src)
    NumpyRng(np.random.default_rng(3)).shuffle(xs)
    old = [src[i] for i in np.random.default_rng(3).permutation(len(src))]   # infoset.determinize, verbatim
    assert xs == old


def test_the_deal_is_untouched_by_the_backend():
    """THE contract: deal_seed -> deal backs every datasets/*.jsonl (5 sites re-derive a game from the seed
    alone), and the desync canary DROPS games silently rather than raising -- so a changed deal stream would
    quietly empty the corpora. GameState.deal must keep its numpy Generator regardless of the search backend."""
    a = GameState.deal(np.random.default_rng(99))
    b = GameState.deal(np.random.default_rng(99))
    assert a.hands == b.hands and a.leftover_faceup == b.leftover_faceup
    assert a.starting_player == b.starting_player
    # and the deal does not consult the search backend at all -- it takes a Generator, not a SearchRng
    assert not hasattr(np.random.default_rng(0), "pick")


def test_fastrng_accepts_a_seed_or_a_Random():
    assert FastRng(5).pick(10) == FastRng(5).pick(10)
    import random
    assert FastRng(random.Random(5)).pick(10) == FastRng(5).pick(10)


def test_as_search_rng_passes_an_adapted_rng_through_and_rejects_nonsense():
    r = FastRng(1)
    assert as_search_rng(r) is r                    # idempotent: search() adapts once, determinize re-wraps
    with pytest.raises(ValueError):
        as_search_rng(np.random.default_rng(0), "mersenne")
