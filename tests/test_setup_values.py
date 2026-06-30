"""Setup partitioning and the effective-value derivation (disgrace / mute / Warlord precedence)."""
from __future__ import annotations

import numpy as np

from imposterkings import cards, rules
from imposterkings.actions import StepKind
from imposterkings.state import GameState, StackCard

from .helpers import mainstate, cid


def test_deal_partitions_18_distinct():
    rng = np.random.default_rng(0)
    st = GameState.deal(rng, starting_player=0)
    seen = set(st.hands[0]) | set(st.hands[1]) | {st.leftover_faceup, st.leftover_facedown}
    assert len(seen) == cards.DECK_SIZE == 18
    assert len(st.hands[0]) == len(st.hands[1]) == 8


def test_setup_yields_six_hand_one_hidden_one_discard():
    rng = np.random.default_rng(1)
    st = GameState.deal(rng, starting_player=0)
    # Resolve the four setup steps by always taking the first legal choice.
    while st.phase in (StepKind.SETUP_HIDE, StepKind.SETUP_DISCARD):
        st = st.apply(st.legal_moves()[0])
    assert st.phase == StepKind.MAIN
    for seat in (0, 1):
        assert len(st.hands[seat]) == rules.HAND_AFTER_SETUP == 6
        assert st.hidden[seat] is not None
        assert st.setup_discard[seat] is not None
    assert st.to_play == 0  # starting player leads first


def test_disgraced_is_zero():
    st = mainstate()
    assert st.effective_stack_value(StackCard(cid("Queen"), disgraced=True)) == 0


def test_mute_sets_three():
    st = mainstate(muted={5})  # base value 5 muted
    assert st.effective_stack_value(StackCard(cid("Soldier"))) == 3


def test_mute_overrides_warlord_landing():
    # Warlord that landed at 9 (override) drops to 3 once value 7 is muted (precedence).
    st = mainstate(muted={7})
    assert st.effective_stack_value(StackCard(cid("Warlord"), value_override=9)) == 3


def test_warlord_hand_value_with_and_without_royalty():
    base = mainstate(hand0=(cid("Warlord"),))
    assert base.effective_hand_value(cid("Warlord")) == 7  # no royalty present
    with_royalty = mainstate(hand0=(cid("Warlord"),), stack=(StackCard(cid("Princess")),))
    assert with_royalty.effective_hand_value(cid("Warlord")) == 8  # +1, does not stack
    muted = mainstate(hand0=(cid("Warlord"),), stack=(StackCard(cid("Princess")),), muted={7})
    assert muted.effective_hand_value(cid("Warlord")) == 3  # muting cancels the buff
