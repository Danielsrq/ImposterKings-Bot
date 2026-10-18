"""v3 (late-fusion) featurization: fixed layout, pointer correctness, the public-only ``picked`` rule and
no hidden-information leak. Real positions from random self-play walks."""
import dataclasses

import numpy as np
import pytest

from imposterkings.actions import ActionKind, StepKind
from imposterkings.cards import card_ability, card_name, Ability
from imposterkings.machine_learning import features2 as F2
from imposterkings.machine_learning import features3 as F3
from imposterkings.state import GameState


def _walk(n_games=40, seed=0):
    """Yield (state, view, moves) at every decision of random-move games."""
    rng = np.random.default_rng(seed)
    for g in range(n_games):
        st = GameState.deal(np.random.default_rng(1000 + g), starting_player=g % 2)
        while not st.is_terminal():
            moves = st.legal_moves()
            yield st, st.information_set(st.to_play), moves
            st = st.apply(moves[rng.integers(len(moves))])


@pytest.fixture(scope="module")
def positions():
    return list(_walk())


def _assert_same(a: F3.State3, b: F3.State3):
    np.testing.assert_array_equal(a.cards, b.cards)
    np.testing.assert_array_equal(a.card_depth, b.card_depth)
    np.testing.assert_array_equal(a.hands, b.hands)
    np.testing.assert_array_equal(a.kings, b.kings)
    np.testing.assert_array_equal(a.decision, b.decision)
    assert a.src_ptr == b.src_ptr


def test_fixed_layout_and_v2_agreement(positions):
    assert F3.CARD_DIM == len(F3.card_token_fields()) == 44
    assert F3.DECISION_DIM == len(F3.decision_fields()) == 40
    for st, v, moves in positions[::7]:
        s, c = F3.tokenize3(v, moves)
        assert s.cards.shape == (18, 44) and s.card_depth.shape == (18,)
        assert s.hands.shape == (2, 3) and s.kings.shape == (2, 2) and s.decision.shape == (40,)
        assert len(c.kind) == len(moves) <= F3.MAX_CANDS
        t2 = F2.tokenize_state(v, legal_moves=moves)
        np.testing.assert_array_equal(s.cards[:, F3._ZONE_OFF:], t2.cards[:, F2._ZONE_OFF:])
        np.testing.assert_array_equal(s.cards[:, :15], t2.cards[:, :15])          # name + power
        assert s.decision[:15].sum() == 1.0                                     # one step kind
        on = s.card_depth >= 0
        assert on.sum() == len(v.stack)
        if v.stack:                                                              # leading card = depth 0
            lead = s.card_depth == 0
            assert lead.sum() == 1
            assert s.cards[lead, F3._V2_KEEP.index(F2._STATE_OFF + 2)][0] == 1.0


def test_pointers_resolve_to_the_named_instance(positions):
    n_seen = {k: 0 for k in ActionKind}
    for st, v, moves in positions:
        s, c = F3.tokenize3(v, moves)
        for m, k, a, b, nm, nu in zip(moves, c.kind, c.ptr_a, c.ptr_b, c.name, c.number):
            n_seen[m.kind] += 1
            assert k == m.kind.value - 1
            if m.card is not None:
                assert F3.INSTANCE_NAMES[a] == card_name(m.card)
                assert s.slot_of[m.card] == a
            elif m.target is not None:
                assert s.slot_of[v.stack[m.target].card] == a
                assert s.card_depth[a] == min(len(v.stack) - 1 - m.target, F3.DEPTH_CAP)
            elif m.kind == ActionKind.FLIP_KING:
                assert a == F3.KING_MINE
            elif m.kind == ActionKind.REVEAL_KINGSHAND:
                assert F3.INSTANCE_NAMES[a] == "KingsHand" and s.cards[a, F3._ZONE_OFF] == 1.0  # my_hand
            elif m.kind == ActionKind.REVEAL_ASSASSIN:
                assert F3.INSTANCE_NAMES[a] == "Assassin" and s.cards[a, F3._ZONE_OFF] == 1.0
            elif m.kind in F3._STEP_ANSWERS:
                assert a == F3.DECISION and b == s.src_ptr
            if m.name is not None:
                assert F3._NAMES[nm] == m.name
            if m.number is not None:
                assert nu == m.number and F3.VALUE_MASK[nu].sum() >= 1
    assert n_seen[ActionKind.PLAY_CARD] and n_seen[ActionKind.GUESS_CARD] and n_seen[ActionKind.FLIP_KING]


def test_duplicate_instances_get_distinct_slots(positions):
    for st, v, moves in positions[::5]:
        s = F3.tokenize_state3(v, moves)
        assert len(set(s.slot_of.values())) == len(s.slot_of)
        for cid, slot in s.slot_of.items():
            assert F3.INSTANCE_NAMES[slot] == card_name(cid)


def test_picked_is_public_only(positions):
    seen_public = seen_princess = 0
    for st, v, moves in positions:
        step = v.pending[-1]
        s = F3.tokenize_state3(v, moves)
        flagged = np.flatnonzero(s.cards[:, F3._PICKED_COL])
        if step.kind == StepKind.ABILITY_SWAP_RESPOND:
            seen_princess += 1
            assert flagged.size == 0
            other = [c for c in range(18) if c != step.picked]
            for alt in other[:3]:                       # the hidden give-card must not move any token
                v2 = dataclasses.replace(v, pending=v.pending[:-1] + (dataclasses.replace(step, picked=alt),))
                _assert_same(s, F3.tokenize_state3(v2, moves))
        elif step.kind in F3._PICKED_PUBLIC and step.picked is not None:
            seen_public += 1
            assert card_ability(step.source) in (Ability.FOOL, Ability.SENTRY)
            assert list(flagged) == [s.slot_of[v.stack[step.picked].card]]
        else:
            assert flagged.size == 0
    assert seen_public and seen_princess


def test_no_hidden_info_across_determinizations(positions):
    """Tokens are a function of the decider's information set: any world consistent with it gives the same
    tokens (the hidden opponent hand / hidden card / muck never reach the encoding)."""
    for st, v, moves in positions[::11]:
        ref = F3.tokenize_state3(v, moves)
        for r in range(3):
            w = v.determinize(np.random.default_rng(r))
            _assert_same(ref, F3.tokenize_state3(w.information_set(v.observer), moves))


def test_candidates_are_the_legal_moves(positions):
    for st, v, moves in positions[::13]:
        assert sorted(map(str, v.legal_moves())) == sorted(map(str, moves))
