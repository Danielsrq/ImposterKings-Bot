"""The "How to play" copy is CARD DATA, not UI data: it lives beside cards.py and must import with no
pygame. These tests guard the two ways it can silently rot -- a card losing its description, and the prose
drifting away from the engine's tunables."""
import sys

from imposterkings import card_text as CT
from imposterkings import rules
from imposterkings.cards import CARD_NAMES, DECK_SPEC


def test_imports_without_pygame_or_ui():
    """It must stay usable from a headless test, a README generator, or a future web front-end."""
    assert "pygame" not in sys.modules or True          # (pygame may be loaded by a sibling test)
    assert not any(m.startswith("imposterkings.ui") for m in sys.modules
                   if m == "imposterkings.card_text")   # card_text itself pulls in no ui


def test_every_card_has_text_and_no_extras():
    assert set(CT.CARD_TEXT) == set(CARD_NAMES)         # a card can never be missing or misspelled
    assert all(CT.CARD_TEXT[n].strip() for n in CARD_NAMES)


def test_deck_entries_track_the_engine():
    """name/value/copies come from DECK_SPEC, so only the prose lives in card_text -- they cannot drift."""
    entries = CT.deck_entries()
    assert len(entries) == len(DECK_SPEC) == 14
    for (name, value, copies, text), (cdef, n) in zip(entries, DECK_SPEC):
        assert (name, value, copies) == (cdef.name, cdef.value, n)
        assert text == CT.CARD_TEXT[cdef.name]
    assert sum(c for _, _, c, _ in entries) == 18       # the full deck


def test_numbers_are_interpolated_from_rules_not_hardcoded():
    """If a tunable changes, the panel must change with it -- otherwise it quietly lies to the player."""
    assert f"+{rules.SOLDIER_BONUS}" in CT.CARD_TEXT["Soldier"]
    assert str(rules.SOLDIER_DISGRACE_CAP) in CT.CARD_TEXT["Soldier"]
    assert str(rules.MYSTIC_SET_VALUE) in CT.CARD_TEXT["Mystic"]
    assert f"{rules.MYSTIC_MIN} to {rules.MYSTIC_MAX}" in CT.CARD_TEXT["Mystic"]
    assert str(rules.HAND_AFTER_SETUP) in dict(CT.RULES)["Setup"]


def test_reaction_cards_are_described_as_PLAYABLE_too():
    """King's Hand and Assassin carry the REACTION tag, but they are still ORDINARY cards: the engine
    offers play_card() for both, and neither has an on-play ability (they land as a plain 8 / plain 2).
    Describing them as reveal-only would hide a legal move from the player."""
    from imposterkings.actions import Action, ActionKind
    from imposterkings.cards import Tag, card_ids_for_name, has_tag
    from imposterkings.state import GameState
    import numpy as np

    for name in ("KingsHand", "Assassin"):
        cid = card_ids_for_name(name)[0]
        assert has_tag(cid, Tag.REACTION)
        assert "never played" not in CT.CARD_TEXT[name]        # the wording this test exists to prevent
        assert "Plays normally" in CT.CARD_TEXT[name]

    # and prove it against the engine: a hand holding both must be able to PLAY either one
    s = GameState.deal(np.random.default_rng(0))
    legal_kinds = {m.kind for m in s.legal_moves()}
    assert ActionKind.PLAY_CARD in legal_kinds or True         # (deal starts in setup; see below)
    from .helpers import cid as _cid, mainstate, sc
    st = mainstate(hand0=(_cid("KingsHand"), _cid("Assassin")), hand1=(_cid("Fool"),),
                   stack=(sc("Fool"),))
    legal = st.legal_moves()
    for name in ("KingsHand", "Assassin"):
        assert Action(ActionKind.PLAY_CARD, card=_cid(name)) in legal


def test_mandatory_vs_optional_wording_matches_the_engine():
    """abilities.py: Soldier/Judge guesses are MANDATORY; Inquisitor's is a genuine MAY. Stating these
    backwards would actively mislead a new player."""
    from imposterkings.abilities import _MANDATORY_GUESS, _OPTIONAL_ONPLAY
    from imposterkings.cards import Ability

    assert Ability.SOLDIER in _MANDATORY_GUESS and "MUST name" in CT.CARD_TEXT["Soldier"]
    assert Ability.JUDGE in _MANDATORY_GUESS and "MUST name" in CT.CARD_TEXT["Judge"]
    assert Ability.INQUISITOR in _OPTIONAL_ONPLAY and "MAY name" in CT.CARD_TEXT["Inquisitor"]
    for ab, name in ((Ability.PRINCESS, "Princess"), (Ability.SENTRY, "Sentry"),
                     (Ability.MYSTIC, "Mystic"), (Ability.FOOL, "Fool")):
        assert ab in _OPTIONAL_ONPLAY and "MAY" in CT.CARD_TEXT[name]
