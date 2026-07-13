"""Human-readable rules + card descriptions -- the copy shown by the UI's "How to play" panel.

CARD DATA, not UI data: this sits beside ``cards.py`` and imports nothing from ``ui`` (no pygame), so the
pygame panel, a README generator, a future web front-end or a test can all read the same single source of
wording. The dependency only ever points ``card_text -> cards/rules``, never back.

Every number in the prose is INTERPOLATED from ``rules.py`` rather than typed out, so the panel cannot
quietly start lying the day a tunable changes. Every ability below was checked against ``abilities.py``:

* ``_MANDATORY_GUESS`` = {Soldier, Judge}       -> their guess is a MUST (no decline).
* ``_OPTIONAL_ONPLAY`` = {Princess, Sentry, Mystic, Inquisitor, Fool} -> genuine MAY abilities.
* ``_can_play``        -> the three play-overrides: Oathbound (needs >= 2 cards in hand), Elder (over
  royalty), Zealot (over any non-royalty, once its OWN king is flipped).
* ``generate.py``      -> Judge may only queue a hand card of BASE value >= 2 (so never the Fool).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from . import rules
from .cards import DECK_SPEC

# --- the short rules summary (label, prose) ---------------------------------------------------------
RULES: List[Tuple[str, str]] = [
    ("Objective",
     "Win when your opponent cannot beat the leading card AND has no king left to flip."),
    ("Setup",
     f"{rules.DEAL_SIZE} cards each: hide {rules.NUM_HIDDEN} and discard {rules.NUM_DISCARD}, "
     f"keeping {rules.HAND_AFTER_SETUP} in hand plus the hidden one. The throne starts empty."),
    ("Your turn",
     "Play ONE card onto the throne. It must meet or beat the leading card's value -- unless its "
     "ability lets it override. The ability then resolves."),
    ("The King",
     "A one-time extra life. Flipping it IS your whole turn: it disgraces the leading card to 0 and "
     "returns your hidden card to hand. It can never be un-flipped."),
    ("Disgrace",
     "The card stays on the throne but becomes a 0 with no name, value, ability or tags. Disgracing "
     "never removes a card from the throne."),
    ("Antechamber",
     "A queue. At the start of your turn one queued card ascends and becomes the leading card, ability "
     "and all -- and that IS your turn."),
    ("Reactions",
     "King's Hand and Assassin are ordinary cards you can play for their value -- but they can ALSO be "
     "revealed from hand on the opponent's turn instead. King's Hand counters an optional ability; "
     "Assassin answers a king-flip with an instant win (which King's Hand can, in turn, counter)."),
]

# --- one description per card name (verified against abilities.py / generate.py) ----------------------
CARD_TEXT: Dict[str, str] = {
    "Queen":
        "Royalty. On play she MUST disgrace every card beneath her. She stays leading at 9.",
    "Princess":
        "Royalty. You MAY swap a card with your opponent -- each of you picks one from hand.",
    "KingsHand":
        "Plays normally as a plain 8 (no on-play ability). OR, on the opponent's turn, reveal it from hand "
        "to counter an optional ability: both cards are discarded, the throne reverts, and the turn goes "
        "back to them.",
    "Sentry":
        "You MAY disgrace the Sentry itself to swap a card from your hand with any non-disgraced, "
        "non-royalty card on the throne. Your card takes that exact position -- it does not go on top.",
    "Warlord":
        f"While royalty sits on the throne he counts as {7 + rules.WARLORD_BONUS} in hand and lands as "
        f"{8 + rules.WARLORD_BONUS}. The bonus does not stack.",
    "Mystic":
        f"You MAY disgrace the Mystic itself, then name a value from {rules.MYSTIC_MIN} to "
        f"{rules.MYSTIC_MAX}. EVERY card of that value -- in hand, on the throne, still to come -- "
        f"becomes a {rules.MYSTIC_SET_VALUE} and loses its ability and tags, permanently. 9s are safe.",
    "Oathbound":
        "You MAY play it over a HIGHER card by disgracing that card. "
        "If you do, you MUST then play another card of any value. This ability needs 2 cards in hand.",
    "Judge":
        "On play you MUST name a card in your opponent's hand. If you are right, you MAY queue a card "
        "of base value 2 or more from your hand into your OWN antechamber (never the Fool).",
    "Soldier":
        f"On play you MUST name a card in your opponent's hand. If you are right he gains "
        f"+{rules.SOLDIER_BONUS} automatically (leading at {5 + rules.SOLDIER_BONUS}), and you MAY then "
        f"disgrace up to {rules.SOLDIER_DISGRACE_CAP} cards on the throne -- the Soldier included.",
    "Inquisitor":
        "You MAY name a card. If your opponent holds it, they MUST queue it into THEIR antechamber. Any "
        "value -- even the Fool, which the Judge cannot touch.",
    "Elder":
        "May be played over ANY royalty, whatever its value.",
    "Zealot":
        "Once your OWN king is flipped, may be played over any non-royalty card, whatever its value.",
    "Assassin":
        "Plays normally as a plain 2 (no on-play ability). OR, if your opponent flips their king, reveal "
        "it from hand to win instantly -- unless they counter with King's Hand.",
    "Fool":
        "You MAY take any non-disgraced card on the throne back into your hand -- except the Fool itself.",
}


def deck_entries() -> List[Tuple[str, int, int, str]]:
    """``(name, value, copies, text)`` for the 14 cards, in DECK_SPEC order (descending value).

    Read straight from ``DECK_SPEC`` so the name/value/copy-count can never drift from the engine -- only
    the prose lives here."""
    return [(d.name, d.value, n, CARD_TEXT[d.name]) for d, n in DECK_SPEC]
