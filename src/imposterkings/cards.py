"""Card definitions and the 18-card deck registry.

A physical card is referenced everywhere by its **instance id** (an int 0..17), never by name or
value -- this preserves the identity of duplicates (the two Oathbounds, Soldiers, Elders,
Inquisitors stay distinguishable for MCTS node keying and for determinization).

``CARD_DEFS[instance_id]`` is the immutable :class:`CardDef` for that card. All value/ability/tag
queries derive from this registry; the JPG asset filename is art-only metadata for the UI and is
decoupled from the rules. Kings are NOT deck cards -- they are a per-seat one-time-life resource
(see ``GameState.kings``) rendered from ``97_King.jpg`` / ``98_TrueKing.jpg``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, FrozenSet, List, Optional, Tuple


class Ability(IntEnum):
    QUEEN = 1
    PRINCESS = 2
    KINGSHAND = 3
    SENTRY = 4
    WARLORD = 5
    MYSTIC = 6
    OATHBOUND = 7
    SOLDIER = 8
    JUDGE = 9
    INQUISITOR = 10
    ZEALOT = 11
    ELDER = 12
    ASSASSIN = 13
    FOOL = 14


class Tag(IntEnum):
    ROYALTY = 1     # Queen, Princess -- referenced by Warlord/Elder and royalty_present()
    REACTION = 2    # Assassin, King's Hand -- revealed from hand, not played onto the stack


@dataclass(frozen=True)
class CardDef:
    name: str
    value: int
    ability: Ability
    tags: FrozenSet[Tag]
    asset: str                      # primary jpg filename under assets/
    alt_asset: Optional[str] = None  # alternate-art reskin, if one exists


_ROYALTY = frozenset({Tag.ROYALTY})
_REACTION = frozenset({Tag.REACTION})
_NONE: FrozenSet[Tag] = frozenset()

# (CardDef template, count). Instance ids are assigned by flattening this in order.
DECK_SPEC: List[Tuple[CardDef, int]] = [
    (CardDef("Queen", 9, Ability.QUEEN, _ROYALTY, "09_Queen.jpg"), 1),
    (CardDef("Princess", 9, Ability.PRINCESS, _ROYALTY, "09_Princess.jpg"), 1),
    (CardDef("KingsHand", 8, Ability.KINGSHAND, _REACTION, "08_King_s-Hand_alt.jpg", "08_Spy.jpg"), 1),
    (CardDef("Sentry", 8, Ability.SENTRY, _NONE, "08_Sentry.jpg"), 1),
    (CardDef("Warlord", 7, Ability.WARLORD, _NONE, "07_Warlord.jpg", "07_Warden_alt.jpg"), 1),
    (CardDef("Mystic", 7, Ability.MYSTIC, _NONE, "07_Mystic.jpg"), 1),
    (CardDef("Oathbound", 6, Ability.OATHBOUND, _NONE, "06_oathbound_alt.jpg", "06_Herald.jpg"), 2),
    (CardDef("Soldier", 5, Ability.SOLDIER, _NONE, "05_Soldier.jpg"), 2),
    (CardDef("Judge", 5, Ability.JUDGE, _NONE, "05_Judge.jpg"), 1),
    (CardDef("Inquisitor", 4, Ability.INQUISITOR, _NONE, "04_Inquisitor_alt.jpg", "04_Executioner.jpg"), 2),
    (CardDef("Zealot", 3, Ability.ZEALOT, _NONE, "03_Zealot.jpg"), 1),
    (CardDef("Elder", 3, Ability.ELDER, _NONE, "03_Elder_alt.jpg"), 2),
    (CardDef("Assassin", 2, Ability.ASSASSIN, _REACTION, "02_Assassin.jpg"), 1),
    (CardDef("Fool", 1, Ability.FOOL, _NONE, "01_Fool.jpg"), 1),
]

# UI-only art assets that are not deck cards.
KING_ASSET = "97_King.jpg"
TRUE_KING_ASSET = "98_TrueKing.jpg"
CARD_BACK_ASSET = "99_back.jpg"

# Flatten the spec into the authoritative per-instance registry.
CARD_DEFS: Tuple[CardDef, ...] = tuple(
    cdef for cdef, count in DECK_SPEC for _ in range(count)
)
DECK_SIZE = len(CARD_DEFS)  # 18

# name -> sorted tuple of instance ids carrying that name (for guess resolution).
_IDS_BY_NAME: Dict[str, Tuple[int, ...]] = {}
for _cid, _cdef in enumerate(CARD_DEFS):
    _IDS_BY_NAME.setdefault(_cdef.name, ())
    _IDS_BY_NAME[_cdef.name] = _IDS_BY_NAME[_cdef.name] + (_cid,)

# Distinct card names, ordered by descending value then name (a stable guess menu).
CARD_NAMES: Tuple[str, ...] = tuple(
    name for name, _ in sorted(
        {cdef.name: cdef.value for cdef in CARD_DEFS}.items(),
        key=lambda kv: (-kv[1], kv[0]),
    )
)


# --- queries ---------------------------------------------------------------------------

def card_def(card: int) -> CardDef:
    return CARD_DEFS[card]


def card_value(card: int) -> int:
    return CARD_DEFS[card].value


def card_name(card: int) -> str:
    return CARD_DEFS[card].name


def card_ability(card: int) -> Ability:
    return CARD_DEFS[card].ability


def has_tag(card: int, tag: Tag) -> bool:
    return tag in CARD_DEFS[card].tags


def is_royalty(card: int) -> bool:
    return Tag.ROYALTY in CARD_DEFS[card].tags


def is_reaction(card: int) -> bool:
    return Tag.REACTION in CARD_DEFS[card].tags


def card_ids_for_name(name: str) -> Tuple[int, ...]:
    """All instance ids carrying ``name`` (e.g. ``"Oathbound" -> (6, 7)``); empty if unknown."""
    return _IDS_BY_NAME.get(name, ())


def asset_path(card: int, alt: bool = False) -> str:
    """Filename (under ``assets/``) for this card's art; falls back to primary if no alt exists."""
    cdef = CARD_DEFS[card]
    if alt and cdef.alt_asset is not None:
        return cdef.alt_asset
    return cdef.asset


def format_card(card: int) -> str:
    """Compact ``Name(value)#id`` label; the ``#id`` disambiguates duplicate instances."""
    cdef = CARD_DEFS[card]
    return f"{cdef.name}({cdef.value})#{card}"
