"""Swappable game constants and the terminal-reward formula.

Centralizing the tunable rules here (as bigtwo does) keeps the search, generators, and state
machine free of magic numbers. This module imports nothing from the rest of the package so it
can never participate in an import cycle.
"""
from __future__ import annotations

from typing import List

NUM_PLAYERS = 2

# --- setup -----------------------------------------------------------------------------
DEAL_SIZE = 8          # cards dealt to each player
NUM_HIDDEN = 1         # cards each player sets aside as the hidden card
NUM_DISCARD = 1        # cards each player discards at setup
# -> each player keeps DEAL_SIZE - NUM_HIDDEN - NUM_DISCARD = 6 in hand.
HAND_AFTER_SETUP = DEAL_SIZE - NUM_HIDDEN - NUM_DISCARD

# --- ability tunables ------------------------------------------------------------------
SOLDIER_DISGRACE_CAP = 3      # Soldier may disgrace up to this many stack cards (incl. itself)
SOLDIER_BONUS = 2             # Soldier gains +2 on the stack on a correct guess
WARLORD_BONUS = 1             # Warlord +1 (does not stack) while royalty is present
MYSTIC_MIN = 1                # Mystic may pick a base value in [MYSTIC_MIN, MYSTIC_MAX]
MYSTIC_MAX = 8                # 9 (Queen/Princess) can never be targeted
MYSTIC_SET_VALUE = 3          # muted cards become this value

# --- rewards ---------------------------------------------------------------------------

def terminal_rewards(winner: int, cards_left: List[int], scaled: bool = True) -> List[float]:
    """Per-seat terminal reward vector. Winner +1, loser -1.

    ``scaled`` is accepted for parity with bigtwo (and a future "how bad was the loss" signal
    weighted by ``cards_left``), but the only confirmed win condition is binary, so both modes
    currently return +1 / -1. ``cards_left`` is threaded through for that future densification.
    """
    rewards = [-1.0] * NUM_PLAYERS
    rewards[winner] = 1.0
    return rewards
