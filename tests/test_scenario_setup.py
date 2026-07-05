"""The interactive scenario-setup screen: pure builder + allocation logic + a headless draw/click pass."""
from __future__ import annotations

import os

import pytest

pygame = pytest.importorskip("pygame")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from imposterkings import cards  # noqa: E402
from imposterkings.actions import StepKind  # noqa: E402
from imposterkings.scenario import cid  # noqa: E402
from imposterkings.ui import headless, scenario_setup as ss  # noqa: E402


def _click(ui, rects, pos):
    ss._route_click(ui, rects, pos)


def _center(rect):
    return (rect.x + rect.w // 2, rect.y + rect.h // 2)


def test_build_from_zones_builds_expected_state():
    zones = {"p0": [cid("Oathbound"), cid("Inquisitor")], "p1": [cid("Elder")],
             "stack": [cid("Sentry")], "p0_hidden": [], "p1_hidden": [cid("Fool")]}
    st = ss._build_from_zones(zones, [False, True], turn_player=0)
    assert st.to_play == 0 and st.phase == StepKind.MAIN
    assert cards.card_name(st.leading.card) == "Sentry" and st.leading_value() == 8
    assert set(st.hands[0]) == {cid("Oathbound"), cid("Inquisitor")}
    assert st.hidden[1] == cid("Fool") and st.kings == (False, True)


def test_palette_allocation_respects_instance_counts():
    screen, fonts = headless.session()
    ui = {"zones": ss._empty_zones(), "active": "p0", "turn_player": 0, "you_play": 0,
          "hotseat": False, "kings": [False, False], "used": set()}
    rects = ss._draw_setup(screen, fonts, ui, (0, 0))
    oath = next(r for r, name in rects["palette"] if name == "Oathbound")
    queen = next(r for r, name in rects["palette"] if name == "Queen")
    # Oathbound has TWO instances -> two clicks allocate ids 6 and 7; a third is refused.
    for _ in range(3):
        _click(ui, rects, _center(oath))
    assert ui["zones"]["p0"] == [6, 7]                       # both Oathbounds, no over-allocation
    # Queen has ONE instance; a second click is refused.
    _click(ui, rects, _center(queen))
    _click(ui, rects, _center(queen))
    assert ui["zones"]["p0"].count(cid("Queen")) == 1
    # clicking an assigned card removes it and frees the instance
    rects = ss._draw_setup(screen, fonts, ui, (0, 0))
    qcard = next(r for r, zk, c in rects["zone_cards"] if c == cid("Queen"))
    _click(ui, rects, _center(qcard))
    assert cid("Queen") not in ui["zones"]["p0"] and cid("Queen") not in ui["used"]


def test_hidden_zone_holds_one_and_pills_toggle():
    screen, fonts = headless.session()
    ui = {"zones": ss._empty_zones(), "active": "p1_hidden", "turn_player": 0, "you_play": 0,
          "hotseat": False, "kings": [False, False], "used": set()}
    rects = ss._draw_setup(screen, fonts, ui, (0, 0))
    fool = next(r for r, name in rects["palette"] if name == "Fool")
    _click(ui, rects, _center(fool))
    rects = ss._draw_setup(screen, fonts, ui, (0, 0))
    elder = next(r for r, name in rects["palette"] if name == "Elder")
    _click(ui, rects, _center(elder))                       # hidden holds ONE -> Fool replaced by Elder
    assert ui["zones"]["p1_hidden"] == [cid("Elder")]
    # pills mutate the toggles
    rects = ss._draw_setup(screen, fonts, ui, (0, 0))
    _click(ui, rects, _center(rects["pills"]["turn1"]))
    _click(ui, rects, _center(rects["pills"]["opp_hot"]))
    _click(ui, rects, _center(rects["pills"]["king0"]))
    assert ui["turn_player"] == 1 and ui["hotseat"] is True and ui["kings"][0] is True
