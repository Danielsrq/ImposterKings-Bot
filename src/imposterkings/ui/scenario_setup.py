"""Interactive scenario-setup screen: pick both hands, the stack (leading = rightmost), optional hidden
cards / king-flip / whose-turn, then play the constructed position in the app -- vs the Bot (like ui.app)
or in Hotseat (control both sides). Powered by ``scenario.build`` on the real engine.

``run_setup`` is a self-contained loop (like ``run_review``); ``_draw_setup`` does one frame + returns the
clickable rects (so it is headless-testable), and ``_build_from_zones`` is the pure state builder."""
from __future__ import annotations

from typing import Dict, List, Optional

from .. import cards, scenario
from . import assets
from .render import BTN, GOLD, INK, MUTE, PANEL, RED, _draw_card, _text

_PW, _PH = 58, 79          # palette card size
_ZW, _ZH = 58, 79          # zone card size
_PX0, _PY0 = 20, 56        # palette origin
_ZY0, _ZROW = 200, 96      # zone rows origin + pitch
_ZONES = [("p0", "P0 hand"), ("p1", "P1 hand"), ("stack", "Stack (lead →)"),
          ("p0_hidden", "P0 hidden"), ("p1_hidden", "P1 hidden")]
_HIDDEN = ("p0_hidden", "p1_hidden")


def _empty_zones() -> Dict[str, list]:
    return {"p0": [], "p1": [], "stack": [], "p0_hidden": [], "p1_hidden": []}


def _free_instances(name: str, used: set) -> List[int]:
    return [i for i in cards.card_ids_for_name(name) if i not in used]


def _build_from_zones(zones: Dict[str, list], kings, turn_player: int):
    """Pure: turn the zone lists (instance ids) into a ``GameState`` at ``MAIN`` for ``turn_player``."""
    hidden = (zones["p0_hidden"][0] if zones["p0_hidden"] else None,
              zones["p1_hidden"][0] if zones["p1_hidden"] else None)
    return scenario.build(hand0=zones["p0"], hand1=zones["p1"], stack=zones["stack"],
                          hidden=hidden, kings=tuple(kings), turn_player=turn_player)


def _draw_setup(screen, fonts, ui, mouse) -> dict:
    """Draw one setup frame from the ``ui`` state dict; return clickable rects for hit-testing."""
    import pygame
    small, med = fonts["small"], fonts["med"]
    screen.fill(PANEL)
    rects = {"palette": [], "zone_labels": {}, "zone_cards": [], "pills": {}, "buttons": {}}

    _text(screen, med, "Scenario Setup", (20, 12), INK)
    _text(screen, small, "Click a zone, then click cards to add them; click an added card to remove it.",
          (220, 20), MUTE)

    # --- palette: the 14 distinct card names, with a "xN left" badge (greyed when exhausted) ----------
    for i, name in enumerate(cards.CARD_NAMES):
        x = _PX0 + i * (_PW + 18)
        free = _free_instances(name, ui["used"])
        img = assets.card_surface(cards.card_ids_for_name(name)[0], (_PW, _PH))
        r = _draw_card(screen, img, (x, _PY0), dim=(not free), size=(_PW, _PH))
        _text(screen, small, f"x{len(free)}", (x + 4, _PY0 + _PH + 1), MUTE if free else RED)
        rects["palette"].append((r, name))

    # --- zone rows: a clickable label (active = GOLD) + the assigned cards (click to remove) ----------
    for row, (zk, label) in enumerate(_ZONES):
        y = _ZY0 + row * _ZROW
        active = ui["active"] == zk
        lr = pygame.Rect(20, y, 160, 26)
        pygame.draw.rect(screen, GOLD if active else BTN, lr, border_radius=4)
        _text(screen, small, label, (lr.x + 8, lr.y + 5), (20, 20, 20) if active else INK)
        rects["zone_labels"][zk] = lr
        cx = 200
        for cid_ in ui["zones"][zk]:
            cr = _draw_card(screen, assets.card_surface(cid_, (_ZW, _ZH)), (cx, y), size=(_ZW, _ZH))
            rects["zone_cards"].append((cr, zk, cid_))
            cx += _ZW + 8

    # --- pills: turn / you-play / opponent / king toggles --------------------------------------------
    def pill(key, label, x, y, on, w=64):
        r = pygame.Rect(x, y, w, 28)
        pygame.draw.rect(screen, GOLD if on else BTN, r, border_radius=14)
        _text(screen, small, label, (x + 10, y + 6), (20, 20, 20) if on else INK)
        rects["pills"][key] = r
        return x + w + 8

    ty = _ZY0 + len(_ZONES) * _ZROW + 8
    x = 20
    _text(screen, small, "Turn:", (x, ty + 6)); x += 56
    x = pill("turn0", "P0", x, ty, ui["turn_player"] == 0)
    x = pill("turn1", "P1", x, ty, ui["turn_player"] == 1) + 24
    _text(screen, small, "You play:", (x, ty + 6)); x += 90
    x = pill("you0", "P0", x, ty, ui["you_play"] == 0)
    x = pill("you1", "P1", x, ty, ui["you_play"] == 1) + 24
    _text(screen, small, "Opponent:", (x, ty + 6)); x += 96
    x = pill("opp_bot", "Bot", x, ty, not ui["hotseat"], w=70)
    x = pill("opp_hot", "Hotseat", x, ty, ui["hotseat"], w=90) + 24
    x = pill("king0", "P0 king ↓", x, ty, ui["kings"][0], w=100)
    x = pill("king1", "P1 king ↓", x, ty, ui["kings"][1], w=100)

    # --- action buttons + a live legal-move readout --------------------------------------------------
    by = ty + 46
    for key, label, bx, w in [("play", "Play", 20, 120), ("clear", "Clear", 152, 100),
                              ("cancel", "Cancel", 264, 100)]:
        r = pygame.Rect(bx, by, w, 32)
        hot = r.collidepoint(mouse)
        pygame.draw.rect(screen, GOLD if key == "play" else BTN, r, border_radius=4)
        if hot and key != "play":
            pygame.draw.rect(screen, GOLD, r, 2, border_radius=4)
        _text(screen, small, label, (bx + 14, by + 8), (20, 20, 20) if key == "play" else INK)
        rects["buttons"][key] = r
    try:
        st = _build_from_zones(ui["zones"], ui["kings"], ui["turn_player"])
        n = 0 if st.is_terminal() else len(st.legal_moves())
        _text(screen, small, f"P{ui['turn_player']} to move · {n} legal move(s)"
              + ("  — unplayable!" if n == 0 else ""), (400, by + 8), RED if n == 0 else MUTE)
    except Exception as exc:
        _text(screen, small, f"(invalid: {exc})", (400, by + 8), RED)
    return rects


def run_setup(screen, fonts, *, human_seat: int = 0) -> Optional[dict]:
    """Interactive board builder. Returns ``{"state","human_seat","hotseat"}`` on Play, ``None`` on
    Cancel/quit."""
    import pygame
    clock = pygame.time.Clock()
    ui = {"zones": _empty_zones(), "active": "p0", "turn_player": 0, "you_play": human_seat,
          "hotseat": False, "kings": [False, False], "used": set()}

    while True:
        mouse = pygame.mouse.get_pos()
        rects = _draw_setup(screen, fonts, ui, mouse)
        pygame.display.flip()
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                return None
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                pos = e.pos
                if _route_click(ui, rects, pos):          # a mutation happened
                    continue
                if rects["buttons"]["play"].collidepoint(pos):
                    st = _build_from_zones(ui["zones"], ui["kings"], ui["turn_player"])
                    return {"state": st, "human_seat": ui["you_play"], "hotseat": ui["hotseat"]}
                if rects["buttons"]["clear"].collidepoint(pos):
                    ui["zones"], ui["used"] = _empty_zones(), set()
                if rects["buttons"]["cancel"].collidepoint(pos):
                    return None
        clock.tick(30)


def _route_click(ui, rects, pos) -> bool:
    """Handle palette/zone/pill clicks; return True if the ui state was mutated (a card or toggle)."""
    for r, zk, cid_ in rects["zone_cards"]:              # remove an assigned card
        if r.collidepoint(pos):
            ui["zones"][zk].remove(cid_)
            ui["used"].discard(cid_)
            return True
    for zk, r in rects["zone_labels"].items():           # select the active zone
        if r.collidepoint(pos):
            ui["active"] = zk
            return True
    for r, name in rects["palette"]:                     # add the next free instance of a name
        if r.collidepoint(pos):
            free = _free_instances(name, ui["used"])
            if not free:
                return True
            zk = ui["active"]
            if zk in _HIDDEN and ui["zones"][zk]:         # hidden holds at most one
                ui["used"].discard(ui["zones"][zk][0])
                ui["zones"][zk] = []
            ui["zones"][zk].append(free[0])
            ui["used"].add(free[0])
            return True
    pills = rects["pills"]
    for key, r in pills.items():
        if not r.collidepoint(pos):
            continue
        if key == "turn0":
            ui["turn_player"] = 0
        elif key == "turn1":
            ui["turn_player"] = 1
        elif key == "you0":
            ui["you_play"] = 0
        elif key == "you1":
            ui["you_play"] = 1
        elif key == "opp_bot":
            ui["hotseat"] = False
        elif key == "opp_hot":
            ui["hotseat"] = True
        elif key == "king0":
            ui["kings"][0] = not ui["kings"][0]
        elif key == "king1":
            ui["kings"][1] = not ui["kings"][1]
        return True
    return False
