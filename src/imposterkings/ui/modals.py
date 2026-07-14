"""Modal overlays drawn on top of the board: the right-click card zoom and the engine-settings panel.

Both follow the same convention -- dim the screen with a scrim, draw a centered PANEL box with a GOLD
border, and return the rects the app loop must hit-test. (The "How to play" modal is big enough to own its
own module; the attention drawer lives with the rest of the attention code in ``attention_view``.)
"""
from __future__ import annotations

from pathlib import Path

import pygame

from . import assets, widgets
from .theme import BTN, BTN_HOVER, BG, GOLD, INK, MUTE, PANEL, WINDOW


_PREVIEW_H = 900                                     # art is natively 700x955 -> 900px tall stays UNDER
_PREVIEW = (int(_PREVIEW_H * 700 / 955), _PREVIEW_H)  # native res (no upscaling blur), aspect preserved


def draw_card_preview(surface, fonts, asset: str, flipped: bool = False) -> None:
    """Right-click zoom: the card's art, near-native size, centered over a dim scrim. ``asset`` is an
    ``assets/`` filename (``render_frame`` hands them out in ``Frame.previews``)."""
    W, H = WINDOW
    scrim = pygame.Surface(WINDOW, pygame.SRCALPHA)
    scrim.fill((0, 0, 0, 190))
    surface.blit(scrim, (0, 0))
    img = assets.image(asset, _PREVIEW)
    if flipped:
        img = pygame.transform.rotate(img, 180)
    x, y = (W - _PREVIEW[0]) // 2, (H - _PREVIEW[1]) // 2
    surface.blit(img, (x, y))
    pygame.draw.rect(surface, GOLD, (x, y, *_PREVIEW), 2)
    hint = "click anywhere / Esc to close"
    widgets.text(surface, fonts["small"], hint,
          ((W - fonts["small"].size(hint)[0]) // 2, y + _PREVIEW[1] + 8), MUTE)



ENGINE_PILLS = [("mcts", "Fixed N"), ("branching", "Branch"), ("hybrid", "Hybrid"), ("nn", "NN+MCTS")]


_SLIDER_RANGES = {"N": (25, 1024), "k": (10, 100), "l": (1, 8)}


def _ckpt_label(path: str, others=None) -> str:
    """A checkpoint's DISPLAY NAME: just the file's own name, e.g. ``attn_d64_L2``.

    Checkpoint paths are absolute (they resolve against the bundle root, not the cwd), so showing the path
    overflows the box -- and the directory is noise anyway. Show the bare name; only when two discovered
    checkpoints share a name (``sweep_v3a/attn_d64_L2`` vs ``gen1_v3c_v2feat/attn_d64_L2``, which happens a
    lot in a dev tree) fall back to ``parent/name`` so the arrows still distinguish them."""
    p = Path(path)
    name = p.stem
    if others:
        clash = sum(1 for o in others if Path(o).stem == name) > 1
        if clash and p.parent.name:
            return f"{p.parent.name}/{name}"
    return name


def draw_settings_overlay(surface, fonts, engine, mouse, nn_available=True,
                          nn_ckpts=None, nn_ckpt_ix=0):
    """Draw the engine-settings modal over the board and return its clickable controls:
    ``{"pills": {mode: rect}, "sliders": [...], "close": rect, "ckpt_prev": rect|None,
    "ckpt_next": rect|None}``.

    ``engine`` = ``{"mode", "N", "k", "l"}``. Fixed mode shows one ``N`` slider; branch/hybrid/nn show ``k``
    and ``l`` (l = effective legal-moves for a sub-decision card at selection). ``nn`` (NN+MCTS) is
    hybrid-only; when ``nn_available`` is False that pill is drawn disabled and its click is ignored.
    In ``nn`` mode a checkpoint cycler picks WHICH net drives the search -- ``nn_ckpts`` (paths) selected
    by ``nn_ckpt_ix``; both MLP and attention checkpoints are offered."""
    med, small = fonts["med"], fonts["small"]
    W, H = WINDOW
    dim = pygame.Surface((W, H), pygame.SRCALPHA)
    dim.fill((0, 0, 0, 160))
    surface.blit(dim, (0, 0))
    is_fixed = engine["mode"] == "mcts"
    show_ckpt = engine["mode"] == "nn" and bool(nn_ckpts)
    bw, bh = 680, (250 if is_fixed else (376 if show_ckpt else 320))
    bx, by = (W - bw) // 2, (H - bh) // 2
    pygame.draw.rect(surface, PANEL, (bx, by, bw, bh), border_radius=8)
    pygame.draw.rect(surface, GOLD, (bx, by, bw, bh), 2, border_radius=8)
    widgets.text(surface, med, "Engine settings  (bot + hint)", (bx + 20, by + 16), INK)

    n = len(ENGINE_PILLS)
    pills, pw = {}, (bw - 40 - 8 * (n - 1)) // n
    x = bx + 20
    for mode, label in ENGINE_PILLS:
        r = pygame.Rect(x, by + 56, pw, 36)
        disabled = (mode == "nn") and not nn_available
        sel = engine["mode"] == mode
        fill = MUTE if disabled else (GOLD if sel else BTN)
        pygame.draw.rect(surface, fill, r, border_radius=18)
        tw = small.size(label)[0]
        widgets.text(surface, small, label, (r.centerx - tw // 2, r.y + 9), (20, 20, 20) if sel else INK)
        pills[mode] = r
        x += pw + 8

    def slider_row(sy, key):
        lo, hi = _SLIDER_RANGES[key]
        val = engine[key]
        widgets.text(surface, small, f"{key} = {val}", (bx + 20, sy - 24), GOLD)
        rng = f"[{lo} .. {hi}]"
        widgets.text(surface, small, rng, (bx + bw - 20 - small.size(rng)[0], sy - 24), MUTE)
        track = pygame.Rect(bx + 20, sy, bw - 40, 8)
        pygame.draw.rect(surface, BTN, track, border_radius=4)
        kx = int(track.x + (val - lo) / (hi - lo) * track.w)
        pygame.draw.circle(surface, GOLD, (kx, track.centery), 10)
        return (track, lo, hi, key)

    ckpt_prev = ckpt_next = None
    if is_fixed:
        sliders = [slider_row(by + 140, "N")]
        preview = f"~ {engine['N']} simulations / decision"
    else:
        sliders = [slider_row(by + 130, "k"), slider_row(by + 196, "l")]
        if engine["mode"] == "nn":
            preview = "NN eval-head + hybrid clamp(k * eff_n(l) * (1 + opp_cards), 64, 4096)"
        elif engine["mode"] == "hybrid":
            preview = "clamp(k * eff_n(l) * (1 + opp_cards), 64, 4096)"
        else:
            preview = "clamp(k * eff_n(l), 64, 4096)"
    if show_ckpt:                                     # WHICH net drives NN+MCTS (MLP or attention)
        cy = by + 246
        widgets.text(surface, small, "NN checkpoint:", (bx + 20, cy - 22), GOLD)
        ckpt_prev = pygame.Rect(bx + 20, cy, 30, 28)
        ckpt_next = pygame.Rect(bx + bw - 20 - 30, cy, 30, 28)
        for r, glyph in ((ckpt_prev, "<"), (ckpt_next, ">")):
            pygame.draw.rect(surface, BTN_HOVER if r.collidepoint(mouse) else BTN, r, border_radius=4)
            widgets.text(surface, small, glyph, (r.centerx - small.size(glyph)[0] // 2, r.y + 5), INK)
        ix = nn_ckpt_ix % len(nn_ckpts)
        name = _ckpt_label(nn_ckpts[ix], nn_ckpts)
        box = pygame.Rect(ckpt_prev.right + 8, cy, ckpt_next.x - ckpt_prev.right - 16, 28)
        pygame.draw.rect(surface, BG, box, border_radius=4)
        while name and small.size(name)[0] > box.w - 10:      # a long name is ellipsized, never overflowed:
            name = name[:-2] + "…"                       # the arrows either side must stay clickable
        tw = small.size(name)[0]
        widgets.text(surface, small, name, (box.centerx - tw // 2, box.y + 5), INK)
        widgets.text(surface, small, f"{ix + 1}/{len(nn_ckpts)}",
              (box.right - 44, box.y + 5), MUTE)
    widgets.text(surface, small, preview, (bx + 20, by + bh - 72), MUTE)

    close = pygame.Rect(bx + bw - 20 - 78, by + bh - 44, 78, 28)
    pygame.draw.rect(surface, BTN_HOVER if close.collidepoint(mouse) else BTN, close, border_radius=4)
    widgets.text(surface, small, "Close", (close.x + 18, close.y + 5), INK)
    return {"pills": pills, "sliders": sliders, "close": close,
            "ckpt_prev": ckpt_prev, "ckpt_next": ckpt_next}

