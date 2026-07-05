"""Post-game review screen: step through a completed MCTS-vs-MCTS game and inspect each decision's
search tree as a ply-banded icicle (or a collapsible outline), with P0's and P1's trees side by side.

    python -m imposterkings.ui.review --iters 800 --seed 0

The reviewed game is generated headlessly (both seats MCTS, so every ply has a real recorded tree),
then the window opens on the in-memory trajectory. Left panel = P0's most-recent decision, right =
P1's; each is drawn from its own seat's perspective (+ = good for that player).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .. import cards
from ..actions import Action, ActionKind, StepKind
from ..infoset import InformationSet
from .render import (BTN, BTN_HOVER, CARD_COLORS, DIVIDER, GOLD, INK, MUTE, P_COLORS, PANEL, RED,
                     WINDOW, _compact_action, _cross, _text, _tick, make_fonts)
from . import assets
from .tree_view import block_at, draw_crown, draw_icicle, draw_outline, draw_tooltip, graft_node

HEADER_H = 76        # title + ply line + button row
# Timeline band: one combined eval graph (both players' lines, shared x & y axes) then the card strip.
TL_TOP = 84
GRAPH_H = 130
GRAPH_GAP = 6
STRIP_H = 62
X0, PAD = 8, 12      # left edge + inner padding for the shared time axis
STRIP_TOP = TL_TOP + GRAPH_H + GRAPH_GAP


def _turn_x(i: int, n: int) -> float:
    """Shared x for turn index ``i`` of ``n`` turns — used by both graphs and the strip so they align."""
    w = WINDOW[0] - 16
    return X0 + PAD + (w - 2 * PAD) * (i / max(1, n - 1))


@dataclass
class PlyRecord:
    seat: int
    move: Action
    view: InformationSet
    result: object            # Optional[SearchResult] (with .root); None for a forced/no-search move
    state: object = None      # the full pre-move GameState (for the board view)
    eval_by_seat: Optional[Tuple[float, float]] = None  # each seat's own-perspective eval at a turn start
    result_by_seat: Optional[Tuple[object, object]] = None  # each seat's SearchResult of this turn's position


def _search_from(state, observer: int, iters: int, rng):
    """Run a search from seat ``observer``'s information set of ``state`` and return the SearchResult
    (its ``info.observer`` is ``observer``, so the icicle draws it in that seat's perspective)."""
    from ..mcts import SearchConfig, search
    return search(state.information_set(observer), SearchConfig(rng=rng, iterations=iters))


def _result_eval(res, state, observer: int) -> float:
    """A SearchResult's value from ``observer``'s perspective (+1 = good for observer). ``root_value()`` is
    the *mover's* perspective, so it is negated when the observer is not to move (zero-sum leaves)."""
    rv = res.root_value()
    return rv if state.to_play == observer else -rv


def budget_iters(state, observer: int, bud) -> int:
    """Iterations for a search from ``observer``'s view under a budget policy ``bud`` (mover-weighted
    branching + the observer's opp-card uncertainty). Shared by the app's live analysis and the review."""
    mover_moves = state.information_set(state.to_play).legal_moves()
    return bud(state.information_set(observer), mover_moves)


def build_trajectory(iters: int, seed: Optional[int], start: Optional[int] = None,
                     cross_eval: bool = True, budget=None, initial_state=None) -> List[PlyRecord]:
    """Play one MCTS-vs-MCTS game and record (seat, move, pre-move view, SearchResult, state) per ply.

    ``budget`` (a :mod:`imposterkings.budget` policy) makes both self-play agents AND the dual-eval use
    that per-decision budget (e.g. the hybrid schedule); ``None`` falls back to fixed ``iters``.
    ``initial_state`` (e.g. from ``scenario.build``) plays from a constructed position instead of a deal --
    the review/icicle can then be exercised on a specific opening.

    With ``cross_eval`` (default), also fill each turn-start ply's ``eval_by_seat`` with BOTH players'
    own-perspective evals of that position -- the mover's from its own search, the opponent's from one
    extra search of the same state -- so the review graph can show every player's read on every turn."""
    from ..agents import MCTSAgent
    from ..arena import play_game

    traj: List[PlyRecord] = []

    def collect(seat, view, move, agent, state):
        traj.append(PlyRecord(seat, move, view, getattr(agent, "last_result", None), state))

    rng = np.random.default_rng(seed)
    # evaluate_forced: search even on forced turns (ascensions, sole reactions) so every turn -- not just
    # ones with a real choice -- carries an eval. A position's value is well-defined even when forced.
    def _agent():
        return (MCTSAgent(budget=budget, evaluate_forced=True) if budget is not None
                else MCTSAgent(iterations=iters, evaluate_forced=True))
    play_game([_agent(), _agent()], rng, on_decision=collect, starting_player=start,
              initial_state=initial_state)

    if cross_eval:
        annotate_dual_evals(traj, budget if budget is not None else iters, np.random.default_rng(seed))
    return traj


def scripted_trajectory(state, moves, *, iters: int = 120, seed: int = 0, budget=None,
                        search: bool = True, cross_eval: bool = True) -> List[PlyRecord]:
    """Drive a FIXED sequence of ``moves`` (Actions) from ``state`` and record a review-ready trajectory.

    Unlike ``build_trajectory`` (bot-chosen moves), the line is scripted -- for reproducing an exact rules
    interaction (e.g. an Oathbound->Inquisitor->King's-Hand counter) and inspecting it in the review. When
    ``search`` (default), each ply is first searched from the mover's seat so ``PlyRecord.result`` (and
    thus the icicle/graft) is populated, THEN the scripted move is applied regardless of what the search
    preferred. ``cross_eval`` fills the dual-eval/result_by_seat as usual. Raises if a scripted move is
    illegal at its ply."""
    traj: List[PlyRecord] = []
    rng = np.random.default_rng(seed)
    st = state
    for i, mv in enumerate(moves):
        if st.is_terminal():
            break
        seat = st.to_play
        legal = st.legal_moves()
        if mv not in legal:
            raise ValueError(f"scripted move {i} ({mv}) is illegal; legal = {legal}")
        result = None
        if search and len(legal) > 1:                 # skip a search on forced single-move plies
            its = iters if budget is None else budget_iters(st, seat, budget)
            result = _search_from(st, seat, its, rng)
        traj.append(PlyRecord(seat, mv, st.information_set(seat), result, state=st))
        st = st.apply(mv)
    if cross_eval:
        annotate_dual_evals(traj, budget if budget is not None else iters, rng)
    return traj


def annotate_dual_evals(traj: List[PlyRecord], iters, rng) -> int:
    """Fill each turn-start ply's ``result_by_seat``/``eval_by_seat`` with BOTH seats' reads of that
    position, REUSING any already present (e.g. stored live by the app during play) and only searching
    the GAPS: a missing mover seat reuses the ply's own recorded ``result`` if it has one, else searches;
    a missing opponent seat searches. ``iters`` is a fixed int OR a budget policy callable (then each gap
    search is sized per-turn via ``budget_iters`` -- so an app review reuses its hybrid/branch budget, not
    a flat number). Returns the number of fresh searches performed (0 if everything was computed live)."""
    computed = 0
    for start_i, _end, _owner in turns_of(traj):
        rec = traj[start_i]
        if rec.state is None:
            continue
        mover = rec.state.to_play                     # the seat that searched here (== owner except in setup)
        rbs = list(rec.result_by_seat) if rec.result_by_seat is not None else [None, None]
        for s in (0, 1):
            if rbs[s] is not None:
                continue                              # already computed (e.g. live during the game)
            if s == mover and rec.result is not None:
                rbs[s] = rec.result                   # reuse the mover's actual decision search
            else:
                its = iters if isinstance(iters, int) else budget_iters(rec.state, s, iters)
                rbs[s] = _search_from(rec.state, s, its, rng)
                computed += 1
        rec.result_by_seat = (rbs[0], rbs[1])
        rec.eval_by_seat = (_result_eval(rbs[0], rec.state, 0), _result_eval(rbs[1], rec.state, 1))
    return computed


_SETUP_KINDS = (StepKind.SETUP_HIDE, StepKind.SETUP_DISCARD)


def _owner_key(rec: PlyRecord) -> int:
    """Who owns this ply's turn. Normally ``turn_player`` (constant through a whole turn incl. the
    opponent's reactions). During SETUP there is no real turn order -- ``turn_player`` stays on the
    starting player while BOTH seats hide/discard -- so attribute setup plies to the actual mover."""
    pending = getattr(rec.view, "pending", None)
    if pending and pending[-1].kind in _SETUP_KINDS:
        return rec.seat
    return rec.view.turn_player


def turns_of(traj: List[PlyRecord]) -> List[Tuple[int, int, int]]:
    """Group plies into turns: maximal runs with the same owner (``turn_player`` normally; the mover
    during setup, which has no real turn order). Returns ``[(start, end, owner), ...]``."""
    out: List[Tuple[int, int, int]] = []
    i = 0
    while i < len(traj):
        owner = _owner_key(traj[i])
        j = i
        while j + 1 < len(traj) and _owner_key(traj[j + 1]) == owner:
            j += 1
        out.append((i, j, owner))
        i = j + 1
    return out


def turn_for_seat(turns, seat: int, cursor: int) -> Optional[Tuple[int, int]]:
    """The latest turn OWNED by ``seat`` whose start is at/before ``cursor`` -> (start, end)."""
    best = None
    for s, e, owner in turns:
        if owner == seat and s <= cursor:
            best = (s, e)
    return best


def played_path(traj: List[PlyRecord], start: int, upto: int) -> List[Action]:
    """The moves actually taken in a turn, from its first ply up to ``upto`` (the played line)."""
    return [traj[j].move for j in range(start, min(upto, len(traj) - 1) + 1)]


def _button(surface, font, label, rect, *, active=False, hover=False):
    import pygame
    r = pygame.Rect(rect)
    pygame.draw.rect(surface, GOLD if active else (BTN_HOVER if hover else BTN), r, border_radius=4)
    surface.blit(font.render(label, True, (20, 20, 20) if active else INK),
                 (r.x + 8, r.y + 4))
    return r


def _current_turn(turns, cursor) -> int:
    """Index into ``turns`` of the turn containing ``cursor`` (0 if none)."""
    return next((i for i, (s, e, o) in enumerate(turns) if s <= cursor <= e), 0)


def _turn_eval(rec, owner: int, seat: int) -> Optional[float]:
    """Seat ``seat``'s own-perspective eval at a turn start. Uses the both-players ``eval_by_seat`` when
    present; otherwise falls back to the mover's search (owner side only)."""
    if rec.eval_by_seat is not None:
        return rec.eval_by_seat[seat]
    if seat == owner and rec.result is not None:
        return rec.result.root_value()
    return None


def _draw_graph(surface, fonts, traj, turns, cursor, top):
    """The combined eval graph: BOTH players' lines on one set of shared axes (x = turns, y = -1..+1).
    With cross-evals every turn carries both seats' reads, so each line has a point on EVERY turn (a
    seat's line moves the turn right after it learns something). Each line stays in its own owner's
    perspective (+1 = good for that player), so the two lines together show the P0/P1 asymmetry."""
    import pygame
    small = fonts["small"]
    x0, w, h = X0, WINDOW[0] - 16, GRAPH_H
    mid = top + h / 2
    pygame.draw.rect(surface, (24, 26, 32), (x0, top, w, h))
    pygame.draw.line(surface, DIVIDER, (x0, mid), (x0 + w, mid))
    _text(surface, small, "eval  +1 / 0 / -1", (x0 + 4, top - 1), MUTE)
    _text(surface, small, "P0", (x0 + w - 74, top - 1), P_COLORS[0])
    _text(surface, small, "P1", (x0 + w - 44, top - 1), P_COLORS[1])
    n = len(turns)
    if n == 0:
        return
    ypx = lambda v: mid - (h / 2 - 8) * max(-1.0, min(1.0, v))
    for seat in (0, 1):                                # both lines on the same axes
        prev = None
        for i, (s, e, owner) in enumerate(turns):
            v = _turn_eval(traj[s], owner, seat)
            if v is None:                              # no eval for this seat here -> break the line
                prev = None
                continue
            x, y = _turn_x(i, n), ypx(v)
            if prev is not None:
                pygame.draw.line(surface, P_COLORS[seat], prev, (x, y), 2)
            pygame.draw.circle(surface, P_COLORS[seat], (int(x), int(y)), 3)
            prev = (x, y)
    cx = int(_turn_x(_current_turn(turns, cursor), n))
    pygame.draw.line(surface, GOLD, (cx, top), (cx, top + h), 1)


def _headline_card(traj, start, end):
    """The card the turn's owner PLAYED (the headline of that turn), or None (e.g. a king flip / setup)."""
    for j in range(start, end + 1):
        mv = traj[j].move
        if mv.kind == ActionKind.PLAY_CARD and mv.card is not None:
            return mv.card
    return None


def _turn_is_flip(traj, start, end) -> bool:
    """True if this turn was a king flip (its headline is FLIP_KING, not a card play)."""
    return any(traj[j].move.kind == ActionKind.FLIP_KING for j in range(start, end + 1))


def _draw_strip(surface, fonts, traj, turns, cursor):
    """Card replay strip: each turn's headline card as mini art, left->right, aligned to ``_turn_x`` so
    the columns line up with both graphs above. Current turn is raised + gold-bordered. Returns
    ``[(rect, turn_start)]`` for click-to-jump."""
    import pygame
    small = fonts["small"]
    n = len(turns)
    if n == 0:
        return []
    spacing = (WINDOW[0] - 16 - 2 * PAD) / max(1, n - 1)
    ch = STRIP_H - 12                                  # leave room for the raised current card + border
    cw = min(int(round(ch / 1.4)), int(max(12, spacing - 2)))
    ch = int(round(cw * 1.4))
    cur = _current_turn(turns, cursor)
    hits = []
    for i, (s, e, owner) in enumerate(turns):
        cx = _turn_x(i, n)
        is_cur = i == cur
        y = STRIP_TOP + (0 if is_cur else 6)
        rect = pygame.Rect(int(cx - cw / 2), int(y), cw, ch)
        card = _headline_card(traj, s, e)
        if card is not None:
            try:
                surface.blit(assets.card_surface(card, (cw, ch)), rect)
            except Exception:                          # missing art -> solid card-colored tile
                pygame.draw.rect(surface, CARD_COLORS.get(cards.card_name(card), MUTE), rect)
        elif _turn_is_flip(traj, s, e):                # king flip -> white cell + upside-down crown
            pygame.draw.rect(surface, (245, 245, 245), rect)
            draw_crown(surface, rect, flipped=True)
        else:                                          # first play (setup hide/discard) -> white + upright crown
            pygame.draw.rect(surface, (245, 245, 245), rect)
            draw_crown(surface, rect, flipped=False)
        pygame.draw.rect(surface, P_COLORS[owner], rect, 2)   # always the player's color (blue P0 / orange P1)
        if is_cur:                                            # selection: white ring (not gold, which reads as P1)
            pygame.draw.rect(surface, INK, rect.inflate(6, 6), 2)
        hits.append((rect, s))
    return hits


class _GraftedResult:
    """A minimal SearchResult stand-in (``.root`` + ``.info``) wrapping a synthetic, grafted tree so it
    can be handed straight to ``draw_icicle`` without touching the real cached search results."""
    __slots__ = ("root", "info")

    def __init__(self, root, info):
        self.root = root
        self.info = info


def _collect_subtree_ids(node, out: set) -> None:
    """Add ``id`` of ``node`` and all its descendants to ``out`` (for dimming a superseded branch)."""
    stack = [node]
    while stack:
        n = stack.pop()
        out.add(id(n))
        stack.extend(n.children.values())


def _grafted_tree(traj, seat: int, start: int, end: int, cursor: int, res0, *, renormalise: bool = False):
    """Once the cursor steps INTO a turn, replace each stepped-past sub-band with that ply's OWN
    authoritative search (full budget), keeping the turn-root band fixed and greying the unchosen
    parent branches. Returns ``(grafted_result, graft_ids, dim_ids, tip_sims)`` or ``None`` when there
    is nothing to graft (cursor at the turn root, or every deeper ply was forced / the other seat's).

    Only this ``seat``'s own per-ply searches are grafted into this ``seat``'s panel, so the whole
    picture stays in one perspective (no value-sign/knowledge seam across bands).

    ``renormalise``: instead of greying the unchosen siblings' whole subtrees (contain mode), replace
    each unchosen sibling with a CHILDLESS greyed clone -- its subtree is dropped so the grafted band can
    be laid full-width (the caller passes ``renormalise=True`` to ``draw_icicle`` too)."""
    upto = min(cursor, end)
    steps_in = upto - start
    if steps_in <= 0 or res0 is None or getattr(res0, "root", None) is None:
        return None
    moves = played_path(traj, start, upto)                # moves[0] = turn-root choice, then one per step
    graft_ids: set = set()
    dim_ids: set = set()
    tip_sims = res0.root.n
    syn_root = graft_node(res0.root, dict(res0.root.children))   # copy the dict so splices don't mutate res0
    cur_syn, cur_orig = syn_root, res0.root
    for i in range(steps_in):
        orig_child = next((c for c in cur_orig.children.values() if c.incoming_move == moves[i]), None)
        if orig_child is None:
            break
        nxt = start + i + 1                               # the ply whose own search feeds the next band
        gsrc = None
        if nxt < len(traj) and traj[nxt].seat == seat:
            gr = traj[nxt].result
            if gr is not None and getattr(gr, "root", None) is not None and gr.root.children:
                gsrc = gr
        if gsrc is not None:                              # graft this ply's real search under the chosen node
            syn_child = graft_node(orig_child, dict(gsrc.root.children))
            graft_ids.add(id(syn_child))
            tip_sims = gsrc.root.n
            for c in cur_orig.children.values():          # grey the now-superseded unchosen siblings
                if c is orig_child:
                    continue
                if renormalise:                           # drop the subtree -> full width for the graft band
                    clone = graft_node(c, {})
                    cur_syn.children[c.incoming_move] = clone
                    dim_ids.add(id(clone))
                else:                                     # contain mode: keep the subtree, just grey it
                    _collect_subtree_ids(c, dim_ids)
            next_orig = gsrc.root                          # next chosen move lives in the grafted band
        else:                                             # forced / other seat: keep the original subtree
            syn_child = graft_node(orig_child, dict(orig_child.children))
            next_orig = orig_child
        cur_syn.children[moves[i]] = syn_child
        cur_syn, cur_orig = syn_child, next_orig
    if not graft_ids:
        return None                                       # nothing actually replaced -> draw res0 as today
    return _GraftedResult(syn_root, res0.info), graft_ids, dim_ids, tip_sims


def _stack_target_cards(root, base_state, *, top_k: int = 6, max_turns: int = 6) -> dict:
    """Map icicle node ``id`` -> the real card its ``CHOOSE_STACK_TARGET`` move (Fool/Sentry/Soldier)
    takes from the stack, for the whole SEARCH tree (not just the played line).

    The stack is public, so we replay the search tree from the true turn-start state ``base_state``
    (``traj[start].state``) through the engine, tracking the concrete ``GameState`` down each branch; a
    stack-target node resolves against its parent state's stack. Only LEGAL moves are followed, so
    determinization-only opponent branches (a reveal the opponent can't actually make) are skipped rather
    than replayed against a false state. Bounded by the same ``top_k``/``max_turns`` the icicle draws.
    Keyed by node id, so grafted clones (whose spine states match the true line) resolve too."""
    out: dict = {}
    if root is None or base_state is None:
        return out
    stack = [(root, base_state, 0)]                  # (node, concrete GameState, turn-band index)
    while stack:
        node, st, ti = stack.pop()
        board = getattr(st, "stack", None)
        legal = None
        for ch in sorted(node.children.values(), key=lambda c: c.n, reverse=True)[:top_k]:
            m = ch.incoming_move
            if m is None:
                continue
            if (m.kind == ActionKind.CHOOSE_STACK_TARGET and board is not None
                    and m.target is not None and 0 <= m.target < len(board)):
                out[id(ch)] = board[m.target].card
            ct = ti if ch.player_just_moved == node.player_just_moved else ti + 1
            if ct >= max_turns:
                continue
            if legal is None:                        # resolve legality once per node (against the TRUE state)
                try:
                    legal = set(st.legal_moves())
                except Exception:
                    legal = set()
            if m in legal:
                try:
                    stack.append((ch, st.apply(m), ct))
                except Exception:
                    pass
    return out


def _draw_panel(surface, fonts, traj, seat, turn, cursor, tree_rect, mode, ost, zoom_stack, last_tree,
                renorm=False):
    """Draw ``seat``'s read of the CURRENT turn's position. Both panels show the same turn: the mover's
    real search on its side, the opponent's search of the same state on the other -- so P1's tree updates
    during P0's turns too. The actually-played line is highlighted in both. Returns icicle blocks."""
    import pygame
    med, small = fonts["med"], fonts["small"]
    tx, ty = tree_rect[0], tree_rect[1]
    if turn is None:
        _text(surface, med, f"P{seat}: (no turn yet)", (tx + 4, ty - 42), MUTE)
        return []
    start, end, owner = turn
    rec0 = traj[start]
    is_mover = seat == owner
    path = played_path(traj, start, min(cursor, end))
    # this seat's SearchResult of the current position (mover reuses its own; opponent uses the cross search)
    res = (rec0.result_by_seat[seat] if rec0.result_by_seat is not None
           else (rec0.result if is_mover else None))
    if res is None:                                           # this seat has no search here -> persist dimmed
        head = (f"P{seat} — forced move: {_compact_action(traj[min(cursor, end)].move)}" if is_mover
                else f"P{seat} — (no read available)")
        _text(surface, med, head, (tx + 4, ty - 42), MUTE)
        prev = last_tree[seat]
        if prev is not None and mode == "icicle":
            draw_icicle(surface, fonts, prev[0], tree_rect, played_path=prev[1], dim=True)
        else:
            _text(surface, small, "(no search)", (tx + 4, ty + 6), MUTE)
        return []
    last_tree[seat] = (res, path)
    zoom_note = "  [zoomed — Backspace out]" if zoom_stack else ""
    forced_note = "  · forced" if len(res.stats) == 1 else ""
    header = (f"P{seat} — {_compact_action(rec0.move)}{forced_note}{zoom_note}   ◄◄ ACTIVE" if is_mover
              else f"P{seat} — reads P{owner}'s turn{zoom_note}")
    _text(surface, med, header, (tx + 4, ty - 42), P_COLORS[seat])
    if rec0.eval_by_seat is not None:                         # this seat's eval of the current position
        _text(surface, small, f"P{seat} eval {rec0.eval_by_seat[seat]:+.2f} (their perspective)",
              (tx + 4, ty - 22), MUTE)
    if mode == "icicle":
        zoom_root = zoom_stack[-1] if zoom_stack else None
        g = _grafted_tree(traj, seat, start, end, cursor, res,   # step into the turn -> authoritative sub-bands
                          renormalise=renorm)
        draw_res = g[0] if g is not None else res
        # @N -> real card: replay the search tree from the true turn-start state (stack is public)
        stack_cards = _stack_target_cards(draw_res.root, getattr(traj[start], "state", None))
        if g is not None:
            gres, graft_ids, dim_ids, tip_sims = g
            blocks = draw_icicle(surface, fonts, gres, tree_rect, played_path=path, zoom_root=zoom_root,
                                 graft_ids=graft_ids, dim_ids=dim_ids, band_sims=tip_sims,
                                 renormalise=renorm, stack_cards=stack_cards)
        else:
            blocks = draw_icicle(surface, fonts, res, tree_rect, played_path=path, zoom_root=zoom_root,
                                 stack_cards=stack_cards)
    else:
        blocks = draw_outline(surface, fonts, res, tree_rect, expanded=ost["exp"],
                              scroll=ost["scroll"], played_move=(path[-1] if path else None))
    if is_mover:                                              # box the active panel: a colored border just
        box = pygame.Rect(*tree_rect).inflate(6, 6)          # OUTSIDE the icicle (so it doesn't clip the
        pygame.draw.rect(surface, P_COLORS[seat], box, 3)    # top band bar), wrapped in a black outline
        pygame.draw.rect(surface, (10, 10, 10), box.inflate(4, 4), 2)
    return blocks


_POPUP_CW, _POPUP_CH = 36, 50       # card jpgs in the true-board popup (~strip size)


def _name_surface(name: str, size):
    """Card art for a card NAME (uses a representative instance) -- for the has/lacks knowledge rows."""
    ids = cards.card_ids_for_name(name)
    return assets.card_surface(ids[0], size) if ids else None


def _draw_board_popup(surface, fonts, state, anchor) -> None:
    """Hover popup: the TRUE board at a turn (post-game, so full info). R1 = P0 hand / hidden / antechamber
    card art; R2 = P1 likewise; R3 = each player's has(tick)/lacks(cross) deductions about the OTHER's hand
    as card art, then the starting face-up card. All from ``state`` (a GameState). No-op if ``state`` None."""
    import pygame
    if state is None:
        return
    small = fonts["small"]
    cw, ch, g, pad = _POPUP_CW, _POPUP_CH, 3, 8
    lh = small.get_linesize()
    art = lambda c: assets.card_surface(c, (cw, ch))

    # each row = list of ("label", text) | ("cards", [surf]) | ("tick"/"cross", [surf])
    rows = []
    for seat in (0, 1):
        row = [("label", f"P{seat}"), ("cards", [art(c) for c in state.hands[seat]])]
        if state.hidden[seat] is not None:
            row += [("label", "hid"), ("cards", [art(state.hidden[seat])])]
        if state.antechambers[seat]:
            row += [("label", "ante"), ("cards", [art(c) for c in state.antechambers[seat]])]
        rows.append(row)
    r3 = []
    for seat in (0, 1):
        v = state.information_set(seat)
        r3 += [("label", f"P{seat}?"),
               ("tick", [_name_surface(n, (cw, ch)) for n in sorted(v.opp_hand_has)]),
               ("cross", [_name_surface(n, (cw, ch)) for n in sorted(v.opp_hand_lacks)])]
    r3 += [("label", "up"), ("cards", [art(state.leftover_faceup)])]
    rows.append(r3)

    def seg_w(kind, val):
        if kind == "label":
            return small.size(val)[0] + 6
        icon = 16 if kind in ("tick", "cross") else 0
        return icon + (len(val) * (cw + g) if val else 14)     # 14 = "-" for an empty tick/cross

    W = max(sum(seg_w(k, v) for k, v in row) for row in rows) + 2 * pad
    H = len(rows) * (ch + 6) + 2 * pad
    x = max(4, min(anchor[0] + 14, WINDOW[0] - W - 4))
    y = max(4, min(anchor[1] + 14, WINDOW[1] - H - 4))
    pygame.draw.rect(surface, PANEL, (x, y, W, H))
    pygame.draw.rect(surface, MUTE, (x, y, W, H), 1)
    for i, row in enumerate(rows):
        cy, cx = y + pad + i * (ch + 6), x + pad
        for kind, val in row:
            if kind == "label":
                _text(surface, small, val, (cx, cy + (ch - lh) // 2), MUTE)
                cx += small.size(val)[0] + 6
                continue
            if kind in ("tick", "cross"):
                (_tick if kind == "tick" else _cross)(surface, cx, cy + (ch - 14) // 2)
                cx += 16
            if not val:
                _text(surface, small, "-", (cx, cy + (ch - lh) // 2), MUTE)
                cx += 14
            for s in val:
                if s is not None:
                    surface.blit(s, (cx, cy))
                cx += cw + g


def render_review_frame(screen, fonts, traj, turns, *, cursor, mode="icicle", renorm=False,
                        ost=None, zoom=None, last_tree=None, mouse=(-1, -1)):
    """Draw ONE review frame (nav bar + eval graph + card strip + both icicle panels + hover popup) and
    return its hit-test data ``{"btns","strip_hits","blocks","mid","ptop"}``. Factored out of
    ``run_review`` so the live loop and headless capture (``ui.headless.review_png``) share one renderer.
    ``ost``/``zoom``/``last_tree`` default to fresh per-panel state (fine for a single headless frame)."""
    import pygame
    med, small = fonts["med"], fonts["small"]
    W, H = WINDOW
    mid = W // 2
    ost = ost if ost is not None else {0: {"exp": set(), "scroll": 0}, 1: {"exp": set(), "scroll": 0}}
    zoom = zoom if zoom is not None else {0: [], 1: []}
    last_tree = last_tree if last_tree is not None else {0: None, 1: None}

    screen.fill(PANEL)
    rec = traj[cursor]
    pend = getattr(rec.view, "pending", None)
    is_setup = bool(pend) and pend[-1].kind in _SETUP_KINDS
    turn_label = "Setup" if is_setup else f"turn P{turns[_current_turn(turns, cursor)][2]}"

    _text(screen, med, "Post-game review  (MCTS vs MCTS)", (12, 6), INK)
    _text(screen, small, f"Ply {cursor + 1}/{len(traj)}  —  {turn_label}  ·  P{rec.seat} played "
                         f"{_compact_action(rec.move)}", (12, 32), GOLD)
    specs = [("prev", "◄ Prev", 12), ("next", "Next ►", 102), ("outline", "Outline", 210),
             ("icicle", "Icicle", 300), ("zout", "⬆ Zoom out", 390), ("renorm", "Renorm", 496)]
    btns = {}
    for key, label, bx in specs:
        r = pygame.Rect(bx, 50, 96 if key in ("zout", "renorm") else 84, 22)
        active = (key == "renorm" and renorm) or (key != "renorm" and key == mode)
        btns[key] = _button(screen, small, label, r, active=active, hover=r.collidepoint(mouse))
    _text(screen, small, "◄/► step · I/O view · R renorm · click card/graph=jump · click box=zoom · "
                         "Backspace=out · Esc quit", (604, 54), MUTE)

    _draw_graph(screen, fonts, traj, turns, cursor, TL_TOP)
    strip_hits = _draw_strip(screen, fonts, traj, turns, cursor)
    ptop = STRIP_TOP + STRIP_H + 10
    pygame.draw.line(screen, DIVIDER, (mid, ptop - 4), (mid, H))

    top = ptop + 44                         # room for the panel header + eval subtitle above the icicle
    lrect = (6, top, mid - 12, H - top - 6)
    rrect = (mid + 6, top, W - mid - 12, H - top - 6)
    cur_turn = turns[_current_turn(turns, cursor)]        # both panels show the SAME (current) turn
    blocks = {
        0: _draw_panel(screen, fonts, traj, 0, cur_turn, cursor, lrect, mode,
                       ost[0], zoom[0], last_tree, renorm),
        1: _draw_panel(screen, fonts, traj, 1, cur_turn, cursor, rrect, mode,
                       ost[1], zoom[1], last_tree, renorm),
    }
    hover_turn = next((s for r, s in strip_hits if r.collidepoint(mouse)), None)
    if hover_turn is not None:               # hovering a strip card -> the true-board popup for its turn
        _draw_board_popup(screen, fonts, traj[hover_turn].state, mouse)
    elif mode == "icicle":                   # else the icicle hover tooltip
        hseat = 0 if mouse[0] < mid else 1
        hb = block_at(blocks[hseat], mouse)
        if hb is not None:
            draw_tooltip(screen, fonts, hb, mouse)
    return {"btns": btns, "strip_hits": strip_hits, "blocks": blocks, "mid": mid, "ptop": ptop}


def run_review(screen, fonts, traj: List[PlyRecord]) -> None:
    import pygame
    if not traj:
        return
    clock = pygame.time.Clock()
    W, H = WINDOW
    cursor, mode = 0, "icicle"
    renorm = False                              # graft sub-bands: contain (False) vs full-width renormalise
    ost = {0: {"exp": set(), "scroll": 0}, 1: {"exp": set(), "scroll": 0}}
    zoom = {0: [], 1: []}                       # per-panel node zoom stacks
    last_tree = {0: None, 1: None}              # last non-None (result, played_path) per seat
    mid = W // 2
    turns = turns_of(traj)

    running = True
    while running:
        mouse = pygame.mouse.get_pos()
        hit = render_review_frame(screen, fonts, traj, turns, cursor=cursor, mode=mode, renorm=renorm,
                                  ost=ost, zoom=zoom, last_tree=last_tree, mouse=mouse)
        btns, strip_hits, blocks, ptop = hit["btns"], hit["strip_hits"], hit["blocks"], hit["ptop"]
        pygame.display.flip()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key == pygame.K_LEFT:
                    cursor = max(0, cursor - 1)
                elif e.key == pygame.K_RIGHT:
                    cursor = min(len(traj) - 1, cursor + 1)
                elif e.key == pygame.K_o:
                    mode = "outline"
                elif e.key == pygame.K_i:
                    mode = "icicle"
                elif e.key == pygame.K_r:
                    renorm = not renorm
                elif e.key == pygame.K_BACKSPACE:
                    for z in zoom.values():
                        if z:
                            z.pop()
            elif e.type == pygame.MOUSEWHEEL and mode == "outline":
                ost[0 if mouse[0] < mid else 1]["scroll"] = max(
                    0, ost[0 if mouse[0] < mid else 1]["scroll"] - e.y)
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 3:   # right-click zooms out
                zseat = 0 if e.pos[0] < mid else 1
                if zoom[zseat]:
                    zoom[zseat].pop()
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                pos = e.pos
                if btns["prev"].collidepoint(pos):
                    cursor = max(0, cursor - 1)
                elif btns["next"].collidepoint(pos):
                    cursor = min(len(traj) - 1, cursor + 1)
                elif btns["outline"].collidepoint(pos):
                    mode = "outline"
                elif btns["icicle"].collidepoint(pos):
                    mode = "icicle"
                elif btns["renorm"].collidepoint(pos):
                    renorm = not renorm
                elif btns["zout"].collidepoint(pos):
                    for z in zoom.values():
                        if z:
                            z.pop()
                elif TL_TOP <= pos[1] < ptop - 6 and turns:   # timeline (graphs + card strip) -> jump
                    hit = next((s for r, s in strip_hits if r.collidepoint(pos)), None)
                    if hit is not None:
                        cursor = hit
                    else:
                        i = min(range(len(turns)), key=lambda k: abs(_turn_x(k, len(turns)) - pos[0]))
                        cursor = turns[i][0]
                elif mode == "icicle":                                # click a box -> zoom in
                    zseat = 0 if pos[0] < mid else 1
                    hb = block_at(blocks[zseat], pos)
                    if hb is not None and hb.node.children and (
                            not zoom[zseat] or hb.node is not zoom[zseat][-1]):  # not the visible root band
                        zoom[zseat].append(hb.node)
                elif mode == "outline":                               # click a row -> expand/collapse
                    for seat_, hitmap in blocks.items():
                        for row_rect, pth in hitmap:
                            if row_rect.collidepoint(pos):
                                ost[seat_]["exp"] ^= {pth}
                                break
        clock.tick(30)


def main(argv=None) -> None:
    import time

    import pygame
    from .. import budget as budget_mod

    p = argparse.ArgumentParser(description="Post-game review of an MCTS-vs-MCTS game.")
    p.add_argument("--p1", default="hybrid", choices=["mcts", "hybrid", "branching"],
                   help="engine mode for both bots + the dual-eval (default: hybrid)")
    p.add_argument("--iters", type=int, default=800, help="MCTS iterations per decision (mcts mode)")
    p.add_argument("--k", type=int, default=50, help="budget multiplier (hybrid/branching)")
    p.add_argument("--l", type=int, default=3,
                   help="effective legal-moves for a sub-decision card at selection (hybrid/branching)")
    p.add_argument("--seed", type=int, default=0, help="deck/deal seed")
    p.add_argument("--start", type=int, default=None, choices=[0, 1], help="force the starting player")
    args = p.parse_args(argv)

    bud = None if args.p1 == "mcts" else budget_mod.make_budget(args.p1, k=args.k, l=args.l)
    cfg = f"iters={args.iters}" if bud is None else bud.label
    print(f"Generating MCTS-vs-MCTS game ({cfg}, seed={args.seed})...")
    t0 = time.perf_counter()
    traj = build_trajectory(args.iters, args.seed, args.start, budget=bud)
    dt = time.perf_counter() - t0
    print(f"  {len(traj)} decisions recorded in {dt:.1f}s. Opening review window...")

    pygame.init()
    screen = pygame.display.set_mode(WINDOW)
    pygame.display.set_caption(f"ImposterKings review  (seed {args.seed}, {cfg})")
    run_review(screen, make_fonts(), traj)
    pygame.quit()


if __name__ == "__main__":
    main()
