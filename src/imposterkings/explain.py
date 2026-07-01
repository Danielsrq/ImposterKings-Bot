"""Human-readable formatters for actions, information sets, and MCTS search results.

Pure string helpers with no I/O, so the CLI, the UI, and record-keeping share them. The search-result
table is filled in alongside the MCTS agent in Phase 3.
"""
from __future__ import annotations

from typing import Optional

from . import cards
from .actions import Action, ActionKind, StepKind


def _stack_value(view, sc) -> int:
    """Effective value of a stack card from public info (mirrors GameState.effective_stack_value)."""
    if sc.disgraced:
        return 0
    base = cards.card_value(sc.card)
    if base in view.muted_values:
        return 3
    if sc.value_override is not None:
        return sc.value_override
    return base


def format_action(action: Action, view=None) -> str:
    k = action.kind
    if k in (ActionKind.HIDE_CARD, ActionKind.DISCARD_CARD, ActionKind.PLAY_CARD,
             ActionKind.CHOOSE_HAND_CARD):
        return f"{k.name.lower()}({cards.format_card(action.card)})"
    if k == ActionKind.GUESS_CARD:
        return f"guess({action.name})"
    if k == ActionKind.CHOOSE_NUMBER:
        return f"number({action.number})"
    if k == ActionKind.CHOOSE_STACK_TARGET:
        if view is not None and 0 <= action.target < len(view.stack):
            sc = view.stack[action.target]
            return f"target(@{action.target} {cards.card_name(sc.card)})"
        return f"target(@{action.target})"
    return k.name.lower()


def format_infoset(view) -> str:
    """Everything the acting player can legally see, plus the current decision."""
    lines = [f"=== seat {view.observer} | decision: {view.pending[-1].kind.name} ==="]

    if view.stack:
        parts = []
        for i, sc in enumerate(view.stack):
            tag = "DISGRACED" if sc.disgraced else f"{cards.card_name(sc.card)}={_stack_value(view, sc)}"
            lead = " <-lead" if i == len(view.stack) - 1 else ""
            parts.append(f"[{i}]{tag}{lead}")
        lines.append("  stack: " + " ".join(parts))
    else:
        lines.append("  stack: (empty)")

    opp = 1 - view.observer
    lines.append(f"  opponent: {view.opp_hand_count} cards, king={'USED' if view.kings[opp] else 'up'}"
                 + ("" if not view.opp_has_hidden else ", has hidden"))
    if any(view.antechambers):
        for seat, ante in enumerate(view.antechambers):
            if ante:
                lines.append(f"  antechamber[{seat}]: " + ", ".join(cards.card_name(c) for c in ante))
    if view.muted_values:
        lines.append(f"  muted values: {sorted(view.muted_values)}")
    lines.append(f"  your king: {'USED' if view.kings[view.observer] else 'up'}"
                 + (f", hidden={cards.card_name(view.own_hidden)}" if view.own_hidden is not None else ""))
    lines.append("  your hand: " + ", ".join(cards.format_card(c) for c in view.own_hand))
    return "\n".join(lines)


def format_search_result(result, top: Optional[int] = None) -> str:  # pragma: no cover - Phase 3
    """Ranked candidate table from an MCTS SearchResult (filled in with the MCTS agent)."""
    lines = [f"MCTS: {result.iterations} sims in {result.elapsed:.2f}s -> {format_action(result.best_move)}"]
    lines.append(f"  {'move':<28} {'visits':>7} {'share':>6} {'meanQ':>7} {'avail':>6}")
    for s in result.stats[:top] if top else result.stats:
        lines.append(f"  {format_action(s.move):<28} {s.visits:>7} {s.visit_share:>6.2f} "
                     f"{s.mean_q:>7.3f} {s.avail:>6}")
    return "\n".join(lines)


def _pv_label(action) -> str:
    """Compact move label for a PV line (card name+value / guess / etc.)."""
    k = action.kind
    if k in (ActionKind.PLAY_CARD, ActionKind.HIDE_CARD, ActionKind.DISCARD_CARD,
             ActionKind.CHOOSE_HAND_CARD):
        return f"{cards.card_name(action.card)}({cards.card_value(action.card)})"
    if k == ActionKind.GUESS_CARD:
        return f"guess {action.name}"
    if k == ActionKind.CHOOSE_NUMBER:
        return f"mute {action.number}"
    if k == ActionKind.CHOOSE_STACK_TARGET:
        return f"target@{action.target}"
    return k.name.lower()


def format_pv_lines(result, top: int = 2, depth: int = 6) -> str:
    """Chess-engine-style principal variations: ``[eval] move[P0] move[P1] …`` for the top lines."""
    lines = result.principal_variations(top=top, depth=depth)
    if not lines:
        return ""
    out = ["Principal variations:"]
    for line in lines:
        moves = " ".join(f"{_pv_label(s.move)}[P{s.player}]" for s in line)
        out.append(f"  [{line[0].mean_q:+.2f}] {moves}")
    return "\n".join(out)
