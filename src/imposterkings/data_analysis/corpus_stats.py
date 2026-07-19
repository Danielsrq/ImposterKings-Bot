"""Corpus statistics: who wins, what the bot keeps, which cards and powers actually pay.

    python -m imposterkings.data_analysis.corpus_stats --data datasets/selfplay_gen1_v3c
    python -m imposterkings.data_analysis.corpus_stats --data datasets/selfplay_gen1_v3c \\
        datasets/selfplay_v3c_k50 --out-dir results/corpus_stats/v3c_selfplay --max-combo 4

Every number here is computed by REPLAYING the game and asking the engine, never by re-deriving state from
the ply log. That is the whole design: "did the Warlord's bonus fire" is ``state.royalty_present()``
(``abilities.py:139``), and reconstructing that from the recorded actions alone would mean reimplementing
disgraces, mutes, King's-Hand removals and Fool grabs -- i.e. reimplementing the engine, with new bugs.

THREE THINGS THAT WILL MISLEAD YOU IF YOU FORGET THEM
  1. **A hand-row is (game, seat), not a game.** One deal makes TWO different 8-card hands with (usually)
     opposite outcomes. Pooled over both seats the base winrate is exactly 50% -- which is the invariant
     that proves the unit is right (a combo worth nothing sits at 50%, so the numbers self-calibrate).
  2. **The played setup move is a SAMPLE, not the policy.** datagen ran at ``temp_plies: 6`` and setup is
     plies 1-2, so ``chosen`` was drawn from the visit distribution. The bot's actual preference is
     ``argmax visit_share`` over the candidates. Both are reported; they are different questions.
  3. **Corpora do not pool freely.** Turn-order pools across symmetric self-play (seat effects cancel by
     construction). Setup policy does not pool across different bots or search budgets, and the asymmetric
     corpus needs its agent x turn-order 2x2 or the strong bot's record swamps the weak one's.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

from .. import cards
from ..actions import ActionKind, StepKind
from ..cards import Ability
from ..state import GameState
from .replay import Game, Ply, replay_all

# The five abilities that open an ABILITY_MAY window, and the action that MEANS "I used it".
# (Mystic/Inquisitor/Fool are flattened -- they declare their parameter directly in the window rather than
#  declaring, waiting for the King's-Hand answer, and only then choosing. See generate.py:67-81.)
MAY_USE_ACTION = {
    Ability.SENTRY: ActionKind.DECLARE_ABILITY,        # declaring == self-disgrace, then the swap
    Ability.PRINCESS: ActionKind.DECLARE_ABILITY,
    Ability.MYSTIC: ActionKind.CHOOSE_NUMBER,          # the muted base value is the parameter
    Ability.INQUISITOR: ActionKind.GUESS_CARD,
    Ability.FOOL: ActionKind.CHOOSE_STACK_TARGET,      # which stack card to take back
}
PLAY_PHASES = (StepKind.MAIN, StepKind.ASCEND, StepKind.OATHBOUND_SECOND)
GUESS_ABILITIES = (Ability.SOLDIER, Ability.JUDGE)     # mandatory; Inquisitor's guess lives in ABILITY_MAY


def wilson(wins: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Point estimate + Wilson score interval.

    Not the textbook normal interval: at the sample sizes that matter here (a 4-card combo shows up in
    ~2% of hands) the normal interval runs off the end of [0,1] and understates the uncertainty. Wilson
    stays inside the unit interval and behaves at small n, which is the whole point of printing an interval.
    """
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = wins / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def override_fired(st: GameState, seat: int, card: int, ascended: bool) -> Optional[str]:
    """Did a passive play-OVERRIDE let this card be played, and which one?

    Mirrors ``abilities._can_play`` exactly. A passive has no decision window, so "was the power used" can
    only mean "did its special rule change what was legal". The test is: the card's effective value does NOT
    reach the leading value, yet the play was legal -- so something overrode it.

    Warlord is deliberately NOT here: its royalty bonus is a VALUE change (+1 in hand, lands at 9), already
    folded into ``effective_hand_value``. It is handled separately, or it would be invisible.
    """
    v_top = st.leading_value()
    if v_top is None or ascended:                 # empty stack, or ascension already granted the play
        return None
    if st.effective_hand_value(card) >= v_top:
        return None                               # it beat the throne honestly -- no override needed
    if cards.card_value(card) in st.muted_values:
        return None                               # muted: a bare 3 with no abilities at all
    lead = st.leading
    lead_royalty = lead is not None and st.active_royalty(lead)
    ability = cards.card_ability(card)
    if ability == Ability.OATHBOUND and v_top > 6 and len(st.hands[seat]) >= 2:
        return "oathbound"
    if ability == Ability.ELDER and lead_royalty:
        return "elder"
    if ability == Ability.ZEALOT and st.kings[seat] and lead is not None and not lead_royalty:
        return "zealot"
    return None


def win_reason(g: Game) -> str:
    """Classify HOW the game ended. The engine has exactly two ``winner=`` paths, so this is a complete
    taxonomy rather than a heuristic: the Assassin's instant kill (abilities.py:456) and the stall-out
    (state.py:234-238, "no card beats the leading card and no king to flip")."""
    last = g.plies[-1]
    if last.phase == StepKind.REACTION_KH_VS_ASSASSIN and last.action.kind == ActionKind.DECLINE_REACTION:
        return "assassin_kills_king"
    if last.action.kind == ActionKind.REVEAL_KINGSHAND:
        # A King's Hand can WIN outright: countering removes the played card from the stack, so the
        # throne reverts to whatever sat beneath -- and the active player may have nothing that beats it
        # (abilities.py:360 resumes _begin_turn, which is where the stall-out fires).
        return "kingshand_counter_stalls_them"
    st, loser = g.terminal, 1 - g.winner
    if not st.hands[loser]:
        return "stalled_out_of_cards"
    if st.kings[loser]:
        return "stalled_king_spent"              # holds cards, none beat the throne, king already flipped
    return "stalled_no_stack"                    # the rare empty-throne case


# --------------------------------------------------------------------------------------------------------


class Stats:
    """Accumulates every table over one corpus (or one pooled group)."""

    def __init__(self, name: str, max_combo: int = 4):
        self.name, self.max_combo = name, max_combo
        self.games = self.desynced = 0
        self.asymmetric = False              # set on ingest: are the two seats different bots?
        # (agent, went_first) -> [n, wins]
        self.turn: Dict[Tuple[str, bool], List[int]] = defaultdict(lambda: [0, 0])
        # setup: card name -> counts, SPLIT BY SEAT ORDER. The bot faces a different game as P1 (who must
        # win from a 41% base) than as P2 (58%), so its keep/ditch policy is not one policy but two -- and a
        # single pooled column would average them into a policy neither seat actually plays.
        # held_* is the denominator: a card can only be hidden if it was dealt.
        self.hide = defaultdict(lambda: Counter())
        self.discard = defaultdict(lambda: Counter())
        # combo (sorted name-multiset) -> [n, wins, n_first, wins_first]
        self.combo: Dict[Tuple[str, ...], List[int]] = defaultdict(lambda: [0, 0, 0, 0])
        self.may = defaultdict(lambda: Counter())        # card name -> Counter of outcome buckets
        self.mystic_number: Counter = Counter()
        self.passive = defaultdict(lambda: Counter())    # card name -> Counter
        self.kh: Counter = Counter()                     # countered card name -> n
        self.kh_phase: Counter = Counter()
        self.guess = defaultdict(lambda: Counter())      # (source, guessed name) -> Counter
        self.stack_trade = defaultdict(lambda: Counter())  # (trade label, card name) -> Counter: which card
        self.fool = Counter()                            # Sentry swaps / Fool grabs, by NAME (index->card)
        self.assassin = Counter()
        self.reasons: Counter = Counter()
        self.last_pair: Counter = Counter()

    # -- ingest ------------------------------------------------------------------------------------------

    def add(self, g: Game) -> None:
        self.games += 1
        if not g.gen.get("self_play", True):
            self.asymmetric = True           # the two seats are different bots -- policy tables are pooled
        self._turn_order(g)
        self._setup_and_hands(g)
        self._plies(g)
        self._fool_and_assassin(g)
        self.reasons[win_reason(g)] += 1
        lp = g.plies[-1]
        self.last_pair[f"{lp.phase.name}/{lp.action.kind.name}"] += 1

    def _turn_order(self, g: Game) -> None:
        for seat in (0, 1):
            first = seat == g.starting_player
            cell = self.turn[(g.agent_of(seat), first)]
            cell[0] += 1
            cell[1] += int(g.won(seat))

    def _setup_and_hands(self, g: Game) -> None:
        for seat in (0, 1):
            hand = g.dealt.hands[seat]
            won = g.won(seat)
            first = seat == g.starting_player
            sfx = "first" if first else "second"
            for c in set(cards.card_name(x) for x in hand):     # 'held' is per NAME, not per instance
                self.hide[c]["held"] += 1
                self.hide[c][f"held_{sfx}"] += 1
                self.discard[c]["held"] += 1
                self.discard[c][f"held_{sfx}"] += 1
            for key in self._combos(hand):
                cell = self.combo[key]
                cell[0] += 1
                cell[1] += int(won)
                if first:
                    cell[2] += 1
                    cell[3] += int(won)

        for p in g.plies[:4]:
            if p.phase not in (StepKind.SETUP_HIDE, StepKind.SETUP_DISCARD):
                continue
            tbl = self.hide if p.phase == StepKind.SETUP_HIDE else self.discard
            sfx = "first" if p.seat == g.starting_player else "second"
            tbl[cards.card_name(p.action.card)]["played"] += 1
            tbl[cards.card_name(p.action.card)][f"played_{sfx}"] += 1
            pref = p.preferred()                     # the search's ACTUAL choice, before temperature
            if pref is not None:
                name = cards.card_name(pref["move"]["card"])
                tbl[name]["preferred"] += 1
                tbl[name][f"preferred_{sfx}"] += 1

    def _combos(self, hand: Sequence[int]) -> List[Tuple[str, ...]]:
        """Every name-multiset up to max_combo present in this 8-card hand.

        Deduped per hand: a hand holding both Elders would otherwise count the combo ("Elder",) twice, once
        per instance -- inflating every duplicate card's denominator and quietly halving its winrate."""
        names = [cards.card_name(c) for c in hand]
        out = set()
        for r in range(1, min(self.max_combo, len(names)) + 1):
            for combo in combinations(names, r):
                out.add(tuple(sorted(combo)))
        return list(out)

    def _plies(self, g: Game) -> None:
        for i, p in enumerate(g.plies):
            nxt = g.plies[i + 1] if i + 1 < len(g.plies) else None
            won = g.won(p.seat)

            if p.phase in PLAY_PHASES and p.action.kind == ActionKind.PLAY_CARD:
                self._passive(p, won)

            elif p.phase == StepKind.ABILITY_MAY:
                self._may(p, nxt, won)

            elif p.phase == StepKind.ABILITY_GUESS:
                self._guess(p, nxt)

            elif p.phase == StepKind.ABILITY_HAND_CARD:
                self._hand_card(p, won)

            elif p.action.kind == ActionKind.REVEAL_KINGSHAND:
                # WHAT it countered lives in the step, not the action. The nested anti-Assassin window
                # carries no `source` at all (abilities.py:444) -- key it by phase or you will silently
                # attribute the block to whatever card happened to be leading the stack.
                if p.phase == StepKind.REACTION_KH_VS_ASSASSIN:
                    self.kh["Assassin"] += 1
                elif p.step.source is not None:
                    self.kh[cards.card_name(p.step.source)] += 1
                self.kh_phase[p.phase.name] += 1

    def _passive(self, p: Ply, won: bool) -> None:
        st, card = p.state, p.action.card
        name = cards.card_name(card)
        muted = cards.card_value(card) in st.muted_values
        row = self.passive[name]
        row["played"] += 1
        row["won" if won else "lost"] += 1
        if muted:
            row["muted_at_play"] += 1
            return                                   # a muted card has NO ability -- nothing could fire
        if cards.card_ability(card) == Ability.WARLORD:
            if st.royalty_present():                 # abilities.py:139 -- it lands at 9 instead of 7
                row["fired"] += 1
                row["fired_won" if won else "fired_lost"] += 1
            return
        if override_fired(st, p.seat, card, ascended=p.phase == StepKind.ASCEND):
            row["fired"] += 1
            row["fired_won" if won else "fired_lost"] += 1

    def _may(self, p: Ply, nxt: Optional[Ply], won: bool) -> None:
        source = p.step.source
        name = cards.card_name(source)
        use = MAY_USE_ACTION.get(cards.card_ability(source))
        row = self.may[name]
        row["window_opened"] += 1
        if len(p.state.legal_moves()) == 1:          # only DECLINE on offer (e.g. the Fool with no targets)
            row["forced"] += 1
        if p.action.kind == ActionKind.DECLINE_ABILITY:
            row["declined"] += 1
            row["declined_won" if won else "declined_lost"] += 1
            return
        if use is not None and p.action.kind == use:
            row["activated"] += 1
            if cards.card_ability(source) == Ability.MYSTIC:
                self.mystic_number[p.action.number] += 1
            # The King's-Hand window opens on the DECLARATION, so the very next ply decides whether the
            # ability actually happened. (An Inquisitor whose guess MISSES gets no window at all -- there
            # is no effect to counter -- so 'no window' means it landed unopposed, not that it was skipped.)
            if nxt is not None and nxt.phase == StepKind.REACTION_KINGSHAND \
                    and nxt.action.kind == ActionKind.REVEAL_KINGSHAND:
                row["countered"] += 1
                row["countered_won" if won else "countered_lost"] += 1
            else:
                row["landed"] += 1
                row["landed_won" if won else "landed_lost"] += 1
                if cards.card_ability(source) == Ability.FOOL and p.action.target is not None:
                    grabbed = p.state.stack[p.action.target].card    # index -> the actual card grabbed
                    self._trade("Fool grab (stack->hand)", grabbed, won)
            if cards.card_ability(source) == Ability.INQUISITOR:
                self._guess(p, nxt)

    def _guess(self, p: Ply, nxt: Optional[Ply]) -> None:
        source, guesser = p.step.source, p.step.actor
        # The engine's own test (abilities.py:236): does the DEFENDER hold a card of that name, right now?
        # Hands mutate mid-game (Princess swap, Inquisitor yank, Judge queue, Fool grab), so this is the one
        # statistic that genuinely cannot be read off the log.
        held = any(cards.card_name(c) == p.action.name for c in p.state.hands[1 - guesser])
        countered = (nxt is not None and nxt.phase == StepKind.REACTION_KINGSHAND
                     and nxt.action.kind == ActionKind.REVEAL_KINGSHAND)
        row = self.guess[(cards.card_name(source), p.action.name)]
        row["n"] += 1
        row["correct"] += int(held)
        row["countered"] += int(countered)
        row["effect_applied"] += int(held and not countered)

    def _trade(self, label: str, card_id: int, won: bool) -> None:
        row = self.stack_trade[(label, cards.card_name(card_id))]
        row["n"] += 1
        row["won" if won else "lost"] += 1

    def _hand_card(self, p: Ply, won: bool) -> None:
        """Sentry's swap resolves by STACK INDEX, so read the actual cards off the replayed state: the hand
        card going IN (``action.card``) and the stack card coming OUT (``stack[step.picked]``). Princess-give
        and Judge-queue also use ABILITY_HAND_CARD but are not stack swaps -- skip them."""
        if cards.card_ability(p.step.source) != Ability.SENTRY or p.action.card is None:
            return
        self._trade("Sentry in (hand->stack)", p.action.card, won)          # what it puts onto the stack
        self._trade("Sentry out (stack->hand)", p.state.stack[p.step.picked].card, won)   # what it takes

    def _fool_and_assassin(self, g: Game) -> None:
        """The full life of a card, from the deal to the result. 'Was the Fool worth keeping?' needs the
        counterfactual arms -- hidden, discarded, kept-but-never-played -- not just the games it was played."""
        fool = cards.card_ids_for_name("Fool")[0]
        assassin = cards.card_ids_for_name("Assassin")[0]
        for seat in (0, 1):
            hand = g.dealt.hands[seat]
            won = "won" if g.won(seat) else "lost"
            for cid, bucket in ((fool, self.fool), (assassin, self.assassin)):
                if cid not in hand:
                    continue
                bucket[f"dealt/{won}"] += 1
                if g.setup_choice(seat, StepKind.SETUP_HIDE) == cid:
                    bucket[f"hidden/{won}"] += 1
                    continue
                if g.setup_choice(seat, StepKind.SETUP_DISCARD) == cid:
                    bucket[f"discarded/{won}"] += 1
                    continue
                used = [p for p in g.plies if p.seat == seat and (
                    (p.action.kind == ActionKind.PLAY_CARD and p.action.card == cid)
                    or (cid == assassin and p.action.kind == ActionKind.REVEAL_ASSASSIN))]
                if not used:
                    bucket[f"kept_unused/{won}"] += 1
                    continue
                bucket[f"played/{won}"] += 1
                if cid != fool:
                    continue
                # The Fool's ability is a MAY: it can be played as a bare 1 (no ability), used and countered
                # by a King's Hand, or used and landed. Only the last is "successfully played".
                i = g.plies.index(used[0])
                may = g.plies[i + 1] if i + 1 < len(g.plies) else None
                if may is None or may.phase != StepKind.ABILITY_MAY:
                    bucket[f"played_bare/{won}"] += 1
                elif may.action.kind != ActionKind.CHOOSE_STACK_TARGET:
                    bucket[f"played_bare/{won}"] += 1        # declined the ability
                else:
                    nxt = g.plies[i + 2] if i + 2 < len(g.plies) else None
                    countered = (nxt is not None and nxt.phase == StepKind.REACTION_KINGSHAND
                                 and nxt.action.kind == ActionKind.REVEAL_KINGSHAND)
                    bucket[f"{'countered' if countered else 'landed'}/{won}"] += 1
        if win_reason(g) == "assassin_kills_king":
            self.assassin["won_the_game_outright"] += 1


# --------------------------------------------------------------------------------------------------------
# reporting


def _rows_turn(s: Stats) -> List[dict]:
    out = []
    for (agent, first), (n, w) in sorted(s.turn.items()):
        p, lo, hi = wilson(w, n)
        out.append({"corpus": s.name, "agent": agent, "went": "first" if first else "second",
                    "games": n, "wins": w, "winrate": round(p, 4),
                    "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4)})
    return out


def _rows_setup(s: Stats, tbl, kind: str) -> List[dict]:
    """Per card, split by whether that seat moves FIRST or SECOND.

    The two seats are playing different games -- P1 must win from a ~41% base, P2 defends a ~59% one -- so
    "what does the bot hide" is really two policies. `p1_minus_p2` is the whole point of the table: near
    zero means the choice is seat-independent, and a large value means the bot's setup plan changes with
    the tempo it was dealt."""
    out = []
    for name, d in tbl.items():
        held, hf, hs = d["held"], d["held_first"], d["held_second"]
        rf = d["played_first"] / hf if hf else None
        rs = d["played_second"] / hs if hs else None
        # Two-proportion z. A P1/P2 gap is only interesting if it clears its own noise: at n~900 per seat
        # the standard error is ~2.3pp, so an eyeballed "5pp difference" is often nothing at all.
        z = None
        if rf is not None and rs is not None and hf and hs:
            se = math.sqrt(rf * (1 - rf) / hf + rs * (1 - rs) / hs)
            z = (rf - rs) / se if se > 0 else None
        out.append({"corpus": s.name, "kind": kind, "card": name, "held": held, "z": round(z, 2) if z else None,
                    "played": d["played"], "rate_given_held": round(d["played"] / held, 4) if held else None,
                    "preferred": d["preferred"],
                    "preferred_rate_given_held": round(d["preferred"] / held, 4) if held else None,
                    "held_first": hf, "played_first": d["played_first"],
                    "rate_first": round(rf, 4) if rf is not None else None,
                    "held_second": hs, "played_second": d["played_second"],
                    "rate_second": round(rs, 4) if rs is not None else None,
                    "p1_minus_p2": round(rf - rs, 4) if (rf is not None and rs is not None) else None,
                    "preferred_rate_first": round(d["preferred_first"] / hf, 4) if hf else None,
                    "preferred_rate_second": round(d["preferred_second"] / hs, 4) if hs else None})
    return sorted(out, key=lambda r: -(r["rate_given_held"] or 0))


def _rows_combo(s: Stats) -> List[dict]:
    """Every combo scored against the baseline OF ITS OWN SEAT.

    50% is the wrong null once turn order is in play: a P1 hand-row starts from ~41% and a P2 one from ~59%.
    A combo that wins 55% looks good pooled, but for P2 that is a LOSS of four points against his own base.
    So `lift_first` / `lift_second` are the honest numbers, and `interaction` (lift_first - lift_second) is
    what answers "is there a hand that specifically rescues the player who moves first".
    """
    bf = sum(n for (_, f), (n, _) in s.turn.items() if f)          # all first-moving hand-rows
    wf = sum(w for (_, f), (_, w) in s.turn.items() if f)
    bs = sum(n for (_, f), (n, _) in s.turn.items() if not f)
    ws = sum(w for (_, f), (_, w) in s.turn.items() if not f)
    base_f = wf / bf if bf else 0.5                                # ~0.412
    base_s = ws / bs if bs else 0.5                                # ~0.588

    out = []
    for key, (n, w, nf, wfc) in s.combo.items():
        p, lo, hi = wilson(w, n)
        ns, wsc = n - nf, w - wfc
        pf, flo, fhi = wilson(wfc, nf)
        ps, slo, shi = wilson(wsc, ns)
        lift_f = pf - base_f if nf else None
        lift_s = ps - base_s if ns else None
        out.append({"corpus": s.name, "combo": "+".join(key), "size": len(key), "hands": n, "wins": w,
                    "winrate": round(p, 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
                    "first_n": nf, "first_wr": round(pf, 4) if nf else None,
                    "first_lo": round(flo, 4) if nf else None, "first_hi": round(fhi, 4) if nf else None,
                    "lift_first": round(lift_f, 4) if lift_f is not None else None,
                    "second_n": ns, "second_wr": round(ps, 4) if ns else None,
                    "second_lo": round(slo, 4) if ns else None, "second_hi": round(shi, 4) if ns else None,
                    "lift_second": round(lift_s, 4) if lift_s is not None else None,
                    "interaction": (round(lift_f - lift_s, 4)
                                    if (lift_f is not None and lift_s is not None) else None)})
    return sorted(out, key=lambda r: (-r["winrate"], -r["hands"]))


def _rows_counter(s: Stats, tbl: Dict[str, Counter], key_name: str) -> List[dict]:
    out = []
    for k, c in tbl.items():
        row = {"corpus": s.name, key_name: k}
        row.update({kk: c[kk] for kk in sorted(c)})
        out.append(row)
    return out


def write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))     # union, first-seen order
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)


def tables(s: Stats) -> Dict[str, List[dict]]:
    guess_rows = [{"corpus": s.name, "source": src, "guessed": nm, **dict(c)}
                  for (src, nm), c in s.guess.items()]
    trade_rows = []
    for (label, nm), c in s.stack_trade.items():
        n, won = c["n"], c.get("won", 0)
        p, lo, hi = wilson(won, n)                    # winrate OF THE ACTOR when they made this trade
        trade_rows.append({"corpus": s.name, "trade": label, "card": nm, "n": n, "won": won,
                           "winrate": round(won / n, 4) if n else 0.0,
                           "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4)})
    trade_rows.sort(key=lambda r: (r["trade"], -r["n"]))
    return {
        "sentry_fool_trades": trade_rows,
        "turn_order": _rows_turn(s),
        "setup_hide": _rows_setup(s, s.hide, "hide"),
        "setup_discard": _rows_setup(s, s.discard, "discard"),
        "hand_combos": _rows_combo(s),
        "ability_usage": _rows_counter(s, s.may, "card"),
        "passives": _rows_counter(s, s.passive, "card"),
        "events_kingshand": [{"corpus": s.name, "countered": k, "n": v} for k, v in s.kh.most_common()],
        "events_guess": sorted(guess_rows, key=lambda r: -r["n"]),
        "events_fool": [{"corpus": s.name, "bucket": k, "n": v} for k, v in sorted(s.fool.items())],
        "events_assassin": [{"corpus": s.name, "bucket": k, "n": v} for k, v in sorted(s.assassin.items())],
        "win_reasons": [{"corpus": s.name, "reason": k, "n": v,
                         "share": round(v / max(1, s.games), 4)} for k, v in s.reasons.most_common()],
        "last_ply": [{"corpus": s.name, "phase_action": k, "n": v} for k, v in s.last_pair.most_common()],
    }


def print_report(s: Stats) -> None:
    W = "=" * 96
    print(f"\n{W}\n{s.name}   ({s.games} games, {s.games * 2} hand-rows)\n{W}")

    print("\nTURN ORDER  (a hand-row is one seat of one game)")
    print(f"  {'agent':>22} {'went':>7} {'games':>6} {'winrate':>8}  95% CI")
    for r in _rows_turn(s):
        print(f"  {r['agent']:>22} {r['went']:>7} {r['games']:>6} {r['winrate'] * 100:>7.1f}%  "
              f"[{r['ci95_lo'] * 100:.1f}, {r['ci95_hi'] * 100:.1f}]")

    if s.asymmetric:
        print("\n  !! ASYMMETRIC CORPUS: the two seats are DIFFERENT BOTS, so the setup/ability/passive "
              "tables below\n     pool two policies into one. The P1/P2 split here is confounded with "
              "'which agent'. Read the\n     setup policy off a SELF-PLAY corpus; only 'preferred' is "
              "single-agent (the opponent is unlabelled).")

    for kind, tbl in (("HIDE", s.hide), ("DISCARD", s.discard)):
        print(f"\nSETUP {kind}   P(x | held), split by seat. played = tau-sampled (temp_plies 6); "
              f"preferred = argmax visit share.\n  P1-P2 near zero => the choice does not depend on the "
              f"tempo you were dealt; |z| > 1.96 is the 95% bar.")
        print(f"  {'card':>12} {'held':>6} {'P(x|held)':>10} {'as P1':>7} {'as P2':>7} {'P1-P2':>8} {'z':>6}  "
              f"{'pref P1':>8} {'pref P2':>8}")
        for r in _rows_setup(s, tbl, kind.lower())[:14]:
            f = lambda v, w=7: f"{v * 100:>{w}.1f}%" if v is not None else " " * (w + 1)
            d, z = r["p1_minus_p2"], r["z"]
            # |z| > 1.96 is the 95% bar. Without it a 5pp gap reads as a strategy when it is often noise.
            flag = "  <<< seat-dependent" if z is not None and abs(z) > 1.96 else ""
            print(f"  {r['card']:>12} {r['held']:>6} {f(r['rate_given_held'], 9)} {f(r['rate_first'])} "
                  f"{f(r['rate_second'])} {(f'{d * 100:>+6.1f}pp' if d is not None else ''):>8} "
                  f"{(f'{z:>+5.1f}' if z is not None else ''):>6}  "
                  f"{f(r['preferred_rate_first'], 7)} {f(r['preferred_rate_second'], 7)}{flag}")

    print("\nMAY ABILITIES   window_opened is the denominator, NOT times played "
          "(Princess' window is skipped when either hand is empty)")
    print(f"  {'card':>12} {'window':>7} {'declined':>9} {'activated':>10} {'landed':>7} {'countered':>10} "
          f"{'win% |used':>11} {'win% |declined':>15}")
    for r in _rows_counter(s, s.may, "card"):
        w, act = r.get("window_opened", 0), r.get("activated", 0)
        lw, ll = r.get("landed_won", 0), r.get("landed_lost", 0)
        dw, dl = r.get("declined_won", 0), r.get("declined_lost", 0)
        uwr = f"{lw / (lw + ll) * 100:>10.1f}%" if lw + ll else " " * 11
        dwr = f"{dw / (dw + dl) * 100:>14.1f}%" if dw + dl else " " * 15
        print(f"  {r['card']:>12} {w:>7} {r.get('declined', 0):>9} {act:>10} {r.get('landed', 0):>7} "
              f"{r.get('countered', 0):>10} {uwr} {dwr}")

    print("\nPASSIVE OVERRIDES   'fired' = the power actually changed what was legal/what it landed at")
    print(f"  {'card':>12} {'played':>7} {'fired':>6} {'P(fire)':>8} {'muted':>6} {'win% |fired':>12} "
          f"{'win% all':>9}")
    for r in _rows_counter(s, s.passive, "card"):
        pl, fi = r.get("played", 0), r.get("fired", 0)
        fw, fl = r.get("fired_won", 0), r.get("fired_lost", 0)
        won, lost = r.get("won", 0), r.get("lost", 0)
        fwr = f"{fw / (fw + fl) * 100:>11.1f}%" if fw + fl else " " * 12
        awr = f"{won / (won + lost) * 100:>8.1f}%" if won + lost else " " * 9
        print(f"  {r['card']:>12} {pl:>7} {fi:>6} {fi / pl * 100:>7.1f}% {r.get('muted_at_play', 0):>6} "
              f"{fwr} {awr}")

    tot = sum(s.kh.values())
    print(f"\nKING'S HAND   {tot} reveals; what it countered:")
    print("   " + "  ".join(f"{k}:{v}({v / tot:.0%})" for k, v in s.kh.most_common()) if tot else "   (none)")
    if s.mystic_number:
        mt = sum(s.mystic_number.values())
        print(f"\nMYSTIC   muted base value ({mt} activations): "
              + "  ".join(f"{k}:{v}({v / mt:.0%})" for k, v in sorted(s.mystic_number.items())))

    gn = sum(c["n"] for c in s.guess.values())
    gc = sum(c["correct"] for c in s.guess.values())
    ge = sum(c["effect_applied"] for c in s.guess.values())
    if gn:
        print(f"\nGUESSES   {gn} made | {gc} landed ({gc / gn:.1%}) | {ge} took effect ({ge / gn:.1%})")
        print("   NB a guaranteed MISS is a legitimate play (a landed guess opens a King's-Hand window "
              "that can\n   discard your own Soldier/Judge from the throne) -- this is not a belief-quality "
              "score.")

    for label, bucket, stages in (
            ("FOOL", s.fool, ("dealt", "hidden", "discarded", "played", "played_bare", "landed",
                              "countered", "kept_unused")),
            ("ASSASSIN", s.assassin, ("dealt", "hidden", "discarded", "played", "kept_unused"))):
        print(f"\n{label} LIFECYCLE   (of the hands DEALT it; 'landed' = ability used and NOT countered)")
        by = defaultdict(dict)
        for k, v in bucket.items():
            if "/" in k:
                stage, res = k.split("/")
                by[stage][res] = v
        for stage in stages:
            d = by.get(stage)
            if not d:
                continue
            w, l = d.get("won", 0), d.get("lost", 0)
            if w + l:
                print(f"  {stage:>13} {w + l:>6}   won {w:>4}  lost {l:>4}   win {w / (w + l) * 100:>5.1f}%")
        if bucket.get("won_the_game_outright"):
            print(f"  {'-> outright kills':>13} {bucket['won_the_game_outright']:>6}")

    print("\nHOW GAMES END")
    for k, v in s.reasons.most_common():
        print(f"  {k:>28} {v:>6} ({v / s.games:.1%})")

    combos = _rows_combo(s)
    named = [("Queen", "Princess"), ("Elder", "Princess", "Queen"),
             ("Elder", "Elder", "Princess", "Queen")]
    bf = sum(n for (_, f), (n, _) in s.turn.items() if f)
    wf = sum(w for (_, f), (_, w) in s.turn.items() if f)
    base_f = wf / bf if bf else 0.5
    base_s = 1 - base_f
    floor = max(20, s.games // 10)

    print(f"\nHAND COMBOS   {len(combos)} distinct up to size {s.max_combo}. Pooled baseline is 50.0% by "
          f"construction,\n  but 50% is the WRONG null once turn order is in play -- P1 starts from "
          f"{base_f * 100:.1f}% and P2 from {base_s * 100:.1f}%.\n  'lift' is each seat's win-rate MINUS ITS "
          f"OWN baseline; 'int' (interaction) = lift_P1 - lift_P2, i.e. does this\n  hand help the player "
          f"who moves first MORE than it helps the one who moves second?")
    hdr = (f"  {'combo':>34} {'hands':>6} {'pooled':>7} | {'as P1':>6} {'lift':>7} | {'as P2':>6} "
           f"{'lift':>7} | {'int':>7}")

    def line(r, tag=""):
        lf, ls, it = r["lift_first"], r["lift_second"], r["interaction"]
        g = lambda v: f"{v * 100:>+6.1f}pp" if v is not None else " " * 8
        print(f"  {r['combo']:>34} {r['hands']:>6} {r['winrate'] * 100:>6.1f}% | "
              f"{(r['first_wr'] or 0) * 100:>5.1f}% {g(lf)} | {(r['second_wr'] or 0) * 100:>5.1f}% {g(ls)} "
              f"| {g(it)}{tag}")

    print(hdr)
    for key in named:
        r = next((x for x in combos if x["combo"] == "+".join(sorted(key))), None)
        if r:
            line(r, "  <- asked for")
    solid = [c for c in combos if c["hands"] >= floor]
    for label, rows in (("strongest", solid[:6]), ("weakest", solid[-6:])):
        print(f"  --- {label} (n >= {floor}) ---")
        for r in rows:
            line(r)

    # THE question the split exists to answer: is any hand SEAT-SPECIFIC, or does every hand just add a
    # constant on top of whatever the tempo already gave you?
    #
    # With ~850 well-powered combos, the LARGEST interaction is guaranteed to look impressive by chance --
    # so never read the extremes without this test first. Compare the spread of the observed interactions
    # against the spread that pure sampling noise would produce under the null "a combo adds the same lift
    # to both seats". If they match, the extremes are the tail of N draws and mean nothing.
    inter = [c for c in solid if c["interaction"] is not None and c["first_n"] and c["second_n"]]
    if len(inter) > 30:
        xs = [c["interaction"] for c in inter]
        mean = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))
        noise = [math.sqrt(c["first_wr"] * (1 - c["first_wr"]) / c["first_n"]
                           + c["second_wr"] * (1 - c["second_wr"]) / c["second_n"]) for c in inter]
        exp_sd = sum(noise) / len(noise)
        z = [abs(x / e) for x, e in zip(xs, noise) if e > 0]
        big = sum(1 for v in z if v > 1.96)
        ratio = sd / exp_sd if exp_sd else float("nan")
        print(f"\n  IS ANY HAND SEAT-SPECIFIC?   ({len(inter)} combos with n >= {floor})")
        print(f"    spread of the observed P1-P2 interaction : {sd * 100:.2f}pp")
        print(f"    spread expected from sampling noise ALONE: {exp_sd * 100:.2f}pp   (ratio {ratio:.2f})")
        print(f"    combos with |z| > 1.96: {big}/{len(inter)} = {big / len(inter):.1%}  "
              f"(5.0% expected by chance)")
        if ratio < 1.15 and big / len(inter) < 0.08:
            print("    => NO. Hand strength and turn order are ADDITIVE: every combo lifts both seats by the\n"
                  "       same amount, and the biggest 'interactions' are just the tail of "
                  f"{len(inter)} draws. Do not\n       read them as signal.")
        else:
            print("    => the interactions exceed noise -- there IS a seat-dependent hand effect; inspect "
                  "the extremes.")
            for r in sorted(inter, key=lambda r: -abs(r["interaction"]))[:6]:
                line(r)


def plots(s: Stats, out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  (matplotlib absent -- skipping plots)")
        return

    import numpy as np
    P1, P2 = "#4c72b0", "#dd8452"                  # one colour per seat, used consistently across all panels
    bf = sum(n for (_, f), (n, _) in s.turn.items() if f)
    wfc = sum(w for (_, f), (_, w) in s.turn.items() if f)
    base_f = (wfc / bf if bf else 0.5) * 100
    base_s = 100 - base_f

    fig, axes = plt.subplots(2, 2, figsize=(17, 10))

    # 1. turn order
    ax = axes[0][0]
    rows = _rows_turn(s)
    labels = [f"{r['agent']}\n{r['went']}" for r in rows]
    vals = [r["winrate"] * 100 for r in rows]
    err = [[(r["winrate"] - r["ci95_lo"]) * 100 for r in rows],
           [(r["ci95_hi"] - r["winrate"]) * 100 for r in rows]]
    ax.bar(labels, vals, yerr=err, capsize=4,
           color=[P1 if r["went"] == "first" else P2 for r in rows])
    ax.axhline(50, ls="--", c="k", lw=1)
    ax.set_ylabel("win %"); ax.set_title("Turn order"); ax.tick_params(labelsize=7)

    # 2 + 3. setup policy, P1 vs P2 side by side. If the pair is level, the seat does not change the plan.
    # ALL 14 cards, never a top-N slice: truncating hid the single most informative row in the table (the
    # Assassin, which the search argmaxes 0/4000 times -- it is a REACTION and hiding it disarms it). A card
    # the bot refuses to touch is a finding, and it sorts to the BOTTOM.
    for ax, tbl, kind in ((axes[0][1], s.hide, "hide"), (axes[1][0], s.discard, "discard")):
        rs = _rows_setup(s, tbl, kind)[::-1]
        y = np.arange(len(rs))
        ax.barh(y + 0.20, [(r["rate_first"] or 0) * 100 for r in rs], height=0.38, color=P1,
                label="moves 1st (P1)")
        ax.barh(y - 0.20, [(r["rate_second"] or 0) * 100 for r in rs], height=0.38, color=P2,
                label="moves 2nd (P2)")
        # A hollow marker for what the SEARCH preferred (argmax visit share) vs what temperature actually
        # played -- the gap between them is the difference between the policy and a sample from it.
        ax.scatter([(r["preferred_rate_given_held"] or 0) * 100 for r in rs], y, s=18, facecolors="none",
                   edgecolors="k", linewidths=0.8, zorder=3, label="preferred (argmax)")
        ax.set_yticks(y); ax.set_yticklabels([r["card"] for r in rs], fontsize=8)
        ax.set_xlabel(f"P({kind} | held)  %"); ax.set_title(f"Setup {kind} policy, by seat (all 14 cards)")
        ax.legend(fontsize=7)

    # 4. combos: each seat against ITS OWN baseline, so bar length is comparable across the two colours
    ax = axes[1][1]
    floor = max(20, s.games // 10)
    combos = [c for c in _rows_combo(s) if c["hands"] >= floor]
    top = (combos[:6] + combos[-6:])[::-1]
    y = np.arange(len(top))
    ax.barh(y + 0.20, [(c["first_wr"] or 0) * 100 for c in top], height=0.38, color=P1, label="as P1")
    ax.barh(y - 0.20, [(c["second_wr"] or 0) * 100 for c in top], height=0.38, color=P2, label="as P2")
    ax.axvline(base_f, ls="--", c=P1, lw=1.2)      # each seat's OWN null -- 50% is the wrong line here
    ax.axvline(base_s, ls="--", c=P2, lw=1.2)
    ax.set_yticks(y); ax.set_yticklabels([c["combo"] for c in top], fontsize=6)
    ax.set_xlabel("win %   (dashed = that seat's baseline: "
                  f"P1 {base_f:.1f}%, P2 {base_s:.1f}%)")
    ax.set_title(f"Hand combos by seat (n >= {floor})")
    ax.legend(fontsize=7)

    fig.suptitle(f"{s.name}   ({s.games} games, {s.games * 2} hand-rows)")
    fig.tight_layout()
    p = os.path.join(out_dir, "overview.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  wrote {p}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Corpus statistics: turn order, setup policy, hands, powers.")
    p.add_argument("--data", nargs="+", default=[os.path.join("datasets", "selfplay_gen1_v3c")],
                   help="one or more corpus dirs; reported SIDE BY SIDE unless --pool")
    p.add_argument("--out-dir", default=os.path.join("results", "corpus_stats"))
    p.add_argument("--max-combo", type=int, default=4, help="largest hand sub-combination to score")
    p.add_argument("--limit", type=int, default=None, help="first N games per corpus (smoke test)")
    p.add_argument("--pool", action="store_true",
                   help="fold every corpus into ONE population. Sound for turn-order across symmetric "
                        "self-play; NOT sound across different bots, search budgets, or the asymmetric corpus")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    groups = ([("+".join(os.path.basename(d.rstrip("/\\")) for d in args.data), args.data)] if args.pool
              else [(os.path.basename(d.rstrip("/\\")), [d]) for d in args.data])

    everything: Dict[str, dict] = {}
    for name, dirs in groups:
        print(f"\nreplaying {name} ...")
        s = Stats(name, args.max_combo)
        for g in replay_all(dirs, args.limit):
            s.add(g)
        if not s.games:
            print(f"  no games found in {dirs}")
            continue
        print_report(s)

        out = os.path.join(args.out_dir, name)
        os.makedirs(out, exist_ok=True)
        t = tables(s)
        for fname, rows in t.items():
            write_csv(os.path.join(out, f"{fname}.csv"), rows)
        everything[name] = {"games": s.games, "hand_rows": s.games * 2,
                            "max_combo": args.max_combo, "tables": t}
        if not args.no_plot:
            plots(s, out)
        print(f"\n  wrote {len(t)} CSVs to {out}")

    if everything:
        js = os.path.join(args.out_dir, "stats.json")
        os.makedirs(args.out_dir, exist_ok=True)
        # MERGE, don't clobber. This file is the collation across every corpus, but corpora are analysed one
        # command at a time -- a plain overwrite silently drops the groups from the previous runs, leaving a
        # "collated" document containing only whatever was analysed last.
        merged: Dict[str, dict] = {}
        if os.path.exists(js):
            try:
                with open(js, encoding="utf-8") as fh:
                    merged = json.load(fh)
            except (OSError, ValueError):
                pass                                   # unreadable/corrupt -> start clean rather than die
        kept = [k for k in merged if k not in everything]
        merged.update(everything)
        with open(js, "w", encoding="utf-8") as fh:
            json.dump(merged, fh, indent=1)
        print(f"  wrote {js}   (the D3 feed: {len(merged)} group(s) -- "
              f"{len(everything)} from this run, {len(kept)} carried over)")


if __name__ == "__main__":
    main()
