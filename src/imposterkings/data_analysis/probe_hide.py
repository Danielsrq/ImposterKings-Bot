"""Does training on hidden-Assassin games make the SEARCH start hiding the Assassin?

    python -m imposterkings.data_analysis.probe_hide \\
        --ckpt models/v3c_gen2_d96/attn_d96_L2.pt models/v3c_gen25/attn_d96_L2.pt --deals 150

The bot picks "hide the Assassin" as its argmax 0 times in 4,000 setup decisions, yet a forced-hide
intervention (2,000 games) says the move is at worst neutral and plausibly worth ~3pp. A direct probe showed
the move IS searched (3-11% of visits) but always ranked 2nd-5th, and the margin is inside MCTS's own noise
(a 0.024 q-gap against a 0.077 label-noise std) -- the signature of a search BIAS, not variance. Likely
cause: strategy fusion. Hiding is an INFORMATIONAL move (it protects the Assassin from an Inquisitor that
must guess), and SO-ISMCTS determinizes the world at the root -- destroying exactly the uncertainty that
makes hiding worth it.

The hypothesis: the NN can learn from OUTCOMES what the search cannot see from determinized rollouts. Train
on games where the Assassin was actually hidden, and the net's value of those states stops being an
extrapolation -> `_leaf_value` returns better numbers inside the hide branch -> the backed-up q rises -> the
gap flips -> the search starts choosing a move it could never find on its own.

This probe measures that, PAIRED: the SAME setup positions and the SAME search seed for every net, so the
only thing that differs is the evaluator. Per net, at each position where a seat holds the Assassin:

  prior_q     the net's RAW q for hide-Assassin at the setup node        (the PRIOR channel)
  V_hidden    the net's value of the state AFTER hiding, max_a q(s',a)    (the VALUE channel -- should move)
  share       hide-Assassin's visit share after a full k20-l3 search     (did PUCT explore it more?)
  q_backed    hide-Assassin's backed-up mean_q, and the GAP to the best  (did the 0.024 deficit close?)
  flipped     is hide-Assassin now the argmax (most-visited) move?        (THE HEADLINE)
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import numpy as np

from .. import cards
from ..actions import Action, ActionKind, StepKind
from ..agents import MCTSAgent
from ..budget import make_budget
from ..state import GameState
from ..machine_learning.agent import NNPolicy
from ..machine_learning.benchmark import _evaluator_for
from .forced_hide import ASSASSIN, assassin_seat


def _setup_hide_node(seed: int, seat: int) -> GameState:
    """Advance a deal to the point where ``seat`` (which holds the Assassin) chooses its hidden card."""
    st = GameState.deal(np.random.default_rng(seed))
    while not (st.to_play == seat and st.phase == StepKind.SETUP_HIDE):
        st = st.apply(st.legal_moves()[0])
    return st


def probe_one(st: GameState, seat: int, agent: MCTSAgent, policy: NNPolicy, evaluator,
              rng_seed: int) -> Optional[Dict]:
    """One setup position. Returns None if hiding the Assassin is not even legal here."""
    hide = Action(ActionKind.HIDE_CARD, card=ASSASSIN)
    view = st.information_set(seat)
    if hide not in view.legal_moves():
        return None

    # PRIOR channel: the net's raw action-value q for each legal setup move (no search).
    q_by_move = dict(policy.evaluate(view))
    prior_q = q_by_move[hide]
    best_prior = max(q_by_move.values())

    # VALUE channel: the net's value of the state AFTER hiding the Assassin (max_a q(s',a)); this is the
    # channel the forced corpus actually feeds -- the downstream hidden-Assassin positions.
    v_hidden = evaluator(st.apply(hide))[0][seat]

    # SEARCH: a full k20-l3 PUCT search, same seed for every net so only the evaluator differs.
    agent.select_move(view, np.random.default_rng(rng_seed))
    stats = sorted(agent.last_result.stats, key=lambda s: -s.visits)
    asn = next((s for s in stats if s.move == hide), None)
    best = stats[0]
    return {
        "prior_q": prior_q, "prior_gap": prior_q - best_prior,
        "v_hidden": v_hidden,
        "share": asn.visit_share if asn else 0.0,
        "q_backed": asn.mean_q if asn else float("nan"),
        "q_gap": (asn.mean_q - best.mean_q) if asn else float("nan"),
        "flipped": best.move == hide,               # is hiding the Assassin the argmax move?
        "best_card": cards.card_name(best.move.card) if best.move.card is not None else best.move.kind.name,
    }


def run(ckpts: List[str], deals: int, base_seed: int, spec_k: int, spec_l: int,
        search_seed: int) -> Dict[str, List[Dict]]:
    seeds, s = [], base_seed
    while len(seeds) < deals:
        if assassin_seat(s) is not None:
            seeds.append(s)
        s += 1
    positions = [(seed, assassin_seat(seed)) for seed in seeds]
    print(f"probing {len(positions)} setup positions (deals where a seat holds the Assassin), "
          f"k{spec_k}-l{spec_l}, same search seed {search_seed} for every net")

    out: Dict[str, List[Dict]] = {}
    for ckpt in ckpts:
        ev = _evaluator_for(ckpt)
        agent = MCTSAgent(budget=make_budget("hybrid", k=spec_k, l=spec_l), evaluator=ev)
        policy = NNPolicy.from_checkpoint(ckpt)
        rows = []
        for seed, seat in positions:
            r = probe_one(_setup_hide_node(seed, seat), seat, agent, policy, ev, search_seed)
            if r is not None:
                rows.append(r)
        out[ckpt] = rows
        print(f"  {os.path.basename(os.path.dirname(ckpt)) or ckpt}: {len(rows)} positions")
    return out


def report(results: Dict[str, List[Dict]]) -> None:
    def mean(rows, k):
        vs = [r[k] for r in rows if not np.isnan(r[k])]
        return float(np.mean(vs)) if vs else float("nan")

    print(f"\n{'=' * 100}")
    print("DOES THE NN TEACH THE SEARCH TO HIDE THE ASSASSIN?   (paired: same positions + search seed)")
    print("=" * 100)
    hdr = (f"  {'net':>26} {'n':>4} | {'prior_q':>8} {'V_hidden':>9} | {'share':>7} {'q_backed':>9} "
           f"{'q_gap':>7} | {'FLIP%':>6}")
    print(hdr)
    names = list(results)
    for ckpt in names:
        rows = results[ckpt]
        if not rows:
            continue
        name = os.path.basename(os.path.dirname(ckpt)) or os.path.basename(ckpt)
        flip = np.mean([r["flipped"] for r in rows]) * 100
        print(f"  {name:>26} {len(rows):>4} | {mean(rows, 'prior_q'):>+8.3f} {mean(rows, 'v_hidden'):>+9.3f} "
              f"| {mean(rows, 'share') * 100:>6.1f}% {mean(rows, 'q_backed'):>+9.3f} "
              f"{mean(rows, 'q_gap'):>+7.3f} | {flip:>5.1f}%")

    if len(names) >= 2:
        a, b = results[names[0]], results[names[1]]
        # pair positions present in BOTH (same seeds/order) for an honest delta
        n = min(len(a), len(b))
        print(f"\n  DELTA ({os.path.basename(os.path.dirname(names[1]))} - "
              f"{os.path.basename(os.path.dirname(names[0]))}), over {n} shared positions:")
        for k, lbl, chan in (("prior_q", "prior q(hide)", "prior"), ("v_hidden", "V(hidden state)", "VALUE"),
                             ("share", "visit share", "search"), ("q_gap", "gap to best", "search")):
            da = np.nanmean([b[i][k] - a[i][k] for i in range(n)])
            print(f"    {lbl:>18} [{chan:>6}] : {da:+.4f}")
        fa = np.mean([a[i]["flipped"] for i in range(n)]) * 100
        fb = np.mean([b[i]["flipped"] for i in range(n)]) * 100
        print(f"    {'argmax flip rate':>18} [behav ] : {fa:.1f}% -> {fb:.1f}%   "
              f"({'THE SEARCH NOW HIDES IT' if fb > fa + 1 else 'no change'})")
        print("\n  Prediction was: prior barely moves, V_hidden RISES (that is the channel the corpus feeds),\n"
              "  and IF the mechanism is real the gap closes and the flip rate goes up. If the prior moved but\n"
              "  V did not, the causal story is backwards.")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Probe whether training on hidden-Assassin games flips the search.")
    p.add_argument("--ckpt", nargs="+", required=True, help="two+ checkpoints; first is the BEFORE baseline")
    p.add_argument("--deals", type=int, default=150)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--l", type=int, default=3)
    p.add_argument("--search-seed", type=int, default=0)
    args = p.parse_args(argv)
    report(run(args.ckpt, args.deals, args.base_seed, args.k, args.l, args.search_seed))


if __name__ == "__main__":
    main()
