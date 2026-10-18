"""Export LF (late-fusion) self-play cross-attention traces for the browser viewer.

    python -m imposterkings.machine_learning.lf_attention_export --lf models/study11/LF-B.pt --games 10 \
        --out lf_attention.json

Plays LF@k20 against itself. Every decision of BOTH seats is recorded from that seat's own information set
(what it can see: its hand, the court, the unseen pool with its zone posterior), with each legal action's
q / policy / search visit share and the cross attention of that action's candidate token over the 23 state
tokens (per head), plus the search's root value for the eval graph. Each seat also gets a final board at
game end and the reason the game ended. The move list is written twice per move: as the mover saw it and
as the other seat saw it (the opponent's setup hide/discard stays hidden).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from typing import Dict, List

import numpy as np

from .. import cards
from ..actions import ActionKind, StepKind
from . import features3 as F3

SEED_BASE = 93_000_000      # outside every corpus and evaluation range


def _card_images(width: int = 150) -> Dict[str, str]:
    from PIL import Image
    from ..paths import asset_dir
    files = {nm: cards.asset_path(cards.card_ids_for_name(nm)[0]) for nm in cards.CARD_NAMES}
    files["King"] = cards.KING_ASSET
    files["Back"] = cards.CARD_BACK_ASSET
    res = {}
    for key, fn in files.items():
        im = Image.open(os.path.join(asset_dir(), fn)).convert("RGB")
        buf = io.BytesIO()
        im.resize((width, int(im.height * width / im.width)), Image.LANCZOS).save(buf, "JPEG", quality=82)
        res[key] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return res


def describe(move, view, own: bool) -> str:
    """Readable move text. ``own`` = as the mover saw it; otherwise the other seat's view (setup hides and
    discards are face-down, so the card is withheld)."""
    k = move.kind
    nm = cards.card_name(move.card) if move.card is not None else None
    if k == ActionKind.HIDE_CARD:
        return f"hid {nm}" if own else "hid a card"
    if k == ActionKind.DISCARD_CARD:
        return f"discarded {nm} face-down" if own else "discarded a card face-down"
    if k == ActionKind.PLAY_CARD:
        if view.pending and view.pending[-1].kind == StepKind.ASCEND:
            return f"{nm} ascended from the antechamber"
        return f"played {nm}"
    if k == ActionKind.FLIP_KING:
        return "flipped their king"
    if k == ActionKind.DECLARE_ABILITY:
        return "used the ability"
    if k == ActionKind.DECLINE_ABILITY:
        return "declined the ability"
    if k == ActionKind.GUESS_CARD:
        return f"guessed {move.name}"
    if k == ActionKind.CHOOSE_NUMBER:
        return f"named value {move.number}"
    if k == ActionKind.CHOOSE_HAND_CARD:
        return f"chose {nm}"
    if k == ActionKind.CHOOSE_STACK_TARGET:
        t = view.stack[move.target].card if 0 <= move.target < len(view.stack) else None
        return f"targeted {cards.card_name(t) if t is not None else '?'} on the court"
    if k == ActionKind.STOP:
        return "stopped"
    if k == ActionKind.REVEAL_KINGSHAND:
        return "revealed King's Hand"
    if k == ActionKind.REVEAL_ASSASSIN:
        return "revealed the Assassin"
    if k == ActionKind.DECLINE_REACTION:
        step = view.pending[-1].kind if view.pending else None
        return {StepKind.REACTION_KINGSHAND: "did not counter with King's Hand",
                StepKind.REACTION_ASSASSIN: "did not reveal an Assassin",
                StepKind.REACTION_KH_VS_ASSASSIN: "did not answer the Assassin with King's Hand"}.get(step, "did not react")
    return k.name.lower()


_IMPERATIVE = [(r"^hid ", "hide "), (r"^discarded ", "discard "), (r"^played ", "play "),
               (r"^(.+) ascended from the antechamber$", r"ascend \1 from the antechamber"),
               (r"^flipped their king$", "flip king"), (r"^used the ability$", "use the ability"),
               (r"^declined the ability$", "decline the ability"), (r"^guessed ", "guess "),
               (r"^named value ", "name value "), (r"^chose ", "choose "), (r"^targeted ", "target "),
               (r"^stopped$", "stop"), (r"^revealed ", "reveal "), (r"^did not reveal an Assassin$", "don't reveal the Assassin"),
               (r"^did not ", "don't ")]


def imperative(text: str) -> str:
    """A move's log text ("flipped their king") as an option to choose ("flip king")."""
    import re
    for pat, rep in _IMPERATIVE:
        if re.search(pat, text):
            return re.sub(pat, rep, text, count=1)
    return text


def _state(view, st: F3.State3) -> dict:
    slot = st.slot_of
    seen = set(slot.values())
    unseen = []
    for i in range(F3.N_CARDS):
        if i not in seen:
            post = st.cards[i, F3._ZONE_OFF:]
            unseen.append({"slot": i, "p": [round(float(post[j]), 3) for j in (3, 4, 5, 11)]})
    court = []
    for sc in view.stack:
        eff = 0 if sc.disgraced else (3 if cards.card_value(sc.card) in view.muted_values else
                                      (sc.value_override or cards.card_value(sc.card)))
        court.append({"slot": slot[sc.card], "disgraced": sc.disgraced, "power": eff})
    o = view.observer
    return {
        "my_hand": [slot[c] for c in view.own_hand],
        "my_hidden": slot.get(view.own_hidden) if view.own_hidden is not None else None,
        "my_setup": slot.get(view.own_setup_discard) if view.own_setup_discard is not None else None,
        "court": court,
        "my_ante": [slot[c] for c in view.antechambers[o]],
        "their_ante": [slot[c] for c in view.antechambers[1 - o]],
        "discard": [slot[c] for c in view.discard],
        "faceup": slot.get(view.leftover_faceup) if view.leftover_faceup is not None and view.leftover_faceup >= 0 else None,
        "unseen": unseen,
        "opp_hand": view.opp_hand_count, "opp_hidden": bool(view.opp_has_hidden),
        "kings": {"mine": bool(view.kings[o]), "theirs": bool(view.kings[1 - o])},
        "muted": sorted(view.muted_values), "lacks": sorted(view.opp_hand_lacks), "has": sorted(view.opp_hand_has),
        "step": view.pending[-1].kind.name if view.pending else "",
        "reactor": view.to_play != view.turn_player,
    }


def _truth(state) -> dict:
    """The full table (observer mode): every card by engine id, wherever it really is."""
    court = []
    for sc in state.stack:
        eff = 0 if sc.disgraced else (3 if cards.card_value(sc.card) in state.muted_values else
                                      (sc.value_override or cards.card_value(sc.card)))
        court.append({"card": sc.card, "disgraced": sc.disgraced, "power": eff})
    return {"hands": [list(h) for h in state.hands], "hidden": list(state.hidden), "setup": list(state.setup_discard),
            "ante": [list(a) for a in state.antechambers], "court": court, "discard": list(state.discard),
            "faceup": state.leftover_faceup, "facedown": state.leftover_facedown, "kings": list(state.kings),
            "muted": sorted(state.muted_values)}


def _slot_engine(st: F3.State3, state) -> List[int]:
    """Engine card id behind each of the mover's 18 card tokens. Visible cards map exactly; an unseen token is
    an unseen copy of its name, so it takes one of that name's cards the mover cannot see (2 unseen copies of a
    name are identical tokens, so which physical copy gets which token does not matter)."""
    out = [-1] * F3.N_CARDS
    for eid, sl in st.slot_of.items():
        out[sl] = eid
    for nm in cards.CARD_NAMES:
        free = sorted(c for c in cards.card_ids_for_name(nm) if c not in st.slot_of)
        for sl in range(F3.N_CARDS):
            if F3.INSTANCE_NAMES[sl] == nm and out[sl] < 0 and free:
                out[sl] = free.pop(0)
    return out


def play_one(lf_path: str, seed: int) -> dict:
    import torch
    torch.set_num_threads(1)
    from ..agents import MCTSAgent
    from ..arena import play_game
    from ..budget import make_budget
    from . import latefusion_model as LF

    model, _ = LF.load(lf_path)
    ev = LF.evaluator_from_model(model)
    agents = [MCTSAgent(budget=make_budget("hybrid", k=20, l=3), evaluator=ev) for _ in range(2)]
    steps: List[dict] = []
    moves: List[dict] = []

    def on_decision(s, view, move, agent, state):
        legal = view.legal_moves()
        st, cd = F3.tokenize3(view, legal)
        with torch.no_grad():
            out = model(LF.collate([st], [cd]))
        attn = out["attn_cross"][0][0].numpy()                 # [H, N, 23]
        q = out["q"][0, :len(legal)].numpy()
        lg = out["logit"][0, :len(legal)].numpy()
        pi = np.exp(lg - lg.max())
        pi /= pi.sum()
        res = agent.last_result
        share = {x.move: x.visit_share for x in res.stats} if res is not None else {}
        cands = [{"label": imperative(describe(m, view, own=True)), "q": round(float(q[i]), 4), "pi": round(float(pi[i]), 4),
                  "visits": round(float(share.get(m, 0.0)), 4), "chosen": m == move,
                  "attn": [[round(float(a), 4) for a in attn[h, i]] for h in range(attn.shape[0])]}
                 for i, m in enumerate(legal)]
        steps.append({"seat": s, "ply": len(moves), "state": _state(view, st), "cands": cands,
                      "truth": _truth(state), "slot_engine": _slot_engine(st, state),
                      "forced": len(legal) == 1,
                      "value": round(float(res.root_value()), 4) if res is not None else None,
                      "net_value": round(float(q.max()), 4)})
        moves.append({"ply": len(moves), "seat": s, "own": describe(move, view, own=True),
                      "other": describe(move, view, own=False), "kind": move.kind.name,
                      "reaction": bool(view.pending) and view.pending[-1].kind in (
                          StepKind.REACTION_KINGSHAND, StepKind.REACTION_ASSASSIN, StepKind.REACTION_KH_VS_ASSASSIN)})

    winner, _, final = play_game(agents, np.random.default_rng(seed), on_decision=on_decision)
    loser = 1 - winner
    if moves and moves[-1]["kind"] == "DECLINE_REACTION" and len(moves) > 1 and moves[-2]["kind"] == "REVEAL_ASSASSIN":
        reason = {"kind": "assassin", "text": "the Assassin struck the flipped king"}
    else:
        reason = {"kind": "stuck", "text": "had no legal move on their turn"}
    finals = {}
    for seat in (0, 1):
        v = final.information_set(seat)
        finals[seat] = _state(v, F3.tokenize_state3(v, legal_moves=[]))
    return {"seed": seed, "starter": final.starting_player, "winner": winner, "loser": loser, "reason": reason,
            "steps": steps, "moves": moves, "finals": finals, "final_truth": _truth(final)}


def main(argv=None) -> None:
    from joblib import Parallel, delayed
    p = argparse.ArgumentParser(description="Export LF self-play cross-attention traces for the viewer.")
    p.add_argument("--lf", default=os.path.join("models", "study11", "LF-B.pt"))
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    games = list(Parallel(n_jobs=a.workers)(delayed(play_one)(a.lf, SEED_BASE + i) for i in range(a.games)))
    tokens = [{"label": lab, "name": nm} for lab, nm in zip(F3.INSTANCE_LABELS, F3.INSTANCE_NAMES)]
    tokens += [{"label": lab, "name": None} for lab in F3.STATE_LABELS[18:]]
    blob = {"tokens": tokens, "heads": 6, "images": _card_images(), "games": games, "lf": a.lf,
            "engine_names": [cards.card_name(c) for c in range(cards.DECK_SIZE)],
            "budget": "hybrid-k20-l3"}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(blob, f, separators=(",", ":"))
    print(f"{len(games)} games, {sum(len(g['steps']) for g in games)} decisions -> {a.out} "
          f"({os.path.getsize(a.out) / 1e6:.1f} MB); first player won {sum(g['winner'] == g['starter'] for g in games)}"
          f"/{len(games)}; endings {[g['reason']['kind'] for g in games]}")


if __name__ == "__main__":
    main()
