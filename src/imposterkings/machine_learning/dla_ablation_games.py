"""Study 10, Phase 4 -- does ablating a head make the GREEDY net die to the Assassin more often? Resumable.

    python -m imposterkings.machine_learning.dla_ablation_games pilot  --ckpt <pt>            # base arm only
    python -m imposterkings.machine_learning.dla_ablation_games run    --ckpt <pt> --arms base,prereg,all-heads
    python -m imposterkings.machine_learning.dla_ablation_games report --ckpt <pt>

Arms: ``base`` (intact vs intact -- greedy play is a deterministic argmax, so ONE game per deal gives both
seats' baselines) and one arm per (layer, head) MEAN-ablated (``L1h2`` etc.). An ablated arm plays the
ablated net in seat 0 and then seat 1 against an intact copy of the same checkpoint, on the same deal
(``benchmark.py``'s mirroring; deal rng = ``default_rng(seed)``, which also draws the starting player, so
the ~60/40 turn-order edge cancels). Every ablated (deal, seat) is therefore PAIRED with the base game.

Primary endpoint (pre-registered in ``prereg.json``): paired change in the focal seat's Assassin-death rate
(it flipped its king, the opponent revealed the Assassin, and it did not counter with King's Hand).

Resumable: each finished chunk appends its rows to ``games/<arm>.jsonl`` (flushed + fsynced); a restart
schedules only the missing (arm, seed) pairs. Arms run in priority order (base -> pre-registered ->
controls); ``--max-minutes`` stops dispatching new chunks and exits cleanly.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..actions import ActionKind, StepKind
from ..cards import card_ids_for_name

ASSASSIN_IDS = set(card_ids_for_name("Assassin"))
PILOT_DEALS = 200


# --- one game ------------------------------------------------------------------------------------

def _seat_stats() -> Dict[str, int]:
    return {"flips": 0, "unsafe_flips": 0, "assassin_death": 0, "kh_counters": 0,
            "opp_reveal_opps": 0, "opp_reveals": 0}


def play_logged(agents, seed: int, ns: int = 0, initial_state=None) -> dict:
    """Play one deal; return per-seat event counts + winner + plies (the Assassin-event logger)."""
    from ..arena import play_game
    st = {0: _seat_stats(), 1: _seat_stats()}
    plies = [0]

    def on_decision(seat, view, move, agent, state):
        plies[0] += 1
        kind = view.pending[-1].kind if view.pending else None
        me, opp = st[seat], st[1 - seat]
        if move.kind == ActionKind.FLIP_KING:
            me["flips"] += 1
            if (ASSASSIN_IDS & set(view.unknown_cards())) and "Assassin" not in view.opp_hand_lacks:
                me["unsafe_flips"] += 1
        if kind == StepKind.REACTION_ASSASSIN:              # the opponent of the flipper decides here
            if any(m.kind == ActionKind.REVEAL_ASSASSIN for m in state.legal_moves()):
                opp["opp_reveal_opps"] += 1                 # counted on the FLIPPER's row ("my opponent could")
                if move.kind == ActionKind.REVEAL_ASSASSIN:
                    opp["opp_reveals"] += 1
        if kind == StepKind.REACTION_KH_VS_ASSASSIN:        # the flipper: counter or die
            if move.kind == ActionKind.REVEAL_KINGSHAND:
                me["kh_counters"] += 1
            else:
                me["assassin_death"] += 1

    winner, _, _ = play_game(agents, np.random.default_rng(seed), on_decision=on_decision,
                             play_rng=np.random.default_rng([seed, ns]), initial_state=initial_state)
    return {"winner": int(winner), "plies": plies[0], "seats": st}


# --- worker --------------------------------------------------------------------------------------

_MODEL_CACHE: Dict[str, object] = {}


def _intact(ckpt: str):
    if ckpt not in _MODEL_CACHE:
        from .attention_model import load
        _MODEL_CACHE[ckpt] = load(ckpt)[0]
    return _MODEL_CACHE[ckpt]


def arm_spec(arm: str):
    """``base`` -> None; ``L{l}h{h}`` -> a mean HeadAblation spec."""
    from .interventions import HeadAblation, InterventionSpec
    if arm == "base":
        return None
    li, h = arm[1:].split("h")
    return InterventionSpec(ablations=(HeadAblation(int(li), int(h), "mean"),))


def _chunk(ckpt: str, means_path: str, arm: str, seeds: List[int]) -> dict:
    import torch
    torch.set_num_threads(1)
    from .agent import NNAgent, NNPolicy
    from .interventions import AblatedModel
    t0 = time.perf_counter()
    model = _intact(ckpt)
    intact = NNAgent(NNPolicy(model), name="intact")
    spec = arm_spec(arm)
    rows = []
    if spec is None:
        for seed in seeds:
            g = play_logged([intact, intact], seed)
            rows.append({"arm": arm, "seed": seed, "games": [
                {"focal": s, "win": int(g["winner"] == s), "plies": g["plies"], **g["seats"][s]} for s in (0, 1)]})
    else:
        means = torch.load(means_path)
        focal = NNAgent(NNPolicy(AblatedModel(model, spec, means)), name=arm)
        for seed in seeds:
            games = []
            for s in (0, 1):
                agents = [intact, intact]
                agents[s] = focal
                g = play_logged(agents, seed, ns=s)
                games.append({"focal": s, "win": int(g["winner"] == s), "plies": g["plies"], **g["seats"][s]})
            rows.append({"arm": arm, "seed": seed, "games": games})
    return {"arm": arm, "rows": rows, "seconds": time.perf_counter() - t0}


# --- resumable driver ----------------------------------------------------------------------------

def read_rows(path: str) -> Dict[int, dict]:
    """seed -> row (last write wins; tolerates a torn final line from a kill)."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[int(r["seed"])] = r
    return out


def append_rows(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torn = False                                    # a kill can leave an unterminated last line: new rows
    if os.path.exists(path) and os.path.getsize(path):  # must start on a fresh line or they'd be glued to it
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            torn = f.read(1) != b"\n"
    with open(path, "a") as f:
        if torn:
            f.write("\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_arms(ckpt: str, out: str, arms: List[str], deals: int, base_seed: int, workers: int, chunk: int,
             max_minutes: Optional[float] = None) -> Dict[str, int]:
    """Play every missing (arm, seed) pair; return rows-per-arm after the run."""
    from joblib import Parallel, delayed
    gdir = os.path.join(out, "games")
    means_path = os.path.join(out, "means.pt")
    seeds = [base_seed + i for i in range(deals)]
    todo: List[Tuple[str, List[int]]] = []
    for arm in arms:
        have = read_rows(os.path.join(gdir, f"{arm}.jsonl"))
        missing = [s for s in seeds if s not in have]
        todo += [(arm, missing[i:i + chunk]) for i in range(0, len(missing), chunk)]
    print(f"{len(todo)} chunks to play across {len(arms)} arm(s) ({deals} deals each)", flush=True)
    t0 = time.perf_counter()
    stopped = [False]

    def jobs() -> Iterator:
        for arm, ch in todo:
            if max_minutes is not None and (time.perf_counter() - t0) / 60 > max_minutes:
                stopped[0] = True
                return
            yield delayed(_chunk)(ckpt, means_path, arm, ch)

    done = 0
    for r in Parallel(n_jobs=workers, return_as="generator_unordered", pre_dispatch="2*n_jobs")(jobs()):
        append_rows(os.path.join(gdir, f"{r['arm']}.jsonl"), r["rows"])
        done += 1
        if done % 10 == 0 or done == len(todo):
            el = (time.perf_counter() - t0) / 60
            print(f"  {done}/{len(todo)} chunks  {el:.1f} min", flush=True)
    if stopped[0]:
        print(f"--max-minutes reached: stopped dispatching; rerun the same command to resume", flush=True)
    return {a: len(read_rows(os.path.join(gdir, f"{a}.jsonl"))) for a in arms}


# --- analysis ------------------------------------------------------------------------------------

def _units(rows: Dict[int, dict], key: str) -> Dict[Tuple[int, int], int]:
    return {(seed, g["focal"]): int(g[key] > 0) if key != "win" else int(g["win"])
            for seed, r in rows.items() for g in r["games"]}


def mcnemar_one_sided(b: int, c: int) -> float:
    """P(X >= b | n = b + c, p = 1/2): arm-only events b vs base-only events c."""
    n = b + c
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, k) for k in range(b, n + 1)) / 2 ** n)


def paired_endpoint(arm_rows, base_rows, key: str, seed: int = 0, n_boot: int = 10_000) -> dict:
    a, b = _units(arm_rows, key), _units(base_rows, key)
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"n": 0}
    da = np.array([a[k] for k in keys], float)
    db = np.array([b[k] for k in keys], float)
    deal = np.array([k[0] for k in keys])
    diff = da - db
    ug, inv = np.unique(deal, return_inverse=True)
    s, n = np.bincount(inv, diff), np.bincount(inv).astype(float)
    rng, G, boots = np.random.default_rng(seed), len(ug), []
    for c in range(0, n_boot, 500):
        w = rng.multinomial(G, np.full(G, 1.0 / G), size=min(500, n_boot - c)).astype(float)
        boots.append((w @ s) / (w @ n))
    boots = np.concatenate(boots)
    b_only, c_only = int(((da == 1) & (db == 0)).sum()), int(((da == 0) & (db == 1)).sum())
    return {"n": len(keys), "arm_rate": float(da.mean()), "base_rate": float(db.mean()), "diff": float(diff.mean()),
            "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975)),
            "arm_only": b_only, "base_only": c_only, "p_one_sided": mcnemar_one_sided(b_only, c_only),
            "identical_games_frac": float((da == db).mean())}


def report(out: str, prereg: dict) -> dict:
    from .dla_faithfulness import atomic_json, holm
    gdir = os.path.join(out, "games")
    base = read_rows(os.path.join(gdir, "base.jsonl"))
    arms = sorted(f[:-6] for f in os.listdir(gdir) if f.endswith(".jsonl") and f != "base.jsonl")
    targets = {f"L{t['layer']}h{t['head']}": t["names"] for t in prereg["targets"]}
    res: Dict[str, dict] = {}
    for arm in arms:
        rows = read_rows(os.path.join(gdir, f"{arm}.jsonl"))
        res[arm] = {k: paired_endpoint(rows, base, k) for k in ("assassin_death", "unsafe_flips", "win")}
        loss = -res[arm]["win"]["diff"] if res[arm]["win"].get("n") else float("nan")
        res[arm]["death_over_loss"] = res[arm]["assassin_death"]["diff"] / loss if loss and loss > 0 else float("nan")
    opp = [g for r in base.values() for g in r["games"]]
    opps = sum(g["opp_reveal_opps"] for g in opp)
    reveal_rate = sum(g["opp_reveals"] for g in opp) / opps if opps else float("nan")
    co_primary = bool(reveal_rate == reveal_rate and reveal_rate < 0.9)
    pv = {a: res[a]["assassin_death"]["p_one_sided"] for a in targets if a in res}
    rej = holm(pv, 0.05) if pv else {}
    deaths = {a: res[a]["assassin_death"]["diff"] for a in res}
    worst_ctrl = max((res[a]["death_over_loss"] for a in res if a not in targets
                      and res[a]["death_over_loss"] == res[a]["death_over_loss"]), default=float("nan"))
    verdicts = {}
    for a in targets:
        if a not in res:
            verdicts[a] = "not run"
            continue
        e = res[a]["assassin_death"]
        spec_ok = (res[a]["death_over_loss"] >= 0.5) or (res[a]["death_over_loss"] > worst_ctrl)
        if rej.get(a) and deaths[a] == max(deaths.values()) and spec_ok:
            verdicts[a] = "confirmed"
        elif e["hi"] < 0.02:
            verdicts[a] = "falsified"
        else:
            verdicts[a] = "inconclusive"
    out_obj = {"base": {"deals": len(base), "death_rate": float(np.mean([g["assassin_death"] > 0 for g in opp])) if opp else None,
                        "opp_reveal_rate_when_held": reveal_rate, "unsafe_flips_co_primary": co_primary,
                        "win_rate_seat_sum": float(np.mean([g["win"] for g in opp])) if opp else None},
               "arms": res, "targets": targets, "holm_reject": rej, "verdicts": verdicts}
    atomic_json(os.path.join(out, "games_report.json"), out_obj)
    return out_obj


def plan_deals(out: str) -> dict:
    """From the base-arm PILOT: deals for 80% power to detect +2pp (one-sided .05), clipped to [2000, 5000]."""
    from .dla_faithfulness import atomic_json
    base = read_rows(os.path.join(out, "games", "base.jsonl"))
    gs = [g for r in base.values() for g in r["games"]]
    p0 = float(np.mean([g["assassin_death"] > 0 for g in gs]))
    p1 = p0 + 0.02
    units = (1.645 + 0.842) ** 2 * (p0 * (1 - p0) + p1 * (1 - p1)) / 0.02 ** 2
    deals = int(min(5000, max(2000, math.ceil(units / 2))))
    plan = {"pilot_deals": len(base), "pilot_death_rate": p0, "deals": deals, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    atomic_json(os.path.join(out, "games", "n_plan.json"), plan)
    return plan


def main(argv=None) -> None:
    from .dla_faithfulness import Ctx, load_prereg
    p = argparse.ArgumentParser(description="Study 10 Phase 4: greedy head-ablation games (resumable).")
    p.add_argument("cmd", choices=["pilot", "run", "report"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", default=None)
    p.add_argument("--out-root", default=os.path.join("reports", "dla_faithfulness"))
    p.add_argument("--arms", default="base,prereg,all-heads")
    p.add_argument("--deals", type=int, default=None, help="override n_plan.json (smoke tests only)")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--chunk", type=int, default=25)
    p.add_argument("--max-minutes", type=float, default=None)
    a = p.parse_args(argv)
    ctx = Ctx(a.ckpt, tag=a.tag, out_root=a.out_root)
    pre = load_prereg(ctx)                                  # refuses on checkpoint/dataset/means mismatch
    base_seed = pre["games"]["base_seed"]
    if a.cmd == "pilot":
        n = run_arms(a.ckpt, ctx.out, ["base"], a.deals or PILOT_DEALS, base_seed, a.workers, a.chunk, a.max_minutes)
        print(n, plan_deals(ctx.out))
        return
    if a.cmd == "report":
        r = report(ctx.out, pre)
        print(json.dumps({"base": r["base"], "verdicts": r["verdicts"],
                          "death_diff": {k: v["assassin_death"].get("diff") for k, v in r["arms"].items()}}, indent=1))
        return
    plan_p = os.path.join(ctx.out, "games", "n_plan.json")
    if a.deals is None and not os.path.exists(plan_p):
        raise SystemExit("run the base-arm `pilot` first (it fixes N in games/n_plan.json; commit it)")
    deals = a.deals or json.load(open(plan_p))["deals"]
    L, H = ctx.model.cfg.n_layers, ctx.model.cfg.n_heads
    tgt = [f"L{t['layer']}h{t['head']}" for t in pre["targets"]]
    arms: List[str] = []
    for part in a.arms.split(","):
        part = part.strip()
        add = ({"prereg": tgt, "all-heads": [f"L{li}h{h}" for li in range(L) for h in range(H)]}.get(part, [part]))
        arms += [x for x in add if x not in arms]
    print(f"arms (priority order): {arms}")
    print(run_arms(a.ckpt, ctx.out, arms, deals, base_seed, a.workers, a.chunk, a.max_minutes))


if __name__ == "__main__":
    main()
