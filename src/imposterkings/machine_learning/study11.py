"""Study 11 -- late-fusion Q-net vs d96: offline eval, speed, attribution, pre-registration, games. Resumable.

    python -m imposterkings.machine_learning.study11 eval       # O1/O2 for every trained arm + d96; selection
    python -m imposterkings.machine_learning.study11 speed      # S1: CPU leaf-evaluator throughput
    python -m imposterkings.machine_learning.study11 attrib     # A1-A3 on the selected arm
    python -m imposterkings.machine_learning.study11 pilot      # d96 vs d96 @k20: per-deal SD + timing (no LF games)
    python -m imposterkings.machine_learning.study11 select     # writes reports/study11/prereg.json  (commit it!)
    python -m imposterkings.machine_learning.study11 games --kind G1|G2 [--max-minutes M]
    python -m imposterkings.machine_learning.study11 report | status

Every step caches ``reports/study11/<step>.json`` with a meta guard (input SHA-256s + params); a changed input
raises ``StaleStep`` unless ``--force <step>``. Games append one JSONL row per finished deal chunk (fsynced) and
a rerun plays only the missing deals. ``games`` refuses to run unless ``prereg.json`` exists and its hashes
match the files on disk -- the pre-registration gate (the user commits prereg.json before any LF game).

Pre-registered thresholds are in ``THRESH``; the arm-selection rule in :func:`select_arm`. Do not change
either after looking at results.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from typing import Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch

from ..actions import ActionKind, StepKind
from . import features3 as F3
from . import latefusion_model as LF
from .dla_ablation_games import append_rows, read_rows
from .dla_faithfulness import (StaleStep, _jsonable, atomic_json, cluster_boot, git_head, sha256, spearman)
from .token_dataset import load_v3
from .train import game_split

OUT = os.path.join("reports", "study11")
ARMS_DIR = os.path.join("models", "study11")
D96 = os.path.join("models", "v3c_gen25", "attn_d96_L2.pt")
NPZ3 = os.path.join("datasets", "tensors", "v3c_gen25_tokens_v3.npz")
NPZ2 = os.path.join("datasets", "tensors", "v3c_gen25_tokens_v2.npz")
CORPUS = ["datasets/selfplay_gen1_v3c:1000", "datasets/selfplay_v3c_k50", "datasets/mixed_v3c_vs_mcts500",
          "datasets/forced_assassin_hide"]
STUDY10_D96 = os.path.join("reports", "dla_faithfulness", "v3c_gen25", "coverage.json")
# Evaluation deals live OUTSIDE every corpus seed decade (gen1 10M, k50 30M, mixed 40M, forced 50M, gen25 60M;
# Study 10 games 90M), so a later arm trained on a pooled corpus can never have seen a G1/G2 deal.
SEED_BASE = {"G1": 91_000_000, "G2": 92_000_000, "pilot": 80_000_000}
BUDGET = ("hybrid", 20, 3)

THRESH = {
    "O1_top1_margin": 0.01, "O1_top1_fail_margin": 0.02, "O1_spearman": 0.60,
    "S1_pass_ratio": 2.0, "S1_fail_ratio": 1.2,
    "G_noninferior_lo": 0.47, "G_superior_lo": 0.50, "G_fail_hi": 0.50,
    "A1_r2": 0.5, "A1_sign": 0.8, "A1_fail_r2": 0.25, "A1_d96_share": 0.47,
    "A2_spearman": 0.3, "A2_sign": 0.7,
    "G1_deals": 2000, "G2_margin": 0.03, "G2_power": 0.80, "G2_alpha": 0.05, "G2_min": 400, "G2_max": 1500,
    "select_tie_pp": 0.005,
}

ASSASSIN = F3.INSTANCE_NAMES.index("Assassin")                          # slot == state-token index (no CLS)
ZONE_SL = slice(F3._ZONE_OFF, F3._ZONE_OFF + 12)
THEIR_HAND = F3._ZONE_OFF + 3                                           # ZONES[3] == "their_hand"
MUTED = F3._V2_KEEP.index(24)                                           # v2 is_muted col -> v3 col
MAIN_IX = list(StepKind).index(StepKind.MAIN)
FLIP_KIND = ActionKind.FLIP_KING.value - 1


# --- infrastructure --------------------------------------------------------------------------------

def _no_throttle() -> None:
    """Windows 11 confines background-spawned process trees to the E-cores at low QoS (see the repo memory on
    EcoQoS): opt THIS process out of power throttling and pin it to the P-cores. No-op elsewhere."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        class _PPT(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                        ("StateMask", wintypes.ULONG)]

        k32 = ctypes.windll.kernel32
        h = k32.GetCurrentProcess()
        st = _PPT(1, 1, 0)
        k32.SetProcessInformation(h, 4, ctypes.byref(st), ctypes.sizeof(st))
        k32.SetProcessAffinityMask(h, ctypes.c_size_t(0xFFFF))
    except Exception:
        pass


class Cache:
    """Step cache keyed by input hashes + params (the Study 10 pattern, without a model-bound context)."""

    def __init__(self, out: str = OUT, force: Sequence[str] = ()):
        self.out, self.force = out, set(force)
        os.makedirs(out, exist_ok=True)
        self._sha: Dict[str, str] = {}

    def sha(self, path: str) -> str:
        if path not in self._sha:
            self._sha[path] = sha256(path)
        return self._sha[path]

    def step(self, name: str, inputs: Sequence[str], params: dict, fn: Callable[[], dict]) -> dict:
        path = os.path.join(self.out, f"{name}.json")
        want = {"inputs": {p: self.sha(p) for p in inputs}, "params": _jsonable(params)}
        if os.path.exists(path) and name not in self.force:
            with open(path) as f:
                blob = json.load(f)
            if blob["meta"]["inputs"] == want["inputs"] and blob["meta"]["params"] == want["params"]:
                print(f"[study11] {name}: cached")
                return blob["result"]
            raise StaleStep(f"{path} was computed from different inputs/params; rerun with --force {name}")
        t0 = time.perf_counter()
        print(f"[study11] {name}: running ...", flush=True)
        res = fn()
        meta = {**want, "git_head": git_head(), "seconds": round(time.perf_counter() - t0, 1),
                "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        atomic_json(path, {"meta": meta, "result": res})
        print(f"[study11] {name}: done in {meta['seconds']}s", flush=True)
        return _jsonable(res)

    def get(self, name: str) -> Optional[dict]:
        path = os.path.join(self.out, f"{name}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)["result"]


def arm_ckpts() -> List[str]:
    return sorted(p for p in glob.glob(os.path.join(ARMS_DIR, "*.pt")) if not p.endswith(".last.pt"))


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- O1 / O2: offline eval + arm selection ------------------------------------------------------------

def select_arm(arms: Dict[str, dict]) -> str:
    """Pre-registered rule: highest val top1_bestq among the beta=1 arms; within 0.5pp, fewer params wins."""
    pool = {k: v for k, v in arms.items() if v.get("beta", 1.0) > 0}
    best = max(v["top1_bestq"] for v in pool.values())
    near = [k for k, v in pool.items() if v["top1_bestq"] >= best - THRESH["select_tie_pp"]]
    return min(near, key=lambda k: (pool[k]["params"], -pool[k]["top1_bestq"]))


def step_eval(cache: Cache) -> dict:
    from .train_latefusion import DeviceData, flat_metrics, predict
    arms = arm_ckpts()

    def run():
        arrs = load_v3(NPZ3)
        tr, va = game_split(arrs["game_id"], 0.1, 0)
        vai = np.where(va)[0]
        with np.load(NPZ2) as z2:
            gid2 = z2["game_id"]
        _, va2 = game_split(gid2, 0.1, 0)
        same_val = set(np.unique(arrs["game_id"][vai]).tolist()) == set(np.unique(gid2[va2]).tolist())
        if not same_val:
            raise RuntimeError("v3 and v2 val games differ -- the comparison with d96 would be unpaired")
        out = {"val_decisions": int(len(vai)), "val_games_match_d96": same_val, "arms": {}}
        # d96 on its own (v2) val rows -- the same games
        from .attention_model import load as load_attn
        from .token_dataset import load as load_rows
        from .train import ranking_metrics
        from .train_tokens import _predict
        rows = load_rows(NPZ2)
        vrows = np.where(va2)[0]
        m96, _ = load_attn(D96)
        dev = _device()
        p96 = _predict(m96.to(dev), rows, vrows, 2048, dev)
        y96 = rows.y[vrows]
        out["d96"] = {"val_mse": float(np.mean((p96 - y96) ** 2)), "params": int(sum(p.numel() for p in m96.parameters())),
                      **ranking_metrics(p96, y96, rows.decision_id[vrows], rows.is_chosen[vrows])}
        # O2 needs the root value per decision: sum_a w*y
        root = (arrs["w"][vai] * arrs["y"][vai]).sum(1)
        maxy = np.where(arrs["cand_kind"][vai] >= 0, arrs["y"][vai], -np.inf).max(1)
        data = DeviceData(arrs, dev)
        for ck in arms:
            model, meta = LF.load(ck)
            model.to(dev)
            q, lg = predict(model, data, vai)
            m = flat_metrics(arrs, vai, q, lg)
            m.pop("by_corpus", None)
            vmax = np.where(arrs["cand_kind"][vai] >= 0, q, -np.inf).max(1)
            name = os.path.splitext(os.path.basename(ck))[0]
            out["arms"][name] = {**m, "params": LF.n_params(model), "cfg": model.cfg.__dict__,
                                 "beta": meta.get("config", {}).get("beta", 1.0), "ckpt": ck,
                                 "O2": {"maxQ_vs_root_mae": float(np.abs(vmax - root).mean()),
                                        "maxQ_vs_root_bias": float((vmax - root).mean()),
                                        "label_max_vs_root_bias": float((maxy - root).mean())}}
        sel = select_arm(out["arms"]) if out["arms"] else None
        out["selected"] = sel
        if sel:
            a, d = out["arms"][sel], out["d96"]
            ok = a["top1_bestq"] >= d["top1_bestq"] - THRESH["O1_top1_margin"] and a["spearman"] >= THRESH["O1_spearman"]
            fail = a["top1_bestq"] < d["top1_bestq"] - THRESH["O1_top1_fail_margin"]
            out["O1_verdict"] = "pass" if ok else ("fail" if fail else "inconclusive")
        return out

    return cache.step("eval", [NPZ3, NPZ2, D96] + arms, {"arms": arms}, run)


# --- val positions (live GameStates) for the speed bench ------------------------------------------------

def val_positions(n_max: int) -> list:
    """Real decision states from the VAL games, replayed from the corpus (decisions with >= 2 candidates)."""
    from ..record import dict_to_action, read_jsonl
    from ..state import GameState
    from .token_dataset import _match, _parse_data, _shards
    arrs_gid = np.load(NPZ3)["game_id"]
    _, va = game_split(arrs_gid, 0.1, 0)
    val_games = set(np.unique(arrs_gid[va]).tolist())
    dirs, caps, preds = _parse_data(CORPUS)
    out, per_dir = [], {}
    for src, f in _shards(dirs):
        if caps.get(src) and per_dir.get(src, 0) >= caps[src]:
            continue
        for rec in read_jsonl(f):
            if caps.get(src) and per_dir.get(src, 0) >= caps[src]:
                break
            if not _match(rec, preds.get(src, {})):
                continue
            per_dir[src] = per_dir.get(src, 0) + 1
            if rec["deal_seed"] not in val_games:
                continue
            st = GameState.deal(np.random.default_rng(rec["deal_seed"]), rec.get("starting_player"))
            for d in rec["decisions"]:
                if len(d.get("candidates") or []) >= 2:
                    out.append(st)
                    if len(out) >= n_max:
                        return out
                st = st.apply(dict_to_action(d["chosen"]))
    return out


def step_speed(cache: Cache, sel_ckpt: str, n: int = 5000) -> dict:
    def run():
        _no_throttle()
        torch.set_num_threads(1)
        from .attention_model import build_evaluator as be96
        states = val_positions(n)
        res = {"positions": len(states), "mean_candidates": float(np.mean([len(s.legal_moves()) for s in states]))}
        for name, ev in (("d96", be96(D96)), ("LF", LF.build_evaluator(sel_ckpt))):
            for s in states[:50]:
                ev(s)                                               # warm-up
            ts = []
            for s in states:
                t0 = time.perf_counter()
                ev(s)
                ts.append(time.perf_counter() - t0)
            ts = np.asarray(ts)
            res[name] = {"median_ms": float(np.median(ts) * 1e3), "mean_ms": float(ts.mean() * 1e3),
                         "evals_per_s": float(1.0 / ts.mean())}
        res["ratio_mean"] = res["LF"]["evals_per_s"] / res["d96"]["evals_per_s"]
        res["ratio_median"] = res["d96"]["median_ms"] / res["LF"]["median_ms"]
        r = res["ratio_median"]
        res["S1_verdict"] = ("pass" if r >= THRESH["S1_pass_ratio"] else
                             "fail" if r < THRESH["S1_fail_ratio"] else "inconclusive")
        return res

    return cache.step("speed", [D96, sel_ckpt, NPZ3], {"n": n, "threads": 1}, run)


# --- A1-A3: attribution ----------------------------------------------------------------------------------

def _decomp_all(model, arrs, idx, cards=None, cross_bias=None, bs=512):
    """Flattened per-candidate decomposition over decisions ``idx``: dict of arrays + offsets."""
    dev = next(model.parameters()).device
    keep = {k: [] for k in ("q_pre", "cand", "attn", "ffn", "tok", "resid")}
    sub = dict(arrs)
    if cards is not None:
        sub = {**arrs, "cards": cards}
    for s in range(0, len(idx), bs):
        ii = idx[s:s + bs]
        b = LF.to_device(LF.batch_from_arrays(sub, ii), dev)
        d = LF.decompose(model, b, cross_bias=cross_bias)
        valid = (b["cand_kind"] >= 0).cpu().numpy()
        keep["q_pre"].append(d.q_pre.cpu().numpy()[valid])
        keep["cand"].append(d.cand.cpu().numpy()[valid])
        keep["attn"].append(d.attn.sum((1, 2, 4)).cpu().numpy()[valid])
        keep["ffn"].append(d.ffn.sum(1).cpu().numpy()[valid])
        keep["tok"].append(d.attn.sum((1, 2)).cpu().numpy()[valid])                         # [rows, S]
        keep["resid"].append(d.resid.cpu().numpy()[valid])
    out = {k: np.concatenate(v) for k, v in keep.items()}
    n = (arrs["cand_kind"][idx] >= 0).sum(1)
    out["off"] = np.r_[0, np.cumsum(n)]
    return out


def _top2(z, off):
    a, b = [], []
    for i in range(len(off) - 1):
        o = off[i] + np.argsort(-z[off[i]:off[i + 1]])
        a.append(o[0]); b.append(o[1])
    return np.array(a), np.array(b)


def _flipF(z, off, flip_pos):
    """F = z(flip) - max non-flip z, for decisions with a flip candidate at local index flip_pos."""
    F = np.empty(len(flip_pos))
    for i, f in enumerate(flip_pos):
        seg = z[off[i]:off[i + 1]]
        F[i] = seg[f] - np.delete(seg, f).max()
    return F


def cf_lacks(cards: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """The v3 'lacks' counterfactual on decisions ``rows``: Assassin's their_hand mass -> 0, renormalised
    (identical to tokenizing with ``opp_hand_lacks | {'Assassin'}``; it touches only this row's zone block)."""
    out = cards.copy()
    post = out[rows, ASSASSIN, ZONE_SL].copy()
    post[:, 3] = 0.0
    post /= post.sum(-1, keepdims=True)
    out[rows, ASSASSIN, ZONE_SL] = post
    return out


def strata(arrs, idx):
    """Study 10's strata on v3 tokens: (threat mask, safe mask, flip local index) per decision in ``idx``."""
    cards = arrs["cards"][idx]
    post = cards[:, ASSASSIN, ZONE_SL]
    unseen = (post.max(-1) < 1 - 1e-6) & (cards[:, ASSASSIN, THEIR_HAND] > 1e-6)
    located = (post.max(-1) >= 1 - 1e-6) | (cards[:, ASSASSIN, MUTED] > 0.5)
    main = arrs["decision"][idx, MAIN_IX] > 0.5
    flip_legal = arrs["kings"][idx, 0, 1] > 0.5
    kinds = arrs["cand_kind"][idx]
    nflip = (kinds == FLIP_KIND).sum(1)
    ncand = (kinds >= 0).sum(1)
    ok = main & flip_legal & (nflip == 1) & (ncand >= 2)
    flip = np.argmax(kinds == FLIP_KIND, axis=1)
    return ok & unseen, ok & located, flip


def step_attrib(cache: Cache, sel_ckpt: str) -> dict:
    def run():
        dev = _device()
        arrs = load_v3(NPZ3)
        _, va = game_split(arrs["game_id"], 0.1, 0)
        vai = np.where(va)[0]
        model, _ = LF.load(sel_ckpt)
        model.to(dev)
        S = F3.S_STATE
        res: dict = {"ckpt": sel_ckpt, "val_decisions": int(len(vai))}
        # A1 -- contrastive coverage (top-1 vs runner-up by the model's own pre-tanh q)
        D = _decomp_all(model, arrs, vai)
        res["max_abs_resid"] = float(np.abs(D["resid"]).max())
        a, b = _top2(D["q_pre"], D["off"])
        delta = D["q_pre"][a] - D["q_pre"][b]
        comps = {"cand_embed": D["cand"], "cross_attn": D["attn"], "cross_ffn": D["ffn"]}
        tot = sum(np.abs(v[a] - v[b]).mean() for v in comps.values())
        contr = {}
        for c, v in comps.items():
            dc = v[a] - v[b]
            contr[c] = {"abs_share": float(np.abs(dc).mean() / tot), "sign_agree": float((dc > 0).mean()),
                        "r2": float(1 - ((delta - dc) ** 2).sum() / ((delta - delta.mean()) ** 2).sum())}
        dtok = D["tok"][a] - D["tok"][b]
        res["A1"] = {"contrastive": contr, "decisions": int(len(a)),
                     "token_abs_contrast": {F3.STATE_LABELS[j]: float(np.abs(dtok[:, j]).mean()) for j in
                                            np.argsort(-np.abs(dtok).mean(0))}}
        ca = contr["cross_attn"]
        if os.path.exists(STUDY10_D96):
            with open(STUDY10_D96) as f:
                s10 = json.load(f)["result"]
            res["A1"]["d96_drawer_study10"] = s10["contrastive"][s10["drawer_component"]]
            res["A1"]["d96_all_components_study10"] = s10["contrastive"]
        res["A1"]["verdict"] = ("adequate" if ca["r2"] >= THRESH["A1_r2"] and ca["sign_agree"] >= THRESH["A1_sign"]
                                and ca["abs_share"] > THRESH["A1_d96_share"] else
                                "fail" if ca["r2"] < THRESH["A1_fail_r2"] else "partial")
        # A2 -- the Assassin 'lacks' counterfactual in the threat stratum
        thr, safe, flip = strata(arrs, vai)
        tdec = vai[thr]
        games = arrs["game_id"][tdec]
        res["A2"] = {"threat_decisions": int(thr.sum()), "threat_games": int(len(np.unique(games))),
                     "safe_decisions": int(safe.sum())}
        if thr.sum() >= 10:
            D0 = _decomp_all(model, arrs, tdec)
            cards_cf = cf_lacks(arrs["cards"], tdec)
            D1 = _decomp_all(model, arrs, tdec, cards=cards_cf)
            F0 = _flipF(D0["q_pre"], D0["off"], flip[thr])
            F1 = _flipF(D1["q_pre"], D1["off"], flip[thr])
            res["A2"]["P(F>0)_base"] = float((F0 > 0).mean())
            res["A2"]["lacks_dF"] = cluster_boot(F1 - F0, games)
            res["A2"]["P0_pass"] = bool(res["A2"]["lacks_dF"]["lo"] > 0)
            T = D1["q_pre"] - D0["q_pre"]
            Dass = D0["tok"][:, ASSASSIN]                                    # cross-attn DLA on the Assassin
            res["A2"]["predicts"] = {"spearman(-D,T)": spearman(-Dass, T),
                                     "sign_agree": float((np.sign(-Dass) == np.sign(T)).mean()), "n": int(len(T))}
            p = res["A2"]["predicts"]
            res["A2"]["pass"] = bool(res["A2"]["P0_pass"] and p["spearman(-D,T)"] >= THRESH["A2_spearman"]
                                     and p["sign_agree"] >= THRESH["A2_sign"])
            # A3 -- mediation of the lacks effect + knockout of candidate -> Assassin
            dt = D1["tok"] - D0["tok"]
            own = dt[:, ASSASSIN]
            other = dt.sum(1) - own
            cand = D1["cand"] - D0["cand"]
            ffn = D1["ffn"] - D0["ffn"]
            den = np.abs(own).mean() + np.abs(other).mean() + np.abs(cand).mean() + np.abs(ffn).mean()
            oth = np.abs(dt).mean(0)
            oth[ASSASSIN] = 0
            kb = torch.zeros(1, 1, 1, S, device=dev)
            kb[..., ASSASSIN] = float("-inf")
            Dk = _decomp_all(model, arrs, tdec, cross_bias=[kb] * model.cfg.n_cross)
            Fk = _flipF(Dk["q_pre"], Dk["off"], flip[thr])
            res["A3"] = {"share_own_assassin": float(np.abs(own).mean() / den),
                         "share_other_tokens": float(np.abs(other).mean() / den),
                         "share_cand_embed": float(np.abs(cand).mean() / den),
                         "share_cross_ffn": float(np.abs(ffn).mean() / den),
                         "top_other_tokens_abs": {F3.STATE_LABELS[j]: float(oth[j]) for j in np.argsort(-oth)[:5]},
                         "closure": float(np.abs(T - (own + other + cand + ffn)).max()),
                         "knockout_dF": cluster_boot(Fk - F0, games),
                         "knockout_over_lacks": float((Fk - F0).mean() / (F1 - F0).mean())
                         if abs((F1 - F0).mean()) > 1e-9 else None}
        return res

    return cache.step("attrib", [NPZ3, sel_ckpt], {}, run)


# --- games -------------------------------------------------------------------------------------------------

def _agent(ckpt: str, kind: str):
    if kind == "G1":
        from .agent import NNAgent
        return NNAgent.from_checkpoint(ckpt)
    from .benchmark import _make_agent
    return _make_agent(("nnmcts", ckpt, BUDGET))


def _chunk(kind: str, lf: str, seeds: List[int]) -> dict:
    """One worker: for each seed, two mirrored games (challenger in seat 0, then seat 1). ``pilot`` plays d96
    against itself (measures per-deal SD and time without looking at the LF result)."""
    _no_throttle()
    torch.set_num_threads(1)
    from ..arena import play_game
    t0 = time.perf_counter()
    gk = "G2" if kind == "pilot" else kind
    chal = _agent(D96 if kind == "pilot" else lf, gk)
    opp = _agent(D96, gk)
    rows = []
    for seed in seeds:
        ts = time.perf_counter()
        wins, plies = 0, []
        for ns in (0, 1):
            agents = [None, None]
            agents[ns], agents[1 - ns] = chal, opp
            winner, _, final = play_game(agents, np.random.default_rng(seed), play_rng=np.random.default_rng([seed, ns]))
            wins += int(winner == ns)
            plies.append(len(final.history))
        rows.append({"seed": seed, "wins": wins, "score": wins / 2.0, "plies": plies,
                     "seconds": time.perf_counter() - ts})
    return {"kind": kind, "rows": rows, "seconds": time.perf_counter() - t0}


def run_games(kind: str, lf: str, deals: int, workers: int, chunk: int, max_minutes: Optional[float]) -> int:
    from joblib import Parallel, delayed
    _no_throttle()
    path = os.path.join(OUT, "games", f"{kind}.jsonl")
    seeds = [SEED_BASE[kind] + i for i in range(deals)]
    have = read_rows(path)
    missing = [s for s in seeds if s not in have]
    todo = [missing[i:i + chunk] for i in range(0, len(missing), chunk)]
    print(f"{kind}: {len(have)} deals on disk, {len(missing)} to play in {len(todo)} chunks", flush=True)
    t0 = time.perf_counter()

    def jobs() -> Iterator:
        for ch in todo:
            if max_minutes is not None and (time.perf_counter() - t0) / 60 > max_minutes:
                print("--max-minutes reached: stopped dispatching; rerun the same command to resume", flush=True)
                return
            yield delayed(_chunk)(kind, lf, ch)

    done = 0
    for r in Parallel(n_jobs=workers, return_as="generator_unordered", pre_dispatch="2*n_jobs")(jobs()):
        append_rows(path, r["rows"])
        done += 1
        if done % 5 == 0 or done == len(todo):
            print(f"  {done}/{len(todo)} chunks  {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return len(read_rows(path))


def summarize(kind: str, deals: Optional[int] = None) -> dict:
    rows = read_rows(os.path.join(OUT, "games", f"{kind}.jsonl"))
    if deals is not None:
        lo = SEED_BASE[kind]
        rows = {s: r for s, r in rows.items() if lo <= s < lo + deals}
    if not rows:
        return {"deals": 0}
    sc = np.array([r["score"] for r in rows.values()])
    n = len(sc)
    sd = float(sc.std(ddof=1)) if n > 1 else float("nan")
    ci = 1.96 * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"deals": n, "games": 2 * n, "winrate": float(sc.mean()), "sd_per_deal": sd, "ci95": ci,
            "lo": float(sc.mean() - ci), "hi": float(sc.mean() + ci),
            "splits": int(sum(r["wins"] == 1 for r in rows.values())),
            "sec_per_deal_worker": float(np.mean([r["seconds"] for r in rows.values()]))}


def verdict(s: dict) -> str:
    if not s.get("deals"):
        return "no data"
    if s["lo"] > THRESH["G_superior_lo"]:
        return "stronger"
    if s["lo"] > THRESH["G_noninferior_lo"]:
        return "non-inferior"
    if s["hi"] < THRESH["G_fail_hi"]:
        return "weaker (fail)"
    return "inconclusive"


def plan_g2(pilot: dict) -> int:
    """Deals for 80% power, one-sided alpha .05, at a -3pp non-inferiority margin, from the pilot's per-deal SD
    (d96 vs itself, so it reveals nothing about LF), clipped to [400, 1500]."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - THRESH["G2_alpha"]) + NormalDist().inv_cdf(THRESH["G2_power"])
    n = math.ceil((z * pilot["sd_per_deal"] / THRESH["G2_margin"]) ** 2)
    return int(min(max(n, THRESH["G2_min"]), THRESH["G2_max"]))


# --- pre-registration --------------------------------------------------------------------------------------

def prereg_path() -> str:
    return os.path.join(OUT, "prereg.json")


def write_prereg(cache: Cache) -> dict:
    ev = cache.get("eval")
    if not ev or not ev.get("selected"):
        raise SystemExit("run `study11 eval` first (selection needs the trained arms)")
    pilot = summarize("pilot")
    if pilot.get("deals", 0) < 50:
        raise SystemExit("run `study11 pilot` first (G2's N comes from the pilot SD)")
    sel = ev["arms"][ev["selected"]]["ckpt"]
    pre = {"selected_arm": ev["selected"], "selected_ckpt": sel, "selected_sha256": sha256(sel),
           "d96": D96, "d96_sha256": sha256(D96), "npz3_sha256": sha256(NPZ3),
           "thresholds": THRESH, "G1_deals": THRESH["G1_deals"], "G2_deals": plan_g2(pilot),
           "pilot": pilot, "seed_base": SEED_BASE, "budget": "hybrid-k20-l3", "prior": "softmax(policy logit)",
           "git_head": git_head(), "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    atomic_json(prereg_path(), pre)
    return pre


def load_prereg() -> dict:
    p = prereg_path()
    if not os.path.exists(p):
        raise SystemExit("no prereg.json -- run `study11 select` and commit it before playing LF games")
    with open(p) as f:
        pre = json.load(f)
    for k, path in (("selected_sha256", pre["selected_ckpt"]), ("d96_sha256", pre["d96"]), ("npz3_sha256", NPZ3)):
        if sha256(path) != pre[k]:
            raise SystemExit(f"prereg hash mismatch for {path} -- the pre-registered file changed")
    return pre


# --- report / status ---------------------------------------------------------------------------------------

def report(cache: Cache) -> dict:
    pre = load_prereg()
    out = {"prereg": {k: pre[k] for k in ("selected_arm", "G1_deals", "G2_deals")}}
    for k in ("eval", "speed", "attrib"):
        out[k] = cache.get(k)
    for kind in ("G1", "G2"):
        s = summarize(kind, pre[f"{kind}_deals"])
        s["complete"] = s.get("deals", 0) >= pre[f"{kind}_deals"]
        s["verdict"] = verdict(s)
        out[kind] = s
    out["pilot"] = summarize("pilot")
    atomic_json(os.path.join(OUT, "report.json"), out)
    return out


def status(cache: Cache) -> None:
    for k in ("eval", "speed", "attrib"):
        print(f"  {k:<8} {'done' if cache.get(k) is not None else 'missing'}")
    print(f"  prereg   {'written' if os.path.exists(prereg_path()) else 'missing'}")
    for kind in ("pilot", "G1", "G2"):
        print(f"  {kind:<8} {len(read_rows(os.path.join(OUT, 'games', f'{kind}.jsonl')))} deals on disk")
    print(f"  arms     {', '.join(os.path.basename(p) for p in arm_ckpts()) or 'none'}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Study 11: late-fusion vs d96 (resumable).")
    p.add_argument("cmd", choices=["eval", "speed", "attrib", "pilot", "select", "games", "report", "status"])
    p.add_argument("--kind", choices=["G1", "G2"], help="games: which pre-registered match")
    p.add_argument("--arm-ckpt", default=None, help="override the selected arm (speed/attrib)")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--chunk", type=int, default=10)
    p.add_argument("--deals", type=int, default=None, help="pilot: deals (default 60)")
    p.add_argument("--max-minutes", type=float, default=None)
    p.add_argument("--force", nargs="*", default=[])
    a = p.parse_args(argv)
    cache = Cache(force=a.force)

    def selected() -> str:
        if a.arm_ckpt:
            return a.arm_ckpt
        ev = cache.get("eval")
        if not ev or not ev.get("selected"):
            raise SystemExit("run `study11 eval` first")
        return ev["arms"][ev["selected"]]["ckpt"]

    if a.cmd == "eval":
        r = step_eval(cache)
        print(json.dumps({k: r[k] for k in ("selected", "O1_verdict") if k in r}, indent=1))
    elif a.cmd == "speed":
        r = step_speed(cache, selected())
        print(json.dumps(r, indent=1))
    elif a.cmd == "attrib":
        step_attrib(cache, selected())
    elif a.cmd == "pilot":
        run_games("pilot", "", a.deals or 60, a.workers, a.chunk, a.max_minutes)
        print(json.dumps(summarize("pilot"), indent=1))
    elif a.cmd == "select":
        print(json.dumps(write_prereg(cache), indent=1))
        print(f"\nCOMMIT {prereg_path()} (git add -f) BEFORE running `study11 games`.")
    elif a.cmd == "games":
        if not a.kind:
            raise SystemExit("--kind G1|G2")
        pre = load_prereg()
        n = run_games(a.kind, pre["selected_ckpt"], pre[f"{a.kind}_deals"], a.workers, a.chunk, a.max_minutes)
        s = summarize(a.kind, pre[f"{a.kind}_deals"])
        print(json.dumps({**s, "verdict": verdict(s), "target_deals": pre[f"{a.kind}_deals"], "on_disk": n}, indent=1))
    elif a.cmd == "report":
        print(json.dumps(report(cache), indent=1))
    else:
        status(cache)


if __name__ == "__main__":
    main()
