"""Study 10 -- is the drawer's DLA faithful? Offline battery (Phases 1-3), resumable.

    python -m imposterkings.machine_learning.dla_faithfulness all     --ckpt models/v3c_gen25/attn_d96_L2.pt
    python -m imposterkings.machine_learning.dla_faithfulness status  --ckpt models/v3c_gen25/attn_d96_L2.pt
    python -m imposterkings.machine_learning.dla_faithfulness select  --ckpt ...   # writes prereg.json
    python -m imposterkings.machine_learning.dla_faithfulness confirm --ckpt ...   # val-B, needs prereg

The drawer's signed mode is direct logit attribution (DLA) on the LAST layer's attention path. Every step
here makes a prediction that can FAIL (thresholds below are fixed in code before any data is looked at):

* 1a ``coverage``  -- full exact decomposition of the logit (``interventions.decompose``); how much of the
  CONTRASTIVE logit (top-1 minus runner-up, the thing that decides the move) the drawer's slice carries.
* 1b ``assassin_const`` -- is the Assassin's DLA constant across a decision's candidates? A constant offset
  cancels in Q(a) - Q(b) and explains no choice.
* 1c ``nulls`` -- how often the Assassin tops a head's signed DLA, vs every other token and vs untrained nets.
* 1d ``p0`` -- the gate: does telling the model "the Assassin is NOT in their hand" make flipping the king
  more attractive in threat positions? If not, there is no Assassin "fear" for any head to carry.
* 2a ``heads`` -- ablate every (layer, head): direct vs indirect effect, flip-contrast, argmax flips.
* 2b ``cf`` -- semantic counterfactuals and their mediation; does the drawer's Assassin bar predict them?
* 2c ``knockout`` -- K1 (CLS->j, last layer) / K2 (CLS->j, all layers) / K3 (all ->j, all layers) per card.
* 2d ``patch`` -- resample-patch the Assassin's residual entering the last layer.
* 2e ``sanity`` -- cascading weight randomisation (Adebayo et al.): a faithful map must decorrelate.

Splits: TRAIN (``game_split(.., 0.1, 0)``, the training default) only for ablation means; the model's own
VAL split is halved by game into val-A (exploration + head selection) and val-B (untouched until
``confirm``). Every analysis is in pre-tanh LOGIT space.

Resumable: each step caches ``<out>/<step>.json`` with a meta guard (checkpoint + dataset SHA-256 + params);
a re-run skips matching steps, refuses mismatched ones unless ``--force <step>``; long steps (heads, sanity)
also save per-unit partials. Writes are atomic (tmp + os.replace).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..actions import ActionKind, StepKind
from . import features2 as F2
from . import interventions as I
from .attention_model import AttentionModel, load as load_attn
from .token_dataset import load as load_rows
from .train import corpus_of, game_split, CORPUS_NAMES

MAIN_IX = list(StepKind).index(StepKind.MAIN)
FLIP_COL = ActionKind.FLIP_KING.value - 1
MUTED_COL = F2._STATE_OFF + F2.STATE_FIELDS.index("is_muted")
J = I.ASSASSIN_SEQ
CARD_SEQ = list(range(1, 19))
NONCLS = list(range(1, 24))
BOOT = 10_000

# Pre-stated thresholds (Study 10). Changing these after looking at results defeats the point.
THRESH = {
    "coverage_adequate_r2": 0.5, "coverage_adequate_sign": 0.8, "coverage_minority_r2": 0.25,
    "const_rho": 0.10,
    "null_ratio": 0.5,
    "cf_spearman": 0.3, "cf_sign": 0.7,
    "sanity_spearman": 0.2,
    "P1_pp": 0.02, "P3_recovery": 0.25, "alpha": 0.05,
}


# --- io ----------------------------------------------------------------------------------------

def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    return o


def atomic_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_jsonable(obj), f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


class StaleStep(RuntimeError):
    pass


# --- context -------------------------------------------------------------------------------------

@dataclass
class Decision:
    s: int                      # [s, e) positions into the analysed index array
    e: int
    game: int
    corpus: int
    flip: int                   # position of the FLIP_KING candidate, or -1
    threat: bool
    safe: bool


class Ctx:
    """One checkpoint + its dataset + splits + the step cache directory."""

    def __init__(self, ckpt: str, npz: Optional[str] = None, tag: Optional[str] = None,
                 out_root: str = os.path.join("reports", "dla_faithfulness"), force: Sequence[str] = (),
                 max_rows: Optional[int] = None):
        self.ckpt = ckpt
        self.model, meta = load_attn(ckpt)
        if self.model.cfg.feat != "v2":
            raise SystemExit("Study 10 targets featurization v2 checkpoints")
        self.npz = npz or meta.get("npz", "").replace("../", "")
        self.tag = tag or os.path.basename(os.path.dirname(os.path.abspath(ckpt)))
        self.out = os.path.join(out_root, self.tag)
        self.force = set(force)
        self.max_rows = max_rows
        os.makedirs(self.out, exist_ok=True)
        self.ckpt_sha, self.npz_sha = sha256(ckpt), sha256(self.npz)
        self.rows = load_rows(self.npz)
        tr, va = game_split(self.rows.game_id, 0.1, 0)              # the training default split
        self.tri = np.where(tr)[0]
        vai = np.where(va)[0]
        rest_b, is_b = game_split(self.rows.game_id[vai], 0.5, 1)   # halve VAL by game: A explore, B confirm
        self.val_a, self.val_b = vai[rest_b], vai[is_b]
        self._cache: Dict[str, object] = {}

    # meta guard -------------------------------------------------------------------------------
    def meta(self, step: str, params: dict) -> dict:
        return {"step": step, "ckpt": self.ckpt, "ckpt_sha256": self.ckpt_sha, "npz": self.npz,
                "npz_sha256": self.npz_sha, "params": params, "max_rows": self.max_rows}

    def _match(self, have: dict, want: dict) -> bool:
        keys = ("ckpt_sha256", "npz_sha256", "params", "max_rows")
        return all(_jsonable(have.get(k)) == _jsonable(want.get(k)) for k in keys)

    def step(self, name: str, params: dict, fn: Callable[[], dict]) -> dict:
        path = os.path.join(self.out, f"{name}.json")
        want = self.meta(name, params)
        if os.path.exists(path) and name not in self.force:
            with open(path) as f:
                blob = json.load(f)
            if self._match(blob["meta"], want):
                print(f"[{self.tag}] {name}: cached")
                return blob["result"]
            raise StaleStep(f"{path} was computed with a different config; rerun with --force {name}")
        t0 = time.perf_counter()
        print(f"[{self.tag}] {name}: running ...", flush=True)
        res = fn()
        want.update(git_head=git_head(), seconds=round(time.perf_counter() - t0, 1),
                    finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        atomic_json(path, {"meta": want, "result": res})
        print(f"[{self.tag}] {name}: done in {want['seconds']}s", flush=True)
        return _jsonable(res)

    def partial(self, name: str, params: dict) -> Tuple[str, dict]:
        """Per-unit partial results of a long step (survives a kill); dropped if the config changed."""
        path = os.path.join(self.out, f"{name}.partial.json")
        if os.path.exists(path) and name not in self.force:
            with open(path) as f:
                blob = json.load(f)
            if self._match(blob["meta"], self.meta(name, params)):
                return path, blob["units"]
        return path, {}

    def save_partial(self, path: str, name: str, params: dict, units: dict) -> None:
        atomic_json(path, {"meta": self.meta(name, params), "units": units})

    # data -----------------------------------------------------------------------------------
    def split(self, which: str) -> np.ndarray:
        idx = {"A": self.val_a, "B": self.val_b}[which]
        return idx if self.max_rows is None else _whole_decisions(self.rows, idx, self.max_rows)

    def means(self) -> torch.Tensor:
        if "means" in self._cache:
            return self._cache["means"]
        path = os.path.join(self.out, "means.pt")
        meta_p = path + ".meta.json"
        want = self.meta("means", {"train_cap": 100_000, "seed": 0})
        if os.path.exists(path) and os.path.exists(meta_p) and "means" not in self.force:
            with open(meta_p) as f:
                if self._match(json.load(f), want):
                    self._cache["means"] = torch.load(path)
                    return self._cache["means"]
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(self.tri, size=min(100_000, len(self.tri)), replace=False))
        print(f"[{self.tag}] means: {len(idx)} TRAIN rows ...", flush=True)
        m = I.head_means(self.model, self.rows, idx)
        torch.save(m, path + ".tmp")
        os.replace(path + ".tmp", path)
        atomic_json(meta_p, want)
        self._cache["means"] = m
        return m

    def means_sha(self) -> str:
        self.means()
        return sha256(os.path.join(self.out, "means.pt"))

    def decisions(self, idx: np.ndarray) -> List[Decision]:
        key = ("dec", idx.ctypes.data, len(idx))
        if key in self._cache:
            return self._cache[key]
        r = self.rows
        did = r.decision_id[idx]
        starts = np.r_[0, np.flatnonzero(np.diff(did) != 0) + 1]
        ends = np.r_[starts[1:], len(idx)]
        cards = r.cards[idx]
        unseen = I.assassin_unseen_in_their_hand(cards)
        located = (cards[:, I.ASSASSIN_SLOT, I.ZONE_SL].max(-1) >= 1.0 - 1e-6) | (
            cards[:, I.ASSASSIN_SLOT, MUTED_COL] > 0.5)
        main = r.phase[idx, MAIN_IX] > 0.5
        flip_legal = r.kings[idx, 0, 3] > 0.5
        is_flip = r.action[idx, FLIP_COL] > 0.5
        corp = corpus_of(r.game_id[idx])
        out = []
        for s, e in zip(starts, ends):
            fl = np.flatnonzero(is_flip[s:e])
            ok = bool(main[s] and flip_legal[s] and len(fl) == 1 and e - s >= 2)
            out.append(Decision(int(s), int(e), int(r.game_id[idx[s]]), int(corp[s]),
                                int(s + fl[0]) if len(fl) == 1 else -1,
                                ok and bool(unseen[s]), ok and bool(located[s])))
        self._cache[key] = out
        return out

    def base(self, idx: np.ndarray) -> I.Decomp:
        key = ("base", idx.ctypes.data, len(idx))
        if key not in self._cache:
            self._cache[key] = decomp_rows(self.model, self.rows, idx)
        return self._cache[key]


def _whole_decisions(rows, idx: np.ndarray, cap: int) -> np.ndarray:
    if len(idx) <= cap:
        return idx
    did = rows.decision_id[idx]
    cut = cap
    while cut < len(idx) and did[cut] == did[cut - 1]:
        cut += 1
    return idx[:cut]


def decomp_rows(model, rows, idx: np.ndarray, spec: I.InterventionSpec = I.InterventionSpec(),
                means=None, cards: Optional[np.ndarray] = None, batch: int = 2048) -> I.Decomp:
    parts = []
    for s in range(0, len(idx), batch):
        sl = idx[s:s + batch]
        c = None if cards is None else cards[s:s + batch]
        parts.append(I.decompose(model, I.batch_from_rows(rows, sl, c), spec, means=means))
    return I.Decomp(*[np.concatenate([getattr(p, f) for p in parts])
                      for f in ("logit", "q", "const", "attn", "attn_head", "ffn", "resid")])


def logits_rows(model, rows, idx, spec=I.InterventionSpec(), means=None, cards=None, batch=4096,
                patch_fn=None) -> np.ndarray:
    out = []
    for s in range(0, len(idx), batch):
        sl = idx[s:s + batch]
        c = None if cards is None else cards[s:s + batch]
        patch = None if patch_fn is None else patch_fn(s, s + len(sl))
        out.append(I.logits(model, I.batch_from_rows(rows, sl, c), spec, means=means, patch=patch))
    return np.concatenate(out)


# --- statistics ----------------------------------------------------------------------------------

def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def cluster_boot(vals: np.ndarray, games: np.ndarray, seed: int = 0, n_boot: int = BOOT) -> dict:
    """Mean of ``vals`` with a 95% CI from resampling GAMES (decisions within a game are correlated)."""
    vals = np.asarray(vals, np.float64)
    if len(vals) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0, "games": 0, "p_le0": float("nan")}
    ug, inv = np.unique(games, return_inverse=True)
    s, n = np.bincount(inv, vals), np.bincount(inv).astype(np.float64)
    rng = np.random.default_rng(seed)
    G, means = len(ug), []
    for c in range(0, n_boot, 500):
        w = rng.multinomial(G, np.full(G, 1.0 / G), size=min(500, n_boot - c)).astype(np.float64)
        means.append((w @ s) / np.maximum(w @ n, 1e-12))
    means = np.concatenate(means)
    return {"mean": float(vals.mean()), "lo": float(np.quantile(means, 0.025)),
            "hi": float(np.quantile(means, 0.975)), "n": int(len(vals)), "games": int(G),
            "p_le0": float((means <= 0).mean())}


def did_boot(a_vals, a_games, b_vals, b_games, seed: int = 0, n_boot: int = BOOT) -> dict:
    """Difference of means (A - B) with games resampled JOINTLY across the two strata."""
    games = np.unique(np.r_[a_games, b_games])
    gi = {g: i for i, g in enumerate(games)}
    G = len(games)
    ia = np.array([gi[g] for g in a_games], dtype=np.int64)
    ib = np.array([gi[g] for g in b_games], dtype=np.int64)
    sa, na = np.bincount(ia, a_vals, G), np.bincount(ia, minlength=G).astype(float)
    sb, nb = np.bincount(ib, b_vals, G), np.bincount(ib, minlength=G).astype(float)
    rng, out = np.random.default_rng(seed), []
    for c in range(0, n_boot, 500):
        w = rng.multinomial(G, np.full(G, 1.0 / G), size=min(500, n_boot - c)).astype(float)
        out.append((w @ sa) / np.maximum(w @ na, 1e-12) - (w @ sb) / np.maximum(w @ nb, 1e-12))
    out = np.concatenate(out)
    return {"diff": float(np.mean(a_vals) - np.mean(b_vals)), "lo": float(np.quantile(out, 0.025)),
            "hi": float(np.quantile(out, 0.975)), "p_le0": float((out <= 0).mean())}


def holm(pvals: Dict[str, float], alpha: float) -> Dict[str, bool]:
    order = sorted(pvals, key=lambda k: pvals[k])
    rej, m = {}, len(order)
    still = True
    for i, k in enumerate(order):
        still = still and pvals[k] <= alpha / (m - i)
        rej[k] = still
    return rej


def flip_contrast(z: np.ndarray, decs: List[Decision]) -> np.ndarray:
    """F = logit(flip) - max non-flip logit, per decision (decisions must have a flip candidate)."""
    out = np.empty(len(decs))
    for i, d in enumerate(decs):
        others = np.r_[np.arange(d.s, d.flip), np.arange(d.flip + 1, d.e)]
        out[i] = z[d.flip] - z[others].max()
    return out


def top2(z: np.ndarray, decs: List[Decision]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For decisions with >= 2 candidates: positions of the model's top-1 and runner-up, and a keep mask."""
    a, b, keep = [], [], []
    for d in decs:
        if d.e - d.s < 2:
            keep.append(False); a.append(d.s); b.append(d.s)
            continue
        o = d.s + np.argsort(-z[d.s:d.e])
        a.append(o[0]); b.append(o[1]); keep.append(True)
    return np.array(a), np.array(b), np.array(keep)


def argmax_per_decision(z: np.ndarray, decs: List[Decision]) -> np.ndarray:
    return np.array([d.s + int(np.argmax(z[d.s:d.e])) for d in decs])


# --- Phase 1 -------------------------------------------------------------------------------------

def step_coverage(ctx: Ctx, which: str = "A") -> dict:
    idx = ctx.split(which)
    D, decs = ctx.base(idx), ctx.decisions(idx)
    comps = D.components
    L = D.attn.shape[1]
    drawer = f"attn_L{L - 1}"
    var = float(D.logit.var())
    per_row = {c: {"var_share": float(v.var() / var), "mean": float(v.mean()),
                   "r2_alone": float(1 - (D.logit - v - (D.logit - v).mean()).var() / var)}
               for c, v in comps.items()}
    a, b, keep = top2(D.logit, decs)
    a, b = a[keep], b[keep]
    delta = D.logit[a] - D.logit[b]
    contr = {}
    tot_abs = sum(np.abs(v[a] - v[b]).mean() for v in comps.values())
    for c, v in comps.items():
        dc = v[a] - v[b]
        contr[c] = {"abs_share": float(np.abs(dc).mean() / tot_abs), "sign_agree": float((dc > 0).mean()),
                    "r2": float(1 - ((delta - dc) ** 2).sum() / ((delta - delta.mean()) ** 2).sum()),
                    "mean": float(dc.mean())}
    # the drawer's per-token view of the contrast (last-layer attention terms, head-summed)
    tok = D.attn[:, L - 1].sum(1)                                             # [B, S]
    dtok = tok[a] - tok[b]
    labels = I.seq_labels()
    tok_share = {labels[j]: float(np.abs(dtok[:, j]).mean()) for j in range(tok.shape[1])}
    # chosen (MCTS-played) vs the model's best alternative
    ch = ctx.rows.is_chosen[idx]
    ca, cb = [], []
    for d in decs:
        cc = np.flatnonzero(ch[d.s:d.e])
        if d.e - d.s < 2 or len(cc) != 1:
            continue
        k = d.s + cc[0]
        oth = [i for i in range(d.s, d.e) if i != k]
        ca.append(k); cb.append(oth[int(np.argmax(D.logit[oth]))])
    ca, cb = np.array(ca), np.array(cb)
    dch = D.logit[ca] - D.logit[cb]
    chosen = {c: {"r2": float(1 - ((dch - (v[ca] - v[cb])) ** 2).sum() / ((dch - dch.mean()) ** 2).sum()),
                  "sign_agree": float((np.sign(v[ca] - v[cb]) == np.sign(dch)).mean())}
              for c, v in comps.items()}
    dr = contr[drawer]
    verdict = ("adequate" if dr["r2"] >= THRESH["coverage_adequate_r2"] and dr["sign_agree"] >= THRESH["coverage_adequate_sign"]
               else "minority" if dr["r2"] < THRESH["coverage_minority_r2"] else "partial")
    return {"split": which, "rows": int(len(idx)), "decisions": len(decs),
            "decisions_contrastive": int(keep.sum()), "dropped_size1": int((~np.array([d.e - d.s >= 2 for d in decs])).sum()),
            "max_abs_resid": float(np.abs(D.resid).max()), "frac_q_saturated": float((np.abs(D.q) > 0.99).mean()),
            "logit_var": var, "per_row": per_row, "contrastive": contr, "contrastive_delta_mean": float(delta.mean()),
            "drawer_component": drawer, "drawer_token_abs_contrast": dict(sorted(tok_share.items(), key=lambda kv: -kv[1])),
            "chosen_vs_best_alt": chosen, "verdict_drawer": verdict}


def step_assassin_const(ctx: Ctx, which: str = "A") -> dict:
    idx = ctx.split(which)
    D, decs = ctx.base(idx), ctx.decisions(idx)
    L, H = D.attn.shape[1], D.attn.shape[2]
    multi = [d for d in decs if d.e - d.s >= 2]
    strata = {"all": multi, "threat": [d for d in multi if d.threat], "safe": [d for d in multi if d.safe]}
    a, b, keep = top2(D.logit, decs)
    keep_by = {id(d): (aa, bb) for d, aa, bb, k in zip(decs, a, b, keep) if k}
    out = {"n": {k: len(v) for k, v in strata.items()}, "heads": {}}
    for li in range(L):
        for h in range(H):
            x = D.attn[:, li, h, J]
            rec = {}
            for name, ds in strata.items():
                if not ds:
                    continue
                pos = np.concatenate([np.arange(d.s, d.e) for d in ds])
                within = np.mean([x[d.s:d.e].var() for d in ds])
                eta = within / max(x[pos].var(), 1e-12)
                pa = np.array([keep_by[id(d)][0] for d in ds]); pb = np.array([keep_by[id(d)][1] for d in ds])
                rho = np.abs(x[pa] - x[pb]).mean() / max(np.abs(x[pos]).mean(), 1e-12)
                rec[name] = {"eta": float(eta), "rho": float(rho), "mean": float(x[pos].mean()),
                             "mean_abs": float(np.abs(x[pos]).mean())}
            thr = strata["threat"]
            if thr:
                fl = np.array([d.flip for d in thr])
                bo = []
                for d in thr:
                    oth = [i for i in range(d.s, d.e) if i != d.flip]
                    bo.append(oth[int(np.argmax(D.logit[oth]))])
                dd = x[fl] - x[np.array(bo)]
                rec["threat_flip_minus_best"] = cluster_boot(dd, np.array([d.game for d in thr]))
            out["heads"][f"L{li}h{h}"] = rec
    thr_rhos = [v["threat"]["rho"] for v in out["heads"].values() if "threat" in v]
    out["all_threat_rho_below"] = bool(thr_rhos) and max(thr_rhos) < THRESH["const_rho"]
    out["prediction"] = ("null: Assassin DLA is a near-constant offset in threat decisions"
                         if out["all_threat_rho_below"] else "Assassin DLA varies across candidates")
    return out


def _ptop(attn_last: np.ndarray) -> dict:
    """attn_last [B,H,S] signed last-layer DLA. Rate each non-CLS token is the per-head argmax."""
    x = attn_last[:, :, NONCLS]
    lab = [I.seq_labels()[j] for j in NONCLS]
    res = {}
    for kind, v in (("signed", x), ("abs", np.abs(x))):
        am = v.argmax(-1)                                                    # [B,H]
        anyhead = {lab[t]: float((am == t).any(1).mean()) for t in range(len(lab))}
        perhead = {lab[t]: [float((am[:, h] == t).mean()) for h in range(am.shape[1])] for t in range(len(lab))}
        res[kind] = {"any_head": dict(sorted(anyhead.items(), key=lambda kv: -kv[1])), "per_head": perhead}
    return res


def step_nulls(ctx: Ctx, which: str = "A") -> dict:
    idx = ctx.split(which)
    D = ctx.base(idx)
    L = D.attn.shape[1]
    out = {"trained": _ptop(D.attn[:, L - 1])}
    sub = idx[: min(len(idx), 20_000)]
    for seed in range(3):
        torch.manual_seed(seed)
        m = AttentionModel(ctx.model.cfg).eval()
        out[f"untrained_{seed}"] = _ptop(decomp_rows(m, ctx.rows, sub).attn[:, L - 1])
    lab = I.seq_labels()[J]
    tr = out["trained"]["signed"]["any_head"][lab]
    un = float(np.mean([out[f"untrained_{s}"]["signed"]["any_head"][lab] for s in range(3)]))
    out["assassin"] = {"trained_any_head": tr, "untrained_mean_any_head": un,
                       "ratio_untrained_over_trained": un / tr if tr > 0 else float("nan"),
                       "trained_per_head": out["trained"]["signed"]["per_head"][lab],
                       "artifact": bool(tr > 0 and un / tr >= THRESH["null_ratio"])}
    return out


def _cf_cards(ctx: Ctx, idx: np.ndarray, kind: str) -> Tuple[np.ndarray, np.ndarray]:
    return I.cf_assassin(ctx.rows.cards[idx], kind)


def step_p0(ctx: Ctx, which: str = "A") -> dict:
    idx = ctx.split(which)
    D, decs = ctx.base(idx), ctx.decisions(idx)
    thr = [d for d in decs if d.threat]
    games = np.array([d.game for d in thr])
    F0 = flip_contrast(D.logit, thr)
    out = {"split": which, "threat_decisions": len(thr), "threat_games": int(len(np.unique(games))),
           "P(F>0)_base": float((F0 > 0).mean()) if len(thr) else float("nan"),
           "by_corpus": {CORPUS_NAMES.get(c, str(c)): int(sum(d.corpus == c for d in thr))
                         for c in sorted({d.corpus for d in thr})}}
    for kind in ("lacks", "dead", "cold"):
        cards, ok = _cf_cards(ctx, idx, kind)
        z = logits_rows(ctx.model, ctx.rows, idx, cards=cards)
        F1 = flip_contrast(z, thr)
        out[kind] = {"dF": cluster_boot(F1 - F0, games), "P(F>0)": float((F1 > 0).mean()) if len(thr) else float("nan"),
                     "applicable_rows": int(ok.sum())}
    out["gate_pass"] = bool(len(thr) and out["lacks"]["dF"]["lo"] > 0)
    return out


# --- Phase 2 -------------------------------------------------------------------------------------

def _head_effect(ctx: Ctx, idx, D0: I.Decomp, decs, spec, li, h, means) -> dict:
    D1 = decomp_rows(ctx.model, ctx.rows, idx, spec, means=means)
    dz = D1.logit - D0.logit
    direct = D1.attn_head[:, li, h] - D0.attn_head[:, li, h]
    indirect = dz - direct
    multi = [d for d in decs if d.e - d.s >= 2]
    am0, am1 = argmax_per_decision(D0.logit, multi), argmax_per_decision(D1.logit, multi)
    y = ctx.rows.y[idx]
    best = np.array([d.s + int(np.argmax(y[d.s:d.e])) for d in multi])
    thr, safe = [d for d in decs if d.threat], [d for d in decs if d.safe]
    res = {"dlogit_mean": float(dz.mean()), "dlogit_abs": float(np.abs(dz).mean()),
           "direct_abs": float(np.abs(direct).mean()), "indirect_abs": float(np.abs(indirect).mean()),
           "indirect_share": float(np.abs(indirect).mean() / max(np.abs(dz).mean(), 1e-12)),
           "argmax_flip_rate": float((am0 != am1).mean()),
           "top1_bestq": [float((am0 == best).mean()), float((am1 == best).mean())],
           "mse": [float(((np.tanh(D0.logit) - y) ** 2).mean()), float(((np.tanh(D1.logit) - y) ** 2).mean())]}
    for name, ds in (("threat", thr), ("safe", safe)):
        if ds:
            F0, F1 = flip_contrast(D0.logit, ds), flip_contrast(D1.logit, ds)
            g = np.array([d.game for d in ds])
            res[f"dF_{name}"] = cluster_boot(F1 - F0, g)
            res[f"P(F>0)_{name}"] = [float((F0 > 0).mean()), float((F1 > 0).mean())]
            res[f"_dF_{name}_vals"] = (F1 - F0).tolist()
            res[f"_games_{name}"] = g.tolist()
    if thr and safe:
        res["did_threat_minus_safe"] = did_boot(np.array(res["_dF_threat_vals"]), np.array(res["_games_threat"]),
                                                np.array(res["_dF_safe_vals"]), np.array(res["_games_safe"]))
    for k in [k for k in res if k.startswith("_")]:
        del res[k]
    return res


def step_heads(ctx: Ctx, which: str = "A") -> dict:
    idx = ctx.split(which)
    D0, decs = ctx.base(idx), ctx.decisions(idx)
    means = ctx.means()
    L, H = ctx.model.cfg.n_layers, ctx.model.cfg.n_heads
    params = {"split": which}
    path, units = ctx.partial(f"heads_{which}", params)
    for li in range(L):
        for h in range(H):
            for mode in ("mean", "zero"):
                key = f"L{li}h{h}:{mode}"
                if key in units:
                    continue
                spec = I.InterventionSpec(ablations=(I.HeadAblation(li, h, mode),))
                units[key] = _jsonable(_head_effect(ctx, idx, D0, decs, spec, li, h, means))
                ctx.save_partial(path, f"heads_{which}", params, units)
                print(f"  {key}: |dlogit| {units[key]['dlogit_abs']:.4f}  indirect {units[key]['indirect_share']:.2f}"
                      f"  flips {units[key]['argmax_flip_rate']:.3f}", flush=True)
    did = {k: v["did_threat_minus_safe"]["diff"] for k, v in units.items()
           if k.endswith(":mean") and "did_threat_minus_safe" in v}
    return {"split": which, "units": units, "rank_did_mean": sorted(did, key=lambda k: -did[k])}


def step_cf(ctx: Ctx, which: str = "A") -> dict:
    idx = ctx.split(which)
    D0, decs = ctx.base(idx), ctx.decisions(idx)
    L = D0.attn.shape[1]
    thr_rows = np.zeros(len(idx), bool)
    for d in decs:
        if d.threat:
            thr_rows[d.s:d.e] = True
    labels = I.seq_labels()
    out = {}
    for kind in ("lacks", "dead", "cold"):
        cards, ok = _cf_cards(ctx, idx, kind)
        sel = np.flatnonzero(ok)
        D1 = decomp_rows(ctx.model, ctx.rows, idx[sel], cards=cards[sel])
        T = D1.logit - D0.logit[sel]
        dtok = (D1.attn - D0.attn[sel]).sum(2)                                   # [n, L, S] head-summed
        own = dtok[:, :, J].sum(1)
        other = dtok.sum(1).sum(-1) - own
        ffn = (D1.ffn - D0.ffn[sel]).sum(1)
        den = np.abs(own).mean() + np.abs(other).mean() + np.abs(ffn).mean()
        oth_tok = np.abs(dtok.sum(1)).mean(0)
        oth_tok[J] = 0
        top_other = {labels[j]: float(oth_tok[j]) for j in np.argsort(-oth_tok)[:5]}
        Dd = D0.attn[sel, L - 1, :, J].sum(-1)                                   # the drawer's Assassin bar
        res = {"rows": int(len(sel)), "T_mean": float(T.mean()), "T_abs": float(np.abs(T).mean()),
               "share_own": float(np.abs(own).mean() / den), "share_other_tokens": float(np.abs(other).mean() / den),
               "share_ffn": float(np.abs(ffn).mean() / den), "top_other_tokens_abs": top_other,
               "own_by_layer_abs": [float(np.abs(dtok[:, li, J]).mean()) for li in range(L)],
               "closure": float(np.abs(T - (own + other + ffn)).max())}
        tsel = thr_rows[sel]
        for name, m in (("all", np.ones(len(sel), bool)), ("threat", tsel)):
            if m.sum() >= 3:
                res[f"drawer_predicts_{name}"] = {"spearman(-D,T)": spearman(-Dd[m], T[m]),
                                                  "sign_agree": float((np.sign(-Dd[m]) == np.sign(T[m])).mean()),
                                                  "n": int(m.sum())}
        pt = res.get("drawer_predicts_threat")
        res["drawer_pass"] = bool(pt and pt["spearman(-D,T)"] >= THRESH["cf_spearman"] and pt["sign_agree"] >= THRESH["cf_sign"])
        out[kind] = res
    return out


def step_knockout(ctx: Ctx, which: str = "A") -> dict:
    idx = ctx.split(which)
    D0, decs = ctx.base(idx), ctx.decisions(idx)
    L, H = ctx.model.cfg.n_layers, ctx.model.cfg.n_heads
    thr = [d for d in decs if d.threat]
    multi = [d for d in decs if d.e - d.s >= 2]
    am0 = argmax_per_decision(D0.logit, multi)
    labels = I.seq_labels()
    path, units = ctx.partial(f"knockout_{which}", {"split": which})
    ladders = {"K1": lambda j: (I.AttnKnockout(L - 1, (j,)),),
               "K2": lambda j: tuple(I.AttnKnockout(li, (j,)) for li in range(L)),
               "K3": lambda j: tuple(I.AttnKnockout(li, (j,), queries=None) for li in range(L))}
    k1_rows = {}
    for j in CARD_SEQ:
        for lad, mk in ladders.items():
            key = f"{labels[j]}:{lad}"
            z = None
            if key not in units or (lad == "K1"):
                z = logits_rows(ctx.model, ctx.rows, idx, I.InterventionSpec(knockouts=mk(j)))
            if lad == "K1":
                k1_rows[j] = z - D0.logit
            if key in units:
                continue
            dz = z - D0.logit
            rec = {"dlogit_abs": float(np.abs(dz).mean()), "dlogit_mean": float(dz.mean()),
                   "argmax_flip_rate": float((argmax_per_decision(z, multi) != am0).mean())}
            if thr:
                rec["dF_threat"] = cluster_boot(flip_contrast(z, thr) - flip_contrast(D0.logit, thr),
                                                np.array([d.game for d in thr]))
            if lad == "K1":
                Dj = D0.attn[:, L - 1, :, j].sum(-1)
                rec["spearman(-D,K1)"] = spearman(-Dj, dz)
                rec["sign_agree(-D,K1)"] = float((np.sign(-Dj) == np.sign(dz)).mean())
            units[key] = rec
            ctx.save_partial(path, f"knockout_{which}", {"split": which}, units)
        print(f"  {labels[j]}: K1 {units[f'{labels[j]}:K1']['dlogit_abs']:.4f}  K3 {units[f'{labels[j]}:K3']['dlogit_abs']:.4f}", flush=True)
    # per-head K1 on the Assassin
    for h in range(H):
        key = f"Assassin:K1h{h}"
        if key not in units:
            z = logits_rows(ctx.model, ctx.rows, idx, I.InterventionSpec(knockouts=(I.AttnKnockout(L - 1, (J,), head=h),)))
            units[key] = {"dlogit_abs": float(np.abs(z - D0.logit).mean()), "dlogit_mean": float((z - D0.logit).mean())}
            ctx.save_partial(path, f"knockout_{which}", {"split": which}, units)
    # per-row token-ranking agreement: |DLA_j| vs |K1_j| over the 18 cards
    K = np.stack([k1_rows[j] for j in CARD_SEQ], 1)                                # [B,18]
    Dm = np.abs(D0.attn[:, L - 1].sum(1)[:, CARD_SEQ])                             # [B,18]
    sub = np.random.default_rng(0).choice(len(idx), size=min(5000, len(idx)), replace=False)
    rs = [spearman(Dm[i], np.abs(K[i])) for i in sub]
    top3 = [len(set(np.argsort(-Dm[i])[:3]) & set(np.argsort(-np.abs(K[i]))[:3])) / 3 for i in sub]
    # Assassin rank among all cards and among 5 attention-mass-matched controls
    with torch.no_grad():
        mass = np.zeros(18)
        for s in range(0, len(idx), 4096):
            b = I.batch_from_rows(ctx.rows, idx[s:s + 4096])
            with I.record(ctx.model) as r:
                I._run(ctx.model, b)
            mass += r["attn"][L - 1][:, :, 0, 1:19].mean(1).sum(0).numpy()
        mass /= len(idx)
    k3 = np.array([units[f"{labels[j]}:K3"]["dlogit_abs"] for j in CARD_SEQ])
    k1 = np.array([units[f"{labels[j]}:K1"]["dlogit_abs"] for j in CARD_SEQ])
    a = I.ASSASSIN_SLOT
    matched = [i for i in np.argsort(np.abs(mass - mass[a])) if i != a][:5]
    return {"split": which, "units": units,
            "row_rank_agreement": {"spearman_median": float(np.nanmedian(rs)), "top3_overlap_mean": float(np.mean(top3))},
            "assassin_rank_K1_of18": int((k1 > k1[a]).sum() + 1), "assassin_rank_K3_of18": int((k3 > k3[a]).sum() + 1),
            "matched_controls": [labels[1 + i] for i in matched],
            "assassin_pct_vs_matched_K3": float((k3[matched] < k3[a]).mean()),
            "mean_cls_mass_last_layer": {labels[1 + i]: float(mass[i]) for i in range(18)}}


def step_patch(ctx: Ctx, which: str = "A") -> dict:
    idx = ctx.split(which)
    D0, decs = ctx.base(idx), ctx.decisions(idx)
    L = D0.attn.shape[1]
    r = ctx.rows
    dead = r.cards[ctx.tri, I.ASSASSIN_SLOT, I.DISCARD_COL] >= 1.0 - 1e-6
    donors = ctx.tri[dead]
    donors = np.sort(np.random.default_rng(0).choice(donors, size=min(20_000, len(donors)), replace=False))
    res_bank = []
    for s in range(0, len(donors), 4096):
        with I.record(ctx.model) as rec:
            I._run(ctx.model, I.batch_from_rows(r, donors[s:s + 4096]))
        res_bank.append(rec["resid_in"][L - 1][:, J].clone())
    bank = torch.cat(res_bank)
    key = lambda ii: r.phase[ii].argmax(-1) * 100 + np.round(r.board[ii, 0] * 10).astype(int)
    dkey = key(donors)
    rng = np.random.default_rng(1)
    target = I.assassin_unseen_in_their_hand(r.cards[idx])
    vkey = key(idx)
    pick = np.empty(len(idx), np.int64)
    by = {}
    for i, k in enumerate(dkey):
        by.setdefault(k, []).append(i)
    allk = np.arange(len(donors))
    for i, k in enumerate(vkey):
        pool = by.get(k)
        pick[i] = rng.choice(pool) if pool else rng.choice(allk)

    def patch_fn(s, e):
        with torch.no_grad(), I.record(ctx.model) as rec:
            I._run(ctx.model, I.batch_from_rows(r, idx[s:e]))
        cur = rec["resid_in"][L - 1][:, J].clone()
        m = torch.from_numpy(target[s:e])
        cur[m] = bank[torch.from_numpy(pick[s:e])][m]
        return (I.ResidualPatch(L - 1, (J,)), cur[:, None, :])

    z = logits_rows(ctx.model, r, idx, patch_fn=patch_fn, batch=2048)
    dz = (z - D0.logit)[target]
    Dj = D0.attn[target, L - 1, :, J].sum(-1)
    thr = [d for d in decs if d.threat]
    out = {"split": which, "patched_rows": int(target.sum()), "donor_pool": int(len(donors)),
           "dlogit_mean": float(dz.mean()), "dlogit_abs": float(np.abs(dz).mean()),
           "spearman(-D,dlogit)": spearman(-Dj, dz), "sign_agree": float((np.sign(-Dj) == np.sign(dz)).mean())}
    if thr:
        out["dF_threat"] = cluster_boot(flip_contrast(z, thr) - flip_contrast(D0.logit, thr), np.array([d.game for d in thr]))
    return out


def _reinit(model, stage: str, seed: int):
    """Cascade (top-down): head -> layers[L-1] -> ... -> layers[0] -> embed+cls, re-initialised from a fresh net."""
    torch.manual_seed(seed)
    fresh = AttentionModel(model.cfg)
    m = copy.deepcopy(model)
    L = model.cfg.n_layers
    order = ["head"] + [f"layers.{li}" for li in range(L - 1, -1, -1)] + ["embed"]
    upto = order[: order.index(stage) + 1]
    sd, fsd = m.state_dict(), fresh.state_dict()
    for k in sd:
        if any(k == p or k.startswith(p + ".") for p in upto) or (stage == "embed" and k == "cls"):
            sd[k] = fsd[k]
    m.load_state_dict(sd)
    return m.eval(), order


def step_sanity(ctx: Ctx, which: str = "A") -> dict:
    idx = ctx.split(which)
    idx = idx[: min(len(idx), 5000)]
    L, H = ctx.model.cfg.n_layers, ctx.model.cfg.n_heads
    D0 = decomp_rows(ctx.model, ctx.rows, idx)
    base = D0.attn[:, L - 1][:, :, NONCLS]
    path, units = ctx.partial(f"sanity_{which}", {"split": which, "n": int(len(idx))})
    _, order = _reinit(ctx.model, "head", 0)
    for stage in order:
        if stage in units:
            continue
        m, _ = _reinit(ctx.model, stage, 1234)
        D1 = decomp_rows(m, ctx.rows, idx)
        x = D1.attn[:, L - 1][:, :, NONCLS]
        # |rank corr| (Adebayo et al.): a sign-flipped map is still the SAME map -- re-initialising only the
        # readout flips u's sign and would otherwise "pass" with a large negative correlation.
        per_head = [float(np.nanmedian([abs(spearman(base[i, h], x[i, h])) for i in range(len(idx))]))
                    for h in range(H)]
        units[stage] = {"median_abs_spearman_per_head": per_head, "median_spearman": float(np.median(per_head)),
                        "ptop_assassin": _ptop(D1.attn[:, L - 1])["signed"]["any_head"][I.seq_labels()[J]]}
        ctx.save_partial(path, f"sanity_{which}", {"split": which, "n": int(len(idx))}, units)
        print(f"  reinit {stage}: median spearman {units[stage]['median_spearman']:.3f}", flush=True)
    key = f"layers.{L - 1}"
    return {"split": which, "rows": int(len(idx)), "stages": units, "order": order,
            "pass": bool(units[key]["median_spearman"] < THRESH["sanity_spearman"])}


# --- Phase 3 -------------------------------------------------------------------------------------

def l0_assassin_mass(ctx: Ctx, idx) -> List[float]:
    """Layer-0 attention mass on the (unseen) Assassin from the 18 card query rows, per head."""
    H = ctx.model.cfg.n_heads
    tot, n = np.zeros(H), 0
    unseen = I.assassin_unseen_in_their_hand(ctx.rows.cards[idx])
    sel = idx[unseen]
    for s in range(0, len(sel), 4096):
        with torch.no_grad(), I.record(ctx.model) as r:
            I._run(ctx.model, I.batch_from_rows(ctx.rows, sel[s:s + 4096]))
        a = r["attn"][0][:, :, 1:19, J]                                             # [B,H,18]
        tot += a.mean(-1).sum(0).numpy()
        n += a.shape[0]
    return (tot / max(n, 1)).tolist()


def select(ctx: Ctx) -> dict:
    res = {name: ctx.step(name, {"split": "A"}, lambda f=f: f(ctx, "A"))
           for name, f in (("nulls", step_nulls), ("assassin_const", step_assassin_const))}
    L, H = ctx.model.cfg.n_layers, ctx.model.cfg.n_heads
    lab = I.seq_labels()[J]
    ph = res["nulls"]["trained"]["signed"]["per_head"][lab]
    h_obs = int(np.argmax(ph))
    con = {h: res["assassin_const"]["heads"][f"L{L - 1}h{h}"].get("threat_flip_minus_best", {}).get("mean", np.nan)
           for h in range(H)}
    h_con = int(min(con, key=lambda h: con[h] if con[h] == con[h] else np.inf))
    l0 = l0_assassin_mass(ctx, ctx.split("A"))
    h_l0 = int(np.argmax(l0))
    heads = {"H_obs": [L - 1, h_obs], "H_con": [L - 1, h_con], "H_L0": [0, h_l0]}
    if H == 4:                                                       # README "head 4" (1-based) on the d64 drawer
        heads["README_head4_L0"] = [0, 3]
        heads["README_head4_L1"] = [L - 1, 3]
    uniq = {}
    for k, v in heads.items():
        uniq.setdefault(tuple(v), []).append(k)
    prereg = {"tag": ctx.tag, "ckpt": ctx.ckpt, "ckpt_sha256": ctx.ckpt_sha, "npz": ctx.npz, "npz_sha256": ctx.npz_sha,
              "means_sha256": ctx.means_sha(), "git_head": git_head(), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
              "selection_evidence": {"ptop_assassin_per_L_last_head": ph, "flip_minus_best_assassin_dla_per_L_last_head": con,
                                     "l0_assassin_mass_per_head": l0},
              "heads": heads, "targets": [{"layer": k[0], "head": k[1], "names": v} for k, v in uniq.items()],
              "thresholds": THRESH,
              "predictions": {
                  "P1": "mean-ablating the head raises P(F>0) in the threat stratum by >= 2pp (val-B)",
                  "P2": "dF_threat - dF_safe > 0 (game-bootstrap CI excludes 0, Holm) AND the head ranks 1st on it",
                  "P3": "E[dF ablation] / E[dF lacks-CF] >= 0.25 in the threat stratum",
                  "G": "greedy games: paired d(assassin_death) > 0 at Holm p<.05, largest of all heads, specific; "
                       "falsified if the 95% CI upper bound < +2pp",
                  "null_from_1b": res["assassin_const"]["prediction"]},
              "games": {"deals": None, "base_seed": 90_000_000, "opponent": "intact copy (greedy)", "mode": "mean"}}
    path = os.path.join(ctx.out, "prereg.json")
    if os.path.exists(path) and "select" not in ctx.force:
        print(f"{path} exists -- not overwriting a pre-registration (use --force select)")
        with open(path) as f:
            return json.load(f)
    atomic_json(path, prereg)
    print(f"wrote {path} -- COMMIT IT before running `confirm` or the games")
    return prereg


def load_prereg(ctx: Ctx) -> dict:
    path = os.path.join(ctx.out, "prereg.json")
    with open(path) as f:
        pre = json.load(f)
    if pre["ckpt_sha256"] != ctx.ckpt_sha or pre["npz_sha256"] != ctx.npz_sha or pre["means_sha256"] != ctx.means_sha():
        raise SystemExit("prereg.json does not match this checkpoint/dataset/means -- refusing to run")
    return pre


def confirm(ctx: Ctx) -> dict:
    pre = load_prereg(ctx)

    def run():
        idx = ctx.split("B")
        D0, decs = ctx.base(idx), ctx.decisions(idx)
        thr = [d for d in decs if d.threat]
        g = np.array([d.game for d in thr])
        F0 = flip_contrast(D0.logit, thr)
        z_cf = logits_rows(ctx.model, ctx.rows, idx, cards=_cf_cards(ctx, idx, "lacks")[0])
        dF_cf = flip_contrast(z_cf, thr) - F0
        L, H = ctx.model.cfg.n_layers, ctx.model.cfg.n_heads
        means = ctx.means()
        allh = {}
        for li in range(L):
            for h in range(H):
                allh[(li, h)] = _head_effect(ctx, idx, D0, decs,
                                             I.InterventionSpec(ablations=(I.HeadAblation(li, h, "mean"),)), li, h, means)
        did_rank = sorted(allh, key=lambda k: -allh[k].get("did_threat_minus_safe", {}).get("diff", -np.inf))
        out, pv = {"split": "B", "threat_decisions": len(thr), "dF_lacks": cluster_boot(dF_cf, g), "targets": {}}, {}
        for t in pre["targets"]:
            k = (t["layer"], t["head"])
            e = allh[k]
            name = f"L{k[0]}h{k[1]}"
            p1 = e[f"P(F>0)_threat"][1] - e[f"P(F>0)_threat"][0] if "P(F>0)_threat" in e else float("nan")
            did = e.get("did_threat_minus_safe", {})
            rec = {"names": t["names"], "P1_dP(F>0)": p1, "P1_pass": bool(p1 >= THRESH["P1_pp"]),
                   "P2_did": did, "P2_rank": did_rank.index(k) + 1,
                   "P3_recovery": e["dF_threat"]["mean"] / dF_cf.mean() if len(thr) and dF_cf.mean() != 0 else float("nan"),
                   "effect": e}
            rec["P3_pass"] = bool(rec["P3_recovery"] >= THRESH["P3_recovery"])
            pv[name] = did.get("p_le0", 1.0)
            out["targets"][name] = rec
        rej = holm(pv, THRESH["alpha"])
        for name, rec in out["targets"].items():
            rec["P2_pass"] = bool(rej[name] and rec["P2_rank"] == 1)
        out["did_rank_all_heads"] = [f"L{a}h{b}" for a, b in did_rank]
        return out
    return ctx.step("confirm", {"split": "B", "prereg_sha": sha256(os.path.join(ctx.out, "prereg.json"))}, run)


# --- driver --------------------------------------------------------------------------------------

PHASE1 = (("coverage", step_coverage), ("assassin_const", step_assassin_const), ("nulls", step_nulls), ("p0", step_p0))
PHASE2 = (("heads", step_heads), ("cf", step_cf), ("knockout", step_knockout), ("patch", step_patch),
          ("sanity", step_sanity))


def run_steps(ctx: Ctx, steps, which: str = "A") -> Dict[str, dict]:
    out = {}
    for name, f in steps:
        out[name] = ctx.step(name if which == "A" else f"{name}_{which}", {"split": which}, lambda f=f: f(ctx, which))
    return out


def status(ctx: Ctx) -> None:
    names = [n for n, _ in PHASE1 + PHASE2] + ["prereg", "confirm"]
    print(f"[{ctx.tag}] {ctx.out}")
    for n in ["means.pt"] + names:
        p = os.path.join(ctx.out, n if n.endswith(".pt") else f"{n}.json")
        part = os.path.join(ctx.out, f"{n}_A.partial.json")
        st = "done" if os.path.exists(p) else "partial" if os.path.exists(part) else "missing"
        print(f"  {n:16s} {st}")
    gdir = os.path.join(ctx.out, "games")
    if os.path.isdir(gdir):
        for f in sorted(os.listdir(gdir)):
            if f.endswith(".jsonl"):
                with open(os.path.join(gdir, f)) as fh:
                    print(f"  games/{f:24s} {sum(1 for _ in fh)} rows")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Study 10: DLA faithfulness battery (resumable).")
    p.add_argument("cmd", choices=["all", "phase1", "phase2", "coverage", "select", "confirm", "status"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--npz", default=None, help="token tensors (default: the checkpoint's training npz)")
    p.add_argument("--tag", default=None)
    p.add_argument("--out-root", default=os.path.join("reports", "dla_faithfulness"))
    p.add_argument("--force", nargs="*", default=[])
    p.add_argument("--max-rows", type=int, default=None, help="cap analysed rows (smoke tests)")
    a = p.parse_args(argv)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    ctx = Ctx(a.ckpt, a.npz, a.tag, a.out_root, a.force, a.max_rows)
    print(f"[{ctx.tag}] rows {len(ctx.rows)}  train {len(ctx.tri)}  val-A {len(ctx.val_a)}  val-B {len(ctx.val_b)}")
    if a.cmd == "status":
        status(ctx)
    elif a.cmd == "coverage":
        run_steps(ctx, PHASE1[:1])
    elif a.cmd == "phase1":
        run_steps(ctx, PHASE1)
    elif a.cmd == "phase2":
        run_steps(ctx, PHASE2)
    elif a.cmd == "all":
        run_steps(ctx, PHASE1 + PHASE2)
    elif a.cmd == "select":
        select(ctx)
    elif a.cmd == "confirm":
        confirm(ctx)


if __name__ == "__main__":
    main()
