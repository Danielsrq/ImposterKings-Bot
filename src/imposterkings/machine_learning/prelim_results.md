# Preliminary ML results

Sections 1–3 are **iteration 0** — one supervised pass on the **`k20l3` corpus** (2000 self-play games at
`hybrid-k20-l3` rollout MCTS, ~240k `(state, action)` rows, target = MCTS `mean_q`), *before* any AlphaZero
bootstrapping; section 4 is the first 2-iteration bootstrapping run. Win-rates are 100 deals ×2 mirrored
(200 games), paired seeds, ±95% CI, unless stated otherwise.

## 1. Architecture sweep (training)

MLP `q`-regressor. Baseline (constant-mean) val MSE = **0.2577**. `gap` = val − train MSE (overfit signal);
`rec@2` = true-best move in the model's top-2; `spear` = per-decision Spearman(pred, q).

| arch | params | trn_mse | val_mse | gap | top-1 | rec@2 | spearman |
|---|---|---|---|---|---|---|---|
| 16 | 3.5k | 0.0787 | 0.0805 | 0.0019 | 47.6% | 72.9% | 0.390 |
| 32 | 7.0k | 0.0597 | 0.0713 | 0.0116 | 50.0% | 74.8% | 0.415 |
| 64 | 14k | 0.0615 | 0.0744 | 0.0129 | 51.5% | 74.6% | 0.430 |
| 128 | 28k | 0.0624 | 0.0748 | 0.0124 | 51.2% | 76.1% | 0.446 |
| **256** | 56k | 0.0537 | **0.0704** | 0.0167 | **53.4%** | 76.8% | **0.464** |
| 128-64 | 36k | 0.0476 | 0.0712 | 0.0237 | 53.3% | **77.1%** | 0.455 |
| 256-128 | 89k | 0.0574 | 0.0739 | 0.0165 | 50.9% | 75.9% | 0.424 |
| 64-32 | 16k | 0.0624 | 0.0768 | 0.0144 | 50.9% | 74.8% | 0.413 |
| 32-16 | 7.5k | 0.0658 | 0.0766 | 0.0108 | 49.0% | 74.2% | 0.416 |

- **Wide single-256 is best** (val MSE 0.0704, Spearman 0.464); **depth doesn't help** (256-128 regresses).
- **No overfitting** — gaps ≤ 0.024 vs the 0.258 baseline; the plain MLP is capacity/feature-limited
  (val MSE plateaus ~0.07 regardless of size), not overfit. Attention is the lever past this wall.
- Capacity helps *ranking* more than MSE (Spearman 0.390 → 0.464 across 16 → 256).

## 2. Standalone greedy-NN (mlp_32) vs MCTS

Greedy over predicted `q` (no search). `s/game` = single-core compute (both agents' work per game).

| opponent | win% | s/game |
|---|---|---|
| hybrid-k20 (training strength) | 30.0% ± 5.7 | 6.99 |
| fixed N=500 (≈ hybrid-k10) | 33.5% ± 5.6 | 3.44 |
| fixed N=300 | 39.0% ± 5.5 | 1.98 |
| hybrid-k30 | 30.0% ± 5.0 | ~10.2 |
| MLP self-play (mlp_32 vs mlp_32) | 50.0% | **0.011** |

Greedy play distills to **~below-k10 strength** (loses to k20 at 30%) — the expected greedy-over-learned-`q`
gap (~50% top-1 agreement). Win% correctly tracks opponent strength. But a pure-NN game is **~0.011 s**
vs ~2–7 s vs MCTS — hundreds× cheaper, which is what makes the eval-head use compelling.

## 3. NN-MCTS (mlp_32 as PUCT value + policy head) vs rollout MCTS

The net wired into ISMCTS as an eval/policy head (`benchmark --nn-mcts <budget>`). Baselines = the same
plain MCTS-vs-MCTS matchups.

| NN-MCTS (head) | vs | NN-MCTS win% | plain-MCTS baseline | **NN lift** |
|---|---|---|---|---|
| @500 | MCTS@500 | 54.0% ± 5.0 | 50% (parity) | **+4** |
| @100 | MCTS@500 | 45.5% ± 5.4 | 33.0% | **+12.5** |
| @k20 | MCTS@k50 | 48.5% ± 5.8 | 38.5% | **+10** |

- Even the weak iteration-0 net lifts under-budgeted search **~+4 to +12 pts**, biggest at the lowest
  relative budget — bringing a k20 / 100-sim search to **near-parity with a stronger (k50 / 500-sim)
  opponent**. This is the "each simulation is worth more" premise, confirmed.
- NN-MCTS is also **~3× cheaper per sim** (numpy leaf eval vs a random rollout): ~0.08 s/decision @500
  vs ~0.25 s for rollout MCTS@500.
- **Green light for the AlphaZero loop** (`machine_learning/loop.py`) — the net helps before any
  bootstrapping; each loop iteration should compound it.

## 4. AlphaZero bootstrapping loop — 2 iterations (`runs/az_k20l3`)

`machine_learning.loop`: NN-MCTS **hybrid-k20-l3** self-play, 2000 games/iter, arch sweep {32,64,128}
selected by top-1, eval = NN-MCTS(best) vs **hybrid-k50-l3** (50 deals ×2). Total wall 3640 s (~61 min,
~30 min/iter; generation is the long pole).

| stage | net (PUCT head) | top-1 | val_mse | vs hybrid-k50 |
|---|---|---|---|---|
| iter 0 (seed) | mlp_32 | 50.0% | 0.071 | 48.5% |
| iter 1 | 64 | 53.8% | 0.046 | **52.0%** |
| iter 2 | 128 | 49.9% | 0.052 | **48.0%** |

**Iter 1 is a real gain** (+3.5 pts, above parity vs the stronger k50 reference) — the loop mechanism works.
**Iter 2 regressed** (52.0 → 48.0; top-1 and val_mse moved the same way, so not pure eval noise — though at
50 deals the CI is ~±10%, trust the direction more than the magnitude). Not self-stabilizing as configured.

Suspected causes, in order of confidence:
1. **No replay buffer** — each iter trains ONLY on that iter's 2000 games; greedy PUCT self-play narrows the
   state distribution, losing the rollout corpus's coverage.
2. **k20 search too weak as an improvement operator** — bootstrapping only ratchets if search beats the raw
   net; at k20 the search may just re-express the net's own value (fixed point, no climb). Strength knee is
   ~k40.
3. **Bias replaced variance** — rollout targets were noisy but unbiased; NN-value targets are smooth but
   biased, and bias feeds its own training set. (The val_mse drop 0.071 → 0.046 is partly "easier targets",
   not "more correct".)
4. **Selection/capacity churn** — best-by-top-1 is a ranking proxy, not playing strength, and the selected
   arch changed each round (64 → 128). Iter-2's best top-1 (49.9%) already trailed iter-1's (53.8%) before
   benchmarking — a leading indicator of less-informative data.

Planned fixes (priority order): **(1) replay buffer** — train on a sliding window of the last 2–3 generations
incl. the iter-0 rollout corpus; **(2) raise self-play to ~k30–40**; **(3) fix the arch** (e.g. 64) and select
by a small head-to-head play-eval, not top-1; **(4) eval on ~200 deals** (±5 not ±10); **(5) run 4–6
iterations** to see the curve. Note: pure-`z` value grounding is a weak anchor here (z is ~75% deal-noise) —
keep `q` targets, fix the loop structure first.

## Reference: MCTS budget scaling (rollout, no NN)

The baselines the NN-MCTS lifts are drawn from the `budget_scaling` sweep (100 deals, mirrored):
win% vs **fixed500**: k10 48%, k20 52.5%, k30 57.5%, k40 60.5%, k50 58.5%; win% vs **hybrid-k50**:
k20 38.5%, k30 46.5%, k40 52.5%, k70 52.5% (hi=8192), k100 55.0% (hi=8192). ~75% of games are
deal-decided (seat variance floor), so mirrored+paired seeds are essential; hybrid-k10 ≈ MCTS@500.

_See `imposterkings-scaling-findings` (project memory) for the full record; regenerate any table with
`train`, `benchmark --nn-mcts …`, `budget_scaling`._
