# Attention Q-Net — Exploration & Experiment Log

Results for the explainable attention q-net (`attention_model.py`) built on the `explainability` branch:
a token-based, action-in q-model (`f(state, action) → q`) with per-head CLS→token attention as the
interpretability readout. This file records the two studies run to (1) profile its cost as an MCTS leaf
evaluator and (2) sweep its capacity (depth `L`, FFN width, and the head-to-head playing strength each buys).

## Setup (common to all results)

- **Game:** 2-player ImposterKings.
- **Dataset:** `datasets/tensors/k20l3_tokens.npz` — **240,499 token rows** post-processed from ~2,000
  self-play games (log-once/derive-many; one row per *(state, candidate action)*). Rows are **game-split**
  into train/val (val_frac 0.1) to avoid leakage; loss is **visit-share-weighted MSE** on the mover-relative
  q label. Note rows are heavily correlated (few independent games), so effective samples ≪ 240k.
- **Model default:** `AttnConfig(d_model=64, n_heads=4, ffn_hidden=128, n_layers=1, dropout=0.1, bounded=True)`.
  Explicit-action encoding ("prong B", `ACTION_DIM=51`); `CARD_DIM=43/44`. Trained with Adam, batch 1024,
  lr 1e-3, up to 50 epochs, early stopping patience 5.
- **Head-to-head protocol:** the net is used as the **MCTS leaf evaluator + policy prior** at budget
  `hybrid(20,3)` ("@k20"), played against a **vanilla-MCTS@k20** opponent (pure random rollouts) over
  **100 mirrored deals = 200 games** (each deal played twice with swapped seats; independent per-orientation
  play-rng). Parallelized over 10 workers. **Winrate is a lower-bound-style signal vs one specific opponent,
  not an exploitability bound** — >50% means "the net's guidance beats raw rollouts at k20."

---

## Study 1 — Profiling the NN-MCTS hot path

10 games each of `<challenger> vs vanilla-MCTS@k20`, serial single-thread under cProfile (`torch` at 1 thread).

| config | wall (10 games) | leaf-eval cost | rollout cost | NN eval vs a rollout |
|---|---:|---:|---:|---|
| **no-NN** (rollout both sides) | 273.5 s | — (rollout *is* the leaf) | 0.89 ms/rollout | — |
| **MLP-MCTS** | **215.0 s** | **0.41 ms**/eval | 0.93 ms/rollout | **2.3× cheaper** than a rollout |
| **attention-MCTS** | **468.5 s** | **3.24 ms**/eval | 0.98 ms/rollout | **3.3× more expensive** than a rollout |

**Leaf-work decomposition** (the leaf value comes from a *rollout* on the vanilla side and from the *net* on
the challenger side; NN-MCTS does **not** roll out — the eval replaces the rollout):

| config | vanilla side (rollouts) | challenger side (leaf method) | total leaf work |
|---|---|---|---:|
| no-NN | 187,584 × 0.89 ms = 166 s | *also rollouts* (both sides) | **166 s** |
| MLP-MCTS | 86,700 × 0.93 ms = 81 s | 75,122 evals × 0.41 ms = 31 s | **112 s** |
| attention-MCTS | 86,820 × 0.98 ms = 85 s | 83,320 evals × 3.24 ms = 270 s | **355 s** |

**Attention eval breakdown** (`_leaf_value` = 274 s of 451 s profiled = **61% of the whole run**), split ≈50/50:
- **`tokenize` — 117 s (43% of eval):** Python featurization *per candidate move* — `_card_token` 20 s,
  `state.with_` 29 s, **`determinize` 27 s over 536,508 calls**.
- **torch `forward` — 115 s (43% of eval):** `nn.linear` dispatch 30 s, **`collate` 21 s**, `layer_norm` 8.5 s.
- **Redundant determinize confirmed:** ~**6.4 determinize/leaf** (attention) vs ~**2/leaf** (MLP) — the
  evaluate re-tokenizes every legal move and each `tokenize` re-runs `legal_moves → determinize`.

**Verdict:** swapping a rollout for the **MLP** eval *speeds the game up* (215 s < 273 s) — the numpy eval is
cheaper than the playout it replaces. The **attention** eval is ~3× a rollout, so attention-MCTS is ~1.7×
slower than pure rollout. The ~7× per-leaf gap over MLP is ~50% Python featurization overhead (redundant
determinize + per-move tokenize + collate) and ~50% torch dispatch — recoverable with a numpy forward +
determinize-once-per-leaf (out of scope; now quantified).

---

## Study 2 — Capacity sweep (depth L, FFN width) + head-to-head strength

All variants trained on the same `k20l3_tokens.npz` (no rebuild). **Winrate = vs vanilla-MCTS@k20, 100
mirrored deals (200 games)**, reported as **winrate (± 95% CI)**.

| variant | config (d/L/ffn) | params | train-time | epochs | val_mse | top1_bestq | recall@2 | spearman | **winrate (± 95% CI)** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **v2** (deploy) | 64 / **1** / 128 | 42,945 | 2018 s | 50 | 0.0647 | 51.3% | 75.8% | 0.434 | **57.5% (± 5.6%)** |
| **v3a** | 64 / **2** / 128 | 78,209 | 2331 s | 19 (best 14) | 0.0646 | 51.7% | 76.3% | 0.437 | **55.5% (± 5.2%)** |
| **v3b** | 64 / 1 / **256** | 61,249 | 643 s | 9 (best 4) | 0.0756 | 48.3% | 74.5% | 0.388 | **51.0% (± 5.9%)** |
| **v3c** | 64 / **2** / **256** | 111,233 | 2088 s | 17 (best 12) | **0.0628** | 51.2% | **76.6%** | 0.422 | **62.0% (± 5.4%)** |
| **MLP-256** (baseline) | mlp 256-wide | 55,809 | n/a | n/a | 0.0704 | **53.4%** | 76.8% | 0.464 | **57.0% (± 5.7%)** |

Raw head-to-head detail (wins / 200, split-rate = fraction of deals split 1–1, CPU-seconds over 10 workers):

| variant | wins | winrate | ci95 | split% | CPU-s |
|---|---:|---:|---:|---:|---:|
| v2 (attention) | 115/200 | 0.575 | ±0.056 | 65% | 22,662 |
| v3a | 111/200 | 0.555 | ±0.052 | 71% | 23,248 |
| v3b | 102/200 | 0.510 | ±0.059 | 64% | 21,228 |
| v3c | 124/200 | 0.620 | ±0.054 | 64% | 25,147 |
| MLP-256 | 114/200 | 0.570 | ±0.057 | 64% | 7,504 |

### Findings

1. **The capacity knobs interact.** Depth alone (v3a) doesn't help; **wide-FFN alone (v3b) hurts** (worst on
   both proxies and play, 51.0%); but **depth + width together (v3c) is best** on val_mse *and* winrate
   (62.0%). So "more capacity" only paid off when both moved together.
2. **v3c vs v2 is suggestive, not conclusive:** +4.5 pp point estimate but the CIs overlap
   ([56.6, 67.4] vs [51.9, 63.1]). v3c *does* clear v3b cleanly. A confirmation run (more deals, or a direct
   v3c-vs-v2 match) would settle it.
3. **Proxies (val_mse/top1/recall) barely separate the L2 variants** (~0.063 val_mse, ~51% top1, ~76%
   recall) yet winrate spread is real (51%→62%) — proxies and playing strength don't always agree
   (MLP-256 has the *highest* top1 at 53.4% but only mid-pack winrate).
4. **Overfitting, not underfitting:** the train/val gap widens with capacity (v2 +0.0007 → v3a +0.0062 →
   v3c +0.0071 → v3b diverges early), consistent with ~240k *correlated* rows (few independent games).
5. **Cost:** attention plays at MLP-equivalent strength but at ~3× the CPU (22.7k vs 7.5k CPU-s for the same
   200 games) — the price of the per-leaf torch forward (Study 1).

### Deploy decision

**v2 `models/attn_d64_L1.pt` (L1)** is the deployed hint model for the explainability UI — simplest, proven,
cheapest eval. At L1 only the **CLS row (row 0)** causally feeds q (card↔card rows are computed-but-discarded),
so the visualization renders row-0 primary. **v3c (L2)** is the standby upgrade: strongest *and* it makes the
card↔card attention rows causal (layer-1 → layer-2 CLS), so swapping to it later buys richer explanations at
(likely) no strength cost — pending a confirmation run.

---

## Metric glossary

- **winrate** — fraction of games the challenger wins vs vanilla-MCTS@k20 (mirrored; 200 games). ± is the
  **95% confidence interval** (`1.96·s/√n` over per-deal scores); overlapping CIs ⇒ difference not significant.
- **top1_bestq** — fraction of decisions where the net's argmax-q move equals the search's best (highest-visit)
  move. **recall@2** — fraction where the search's best move is in the net's top-2 by q. **spearman** — rank
  correlation between the net's q ordering and the search's visit ordering over legal moves.
- **val_mse** — visit-share-weighted MSE of the predicted q on the held-out (game-split) validation set;
  baseline (predict-the-mean) ≈ 0.258.
- **params** — total learnable parameters. **train-time** — wall-clock to train (CPU, torch 2.12.1+cpu).
- **eval cost / rollout cost** (Study 1) — mean ms per NN leaf evaluation vs per random rollout playout.
