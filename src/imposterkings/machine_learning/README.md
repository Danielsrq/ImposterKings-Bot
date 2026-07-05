# `imposterkings.machine_learning`

Learn a move-evaluator from the self-play corpus. Imports the engine and the corpus produced by
`imposterkings.data_analysis.datagen`; the engine never imports back. `features` / `dataset` are
numpy-only (no torch at import); `mlp` / `train` need PyTorch (`pip install -e .[ml]`).

## Pipeline

```bash
# 1) corpus (JSONL) -> training tensors (state,action -> q)
python -m imposterkings.machine_learning.dataset --data datasets/selfplay_k20l3 --out datasets/tensors/k20l3.npz

# 2) train an architecture sweep
python -m imposterkings.machine_learning.train --npz datasets/tensors/k20l3.npz --sweep "16;32;64"
python -m imposterkings.machine_learning.train --npz ... --sweep "16,16;32,32;64,64"   # multi-layer
```

## What it learns — `q`, not `z`

The output neuron predicts the **MCTS action-value `q`** (`mean_q` of that action). `z` (game outcome) is a
**state** value — identical for every candidate of a decision — so it can't rank *actions*; `q` differs per
action and is the whole point of a move-evaluator. (`z` is kept in the npz for later `λ·z+(1−λ)·q` blend
experiments.) Loss is MSE **weighted by `visit_share`** so well-searched (reliable) `q`'s dominate; the
train/val split is **by game** (rows within a game are correlated).

## Components

| module | needs torch | what |
|---|---|---|
| `features.py` | no | `encode(view, action) -> float32[FEATURE_DIM=216]` — the shared featurizer (info-set based; bag-of-located-cards + globals + action). `feature_names()` for interpretability. |
| `dataset.py`  | no | replay corpus → `.npz` (`X`, `y`=q, `w`=visit_share, `z`, `game_id`, `decision_id`, `is_chosen`) + `.meta.json`. |
| `mlp.py`      | yes | `MLP(in_dim, hidden_dims, dropout)` — any shape (`[16]`, `[16,16]`, `[32,32,64]`, `[]`=linear); `Tanh` output bounds to `q`'s [-1,1]. |
| `train.py`    | yes | game-split, weighted-MSE training w/ early stopping, the `--sweep`, and metrics; saves `mlp_<arch>.pt` + `sweep_results.csv` to `--out-dir` (default `models/`). |

## Metrics

- **Regression:** val MSE / MAE on `q`, vs a constant-`mean(q)` baseline MSE.
- **Ranking** (grouped by `decision_id`): per-decision **top-1 agreement** — does `argmax` predicted value
  pick (`top1_bestq`) the highest-`q` candidate and (`top1_chosen`) the actually-played move — plus mean
  per-decision Spearman(pred, q).

Reference run (k20l3, 2000 games ≈ 240k rows): all archs beat the 0.26 baseline (~0.07–0.08 val MSE); the
`32`-unit model is the MSE sweet spot, `64` ranks marginally better, and a 2-layer `32,32` doesn't beat
single-layer — consistent with the small, largely combinatorial state (the relational structure is what
the future attention model is meant to capture; both share `features.py`).

## Deferred
MLPAgent (greedy over predicted `q`) + win-rate benchmark vs MCTS; the attention model (token adapter of
the same featurizer); target blends and belief features.
