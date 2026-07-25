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
action and is the whole point of a move-evaluator. (`z` is kept in the npz for the `λ·z+(1−λ)·q` blend
experiment below.) Loss is MSE **weighted by `visit_share`** so well-searched (reliable) `q`'s dominate; the
train/val split is **by game** (rows within a game are correlated).

### Verdict on the `z` blend (2026-07)

We ran the controlled A/B: gen2.5's exact 6k-game tensor, one net at **λ=0** (pure `q`, the deployed gen2.5)
vs one at **λ=0.5** (`models/v3c_gen25_z50`) — only λ differs. Against a fixed hybrid-k20-l3 opponent the two
were a wash (63.0% vs 62.5%), so the decider was a **direct head-to-head, 200 mirrored deals**: pure-`q` beat
the blend **56.2% vs 43.8%** (Wilson CI [52.3, 60.1], excludes 50). **Keep λ=0.** Two structural reasons it
underperforms here, independent of tuning:

- **This is an action-in model, so `z` can't teach it to rank.** Each row is `(state, action)` and the value
  head *is* the Q head (policy = `softmax(Q)` at inference). But `z` is the *same* number for every candidate
  action of a decision — feeding the action in doesn't change the target. Blending `z` in therefore pulls all
  of a decision's candidate values toward one shared outcome, **flattening the within-decision spread the
  policy is built on**. It recalibrates value *levels*, never *rankings* — and the ranking is the whole
  product. (`q`, by contrast, differs per action, which is why it can supervise a move-evaluator at all.)
- **`z` is high-variance and would need far more data.** `q` is a bootstrapped, low-variance (if biased)
  estimate; `z` is a single ±1 Monte-Carlo sample of the outcome — unbiased but very noisy, and with the
  ~75% deal-variance in this game one game's `z` is a near-coin-flip label for the state. MC variance falls
  as ~1/N, so recovering the true expectation from `z` needs **many** more games covering the same states
  than `q` does. At 6k games `z` is mostly noise; it *might* help at a much larger corpus, but only as a
  *level* correction (see above), never to sharpen action ranking.

Caveat that keeps the door ajar: the offline ranking metrics (top-1 / top-2 recall) implicitly treat MCTS
`q` as ground truth, so they *cannot* credit `z` for correcting search bias — which is exactly why the
head-to-head, not val-MSE, was the arbiter. If the search were later shown to be materially biased, a
small-λ blend (e.g. 0.25) on a **much** larger corpus is the only version worth revisiting.

## Components

| module | needs torch | what |
|---|---|---|
| `features.py` | no | `encode(view, action) -> float32[FEATURE_DIM=216]` — the shared featurizer (info-set based; bag-of-located-cards + globals + action). `feature_names()` for interpretability. |
| `dataset.py`  | no | replay corpus → `.npz` (`X`, `y`=q, `w`=visit_share, `z`, `game_id`, `decision_id`, `is_chosen`) + `.meta.json`. |
| `mlp.py`      | yes | `MLP(in_dim, hidden_dims, dropout)` — any shape (`[16]`, `[16,16]`, `[32,32,64]`, `[]`=linear); `Tanh` output bounds to `q`'s [-1,1]. |
| `train.py`    | yes | game-split, weighted-MSE training w/ early stopping, the `--sweep`, and metrics; saves `mlp_<arch>.pt` (a checkpoint) + `sweep_results.csv` to `--out-dir` (default `models/`). |
| `checkpoint.py` | yes | `save(path, model, meta)` / `load(path) -> (MLP, meta)` — one self-describing checkpoint format for train, agent, and UI. |
| `agent.py`    | yes | `NNPolicy` — the reusable eval-move-picker (`evaluate`/`best_move`, `from_checkpoint`); `NNAgent` — a thin `Agent`-protocol wrapper (greedy over predicted `q`). Also seats a checkpoint as the `ui.app` bot via `--nn`. |
| `benchmark.py` | yes | win-rate of a checkpoint vs parameterizable MCTS opponents (`fixed<N>` / `hybrid-k<k>-l<l>`), mirrored + paired seeds. |

## Metrics

- **Regression:** val MSE / MAE on `q`, vs a constant-`mean(q)` baseline MSE.
- **Ranking** (grouped by `decision_id`): per-decision **top-1 agreement** — does `argmax` predicted value
  pick (`top1_bestq`) the highest-`q` candidate and (`top1_chosen`) the actually-played move — plus mean
  per-decision Spearman(pred, q).

Reference run (k20l3, 2000 games ≈ 240k rows): all archs beat the 0.26 baseline (~0.07–0.08 val MSE); the
`32`-unit model is the MSE sweet spot, `64` ranks marginally better, and a 2-layer `32,32` doesn't beat
single-layer — consistent with the small, largely combinatorial state (the relational structure is what
the future attention model is meant to capture; both share `features.py`).

## Play & benchmark

```bash
# win-rate of a checkpoint vs MCTS opponents (sanity: ~parity vs k20, then N=500, k=30)
python -m imposterkings.machine_learning.benchmark --model models/mlp_32.pt \
    --opponent hybrid-k20-l3 fixed500 hybrid-k30-l3 --deals 100 --workers 10
# play a human vs the NN bot live
python -m imposterkings.ui.app --nn models/mlp_32.pt
```

Note: greedy-over-learned-`q` is typically weaker than the MCTS that generated the targets (~50% top-1
agreement), so ~50-50 vs k20 is the optimistic ceiling; the benchmark quantifies the distillation gap.

## AlphaZero-style self-play loop

Wire the net into ISMCTS as a **value + policy head** (PUCT selection, `V=max_a Q`, `P=softmax Q`, no
rollout) and bootstrap: NN-MCTS self-play → better `mean_q` targets → retrain → repeat.

| module | what |
|---|---|
| `evaluator.py` | `build_evaluator(ckpt)` → `state -> ([per-seat value], {move: prior})` (numpy inference; the `mcts.SearchConfig.evaluator` hook). |
| `loop.py` | iterates: NN-MCTS self-play (`datagen --value-checkpoint`) → `dataset` → `train` → head-to-head vs a fixed rollout-MCTS reference. |

```bash
python -m imposterkings.machine_learning.loop --iterations 3 --games 2000 --mode fixed --k 600 \
    --arch 256 --temp-plies 6 --init-checkpoint models/mlp_32.pt --out-dir runs/az1 --workers 10
```
The net is used iff attached (`SearchConfig.evaluator`); the default rollout MCTS is untouched. NN-MCTS is
fast (a numpy leaf eval, no rollout), so `--k` sims can be lower than pure MCTS.

## Deferred
Two-head net (value→`z`, policy→`π`); the attention model (token adapter of the same `features.py`);
Dirichlet root noise + `c_puct`/FPU tuning; batched leaf evaluation.
