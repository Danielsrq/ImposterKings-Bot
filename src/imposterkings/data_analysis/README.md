# `imposterkings.data_analysis` — data-science layer

Search / budget scaling studies and their reporting, kept separate from the core game engine
(state, rules, MCTS). Modules here import **up** into the engine (`from ..state import ...`); the
engine never imports back down. Nothing in here is needed to play the game — it's for measuring the
bot, sizing the future NN dataset, and understanding how much of a game is decided by the deal.

Modules:

| module | what it does |
|---|---|
| `budget_scaling` | Sweep a **hybrid MCTS@k,l** challenger vs fixed references; win-rate, seed split/sweep, opening evals, compute. |
| `search_scaling` | Sweep an **integer-N** challenger (MCTS@N) vs one fixed-N baseline; win-rate + starting-eval calibration. |
| `eval_slice`     | Reshape `search_scaling`'s raw `eval_scaling.csv` into readable per-seed / per-game / cross-N sweep tables. |
| `merge_sweeps`   | De-duped, non-clobbering merge of multiple `budget_scaling` run dirs into `merged_*.csv`. |
| `datagen`        | **Self-play → replayable JSONL corpus** (the NN dataset); see [DATASET.md](DATASET.md). |
| `replay_tools`   | Slice a dataset shard into per-game replay files, or bundle replay files back into a shard. |

All three are runnable modules and write CSVs (+ PNGs) to an `--out-dir`. Runs are deterministic:
paired per-deal seeds and mirrored seating, so results are reproducible and low-variance.

See **[DATASET.md](DATASET.md)** for the NN self-play data-collection spec (schema, the shared MLP/attention
featurizer, exploration query cookbook, `ui.review --replay`, and generation policy).

---

## `budget_scaling` — hybrid MCTS@k,l vs fixed references

```bash
# defaults == the standard study: k=10..50 (l=3) vs {fixed500, hybrid-k50}, 50 deals x2 = 100 games/matchup
python -m imposterkings.data_analysis.budget_scaling --workers 10

# larger, e.g. 100 deals (200 games/matchup):
python -m imposterkings.data_analysis.budget_scaling --deals 100 --workers 10
```

The challenger is `hybrid(k, l)` swept over `--k`; it plays **two** reference opponents at once:
a fixed **MCTS@N** (`--fixed-baseline`) and a fixed **hybrid MCTS@k** (`--hybrid-baseline`). Note
`k == hybrid-baseline` is **self-play** — the deal-variance floor built into the sweep.

### Arguments & defaults

| flag | default | meaning |
|---|---|---|
| `--k` | `10 20 30 40 50` | challenger hybrid-k values to sweep (the x-axis) |
| `--l` | `3` | sub-decision weight: a guess/select card counts as `l` legal moves for budget sizing |
| `--deals` | `50` | number of distinct **seeds** (deals); each is played **mirrored** → 2 games |
| `--fixed-baseline` | `500` | the **N** of the fixed MCTS@N reference |
| `--hybrid-baseline` | `50` | the **k** of the hybrid reference (same `l`) |
| `--base-seed` | `0` | first deal seed; deal `d` uses `base_seed + d` |
| `--workers` | `5` | parallel processes (set to ~cores−2) |
| `--chunk` | `10` | **deals** per parallel job (10 deals = 20 mirrored games/job) |
| `--out-dir` | `results/budget_scaling` | output directory (created if absent) |
| `--shared-rng` | off | share play-RNG across the mirror; **off is correct** — independent RNG is needed for meaningful split rates |
| `--no-eval` | off | skip the starting-position eval phase (bar 2) |
| `--no-plot` | off | skip PNG generation |

**Units of work / progress bars.** For `matchups = len(k) × 2 baselines` and
`chunks_per = ceil(deals / chunk)`:

- **bar 1 `chunks:`** = `matchups × chunks_per` jobs (win-rate games). E.g. `--deals 100`:
  `10 × 10 = 100` jobs.
- **bar 2 `evals:`** = `(len(k) + 2) × chunks_per` jobs (one search per starting position per spec).
  E.g. `--deals 100`: `7 × 10 = 70` jobs.

A chunk is measured in **deals**, not games — 10 deals = 20 games. The startup line prints the exact
job count (`chunk=10 -> 100 jobs`).

### Data collected

`winrate.csv` — one row per (challenger-k, baseline) matchup:

| column | meaning |
|---|---|
| `challenger`, `baseline`, `k`, `l` | the matchup |
| `deals`, `games`, `wins` | sample size and challenger wins |
| `winrate`, `ci95` | challenger win-rate (paired, mirrored) ± 95% CI |
| `splits` | # deals that split 1–1 (deal-decided, see below) |
| `plies`, `decisions`, `branching` | mean game length, real (non-forced) decisions, effective branching factor |
| `iters_challenger`, `iters_baseline` | mean MCTS iterations **actually spent** per game by each side (compute cost) |
| `seconds` | matchup compute-time (Σ chunk-task durations; single-core equivalent, so > wall) |

`deal_outcomes.csv` — one row per (matchup, seed) — the **seed split/sweep** data:

| column | meaning |
|---|---|
| `seed`, `deal`, `k`, `challenger`, `baseline` | the deal + matchup |
| `challenger_wins` | 0, 1, or 2 (of the 2 mirrored games) |
| `outcome` | `split` (1) / `sweep_challenger` (2) / `sweep_baseline` (0) |
| `split` | 1 if the pair split 1–1 |
| `plies`, `decisions`, `branching`, `iters_*` | per-deal game shape/compute |

`evals.csv` — one row per (spec, seed): starting-position eval from the mover's view —
`root` (visit-weighted value), `q1`/`q2` (top-1 / top-2 root-move Q), `iters` (budget spent),
`start_seat`.

### Graphs & relationships analysed

- `winrate_split.png` — **(left)** win-rate vs k, one curve per baseline, with CI; the 50% line is
  parity with each reference. **(right)** split-rate vs k: the fraction of games **decided by the
  dealt cards** rather than skill.
- `opening_eval.png` — top-1 / top-2 root-move Q vs k, with dashed reference lines for each baseline:
  watch the opening verdict converge as k grows.
- **Console: seed split/sweep analysis** — per baseline, the mean split-rate and the count of
  **deal-locked** seeds (split at *every* k → the deal, not skill, decides) vs **skill-locked** seeds
  (swept at every k). This answers "how many games are decided by the deal".
- **Console: timing** — single-core-equivalent compute, realized parallel speedup, per-game cost, the
  heaviest matchup, and a `~Xs per deal` projection to size a larger run.

**Split vs sweep — what it means.** In a mirror the deal `D` is fixed and the challenger plays both
seats. A **1–1 split** means whoever sat in a given seat won both times → the **seat/deal** decided
it, not skill. A **2–0 / 0–2 sweep** means one bot won regardless of seat → **skill** decided it.
Independent play-RNG (default) is required or the equal-strength mirror splits trivially.

---

## `search_scaling` — MCTS@N vs a fixed-N baseline

```bash
python -m imposterkings.data_analysis.search_scaling                    # N = 25..500 step 25, 50 deals each
python -m imposterkings.data_analysis.search_scaling --deals 100 --workers 8 --knowledge
```

Sweeps integer `N` (`--min/--max/--step`, default 25..500 step 25) against `--baseline` (default 500),
`--deals` deals (default 50) mirrored with paired seeds. `--independent-rng` measures the irreducible
variance at equal strength; `--knowledge` also records the first ply each seat reaches
binary/perfect knowledge of the opponent's hand. Outputs: `search_scaling.csv` (win-rate curve),
`eval_scaling.csv` (per-deal starting evals + calibration), and `search_scaling.png` /
`eval_scaling.png` / `calibration.png`.

## `eval_slice` — reshape the raw eval CSV

```bash
python -m imposterkings.data_analysis.eval_slice --n 200                 # per-seed table at N=200
python -m imposterkings.data_analysis.eval_slice --n 200 --per-game      # 2 rows/seed (one per mirrored game)
python -m imposterkings.data_analysis.eval_slice --sweeps                # who swept each deal, across all N
```

Reads `eval_scaling.csv` (`--in`), emits a readable slice (`--out`). `--per-game` splits each mirrored
pair into one row per game (its own starter, eval, prediction, result); `--sweeps` classifies each
deal across all N as won-by-baseline / won-by-N (upset) / split.

---

## Notes

- **Don't overwrite saved data.** Default out-dirs land in `results/`; point `--out-dir` at a scratch
  path for experiments. `results/` is the tracked copy of the canonical sweeps.
- Requires `joblib`, `tqdm`, `numpy`, and (for PNGs) `matplotlib` — all in the project venv.
- Reproducible: same seeds + specs → same numbers. Bump `--base-seed` for an independent replication.
