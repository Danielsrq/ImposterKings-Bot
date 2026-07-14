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
| `replay`         | **The shared replay primitive**: re-derive a recorded game ply by ply, handing back the state *before* each action. Has the legality canary; use it instead of writing a fifth replay loop. |
| `corpus_stats`   | **Who wins and why**: turn-order advantage, setup policy, hand-composition win-rates, ability usage, passive overrides, King's-Hand counters, guess accuracy, win reasons. CSV + `stats.json` + PNGs. |
| `build_tables`   | Older exploration tables (card locations, ply events, per-game facts) + a query cookbook. |

All are runnable modules and write CSVs (+ PNGs) to an `--out-dir`. Runs are deterministic:
paired per-deal seeds and mirrored seating, so results are reproducible and low-variance.

See **[DATASET.md](DATASET.md)** for the NN self-play data-collection spec (schema, the shared MLP/attention
featurizer, exploration query cookbook, `ui.review --replay`, and generation policy).

---

## `corpus_stats` — who wins, what the bot keeps, which powers pay

```bash
# one corpus (the default)
python -m imposterkings.data_analysis.corpus_stats --data datasets/selfplay_gen1_v3c

# several, reported side by side; --pool folds them into one population instead
python -m imposterkings.data_analysis.corpus_stats \
    --data datasets/selfplay_gen1_v3c datasets/selfplay_v3c_k50 --pool --max-combo 4
```

| flag | default | meaning |
|---|---|---|
| `--data` | `datasets/selfplay_gen1_v3c` | one or more corpus dirs; **side by side** unless `--pool` |
| `--out-dir` | `results/corpus_stats` | one sub-dir per corpus + a shared `stats.json` |
| `--max-combo` | `4` | largest hand sub-combination scored (see below — it does not explode) |
| `--limit` | – | first N games per corpus (smoke) |
| `--pool` | off | one population. Sound for turn-order across symmetric self-play; **not** across different bots, budgets, or the asymmetric corpus |
| `--no-plot` | off | skip the PNG |

Every number is produced by **replaying** the game and asking the engine, never by re-deriving state from
the ply log — `royalty_present()`, `active_royalty()`, `effective_hand_value()`, "does the defender hold the
named card". Reconstructing those from the recorded actions would mean reimplementing disgraces, mutes,
King's-Hand removals and Fool grabs, i.e. reimplementing `abilities.resolve` with fresh bugs.

### Three things that will mislead you if you forget them

1. **A hand-row is `(game, seat)`, not a game.** One deal makes two different 8-card hands with opposite
   outcomes. Pooled over both seats the base win-rate is exactly **50%** — the invariant that proves the unit
   is right, and the reason a combo's distance from 50% *is* its value.
2. **The played setup move is a sample, not the policy.** datagen ran at `temp_plies: 6` and setup is plies
   1–2, so `chosen` was drawn from the visit distribution. The bot's actual preference is `argmax
   visit_share` over the candidates. Both columns are reported; they are different questions.
3. **Corpora do not pool freely.** Turn-order pools across symmetric self-play (seat effects cancel by
   construction). Setup *policy* does not pool across different bots or search budgets, and the asymmetric
   corpus needs its agent × turn-order 2×2 or the strong bot's record swamps the weak one's.

### Why the combinatorics don't explode

Enumeration runs over **observed** hands, not the theoretical space. Each dealt hand is 8 cards, so
sub-combinations up to size 4 number `C(8,1..4) = 162`; across 8,000 hand-rows that is 1.3M counter
increments (sub-second), collapsing to ~1,800 distinct keys — because the deck has **14 distinct names**, not
18, so `Elder+Elder` is one key. Combos are deduped *per hand*, or a hand holding both Elders would count
`("Elder",)` twice and quietly drag every duplicate card's win-rate toward 50%.

Win-rates carry **Wilson** intervals, not normal ones: a 4-card combo lands in ~2% of hands, and at that n
the textbook interval runs off the end of `[0,1]`.

### Everything is split by seat (P1 = moves first, P2 = moves second)

Because **50% is the wrong null once turn order exists**: P1 wins 41.2% of hand-rows and P2 wins 58.8%. A combo
that wins 55% is *+14pp for P1* but *−4pp for P2* — pooling reports "55%, above average" and hides that it is a
bad hand for one of the two seats. So `hand_combos.csv` carries `lift_first` / `lift_second` (each seat against
**its own** baseline) and `interaction` (`lift_first − lift_second`), and `setup_hide.csv` /
`setup_discard.csv` carry `rate_first` / `rate_second` / `p1_minus_p2` with a two-proportion `z`.

### What it found (`selfplay_gen1_v3c`, 2,000 games / 4,000 hand-rows)

- **Moving second is worth +17.6pp** (58.8% vs 41.2%) — and it is *structural*, not skill-dependent. The
  asymmetric corpus gives the same lift to **both** agents to a tenth of a point (weak `fixed500`
  29.5%→46.8%, strong v3c 53.2%→70.5%), and the effect replicates at a deeper search budget (k50: 58.9%).
- **No hand is seat-specific.** Hand strength and turn order are **additive**. Across 848 well-powered combos
  the spread of the P1/P2 interaction is **4.70pp against 4.89pp expected from sampling noise alone**
  (ratio 0.96; only 2.7% clear |z| > 1.96, vs 5% by chance), and that null replicates at ratio **0.96 in all
  three corpora**. Princess+Queen lifts P1 by +23.6pp and P2 by +25.1pp — the same. **There is no hand that
  rescues the player who moves first.** The report prints this test and refuses to list the extremes when it
  passes, because with ~850 combos the largest "interaction" is guaranteed to look impressive by chance.
- **Princess+Queen wins 74.2%** of the hands that hold it; the best trio (`Assassin+KingsHand+Princess`)
  wins 85.0% (94.1% as P2).
- The Mystic's mute lands on **base value 2 (the Assassin) 42%** of the time — the bot learned to mute the
  card it fears. The King's Hand's most common target is likewise the **Assassin (23%)**.
- Games end by stall-out **70%** of the time, by Assassin **29.6%**, and by a King's Hand countering into an
  unbeatable throne **0.7%** (14 games).

### ⚠️ Two traps this module exists to keep you out of

**1. Setup policy looks seat-dependent, and the effect is a search artifact.** In gen-1 (k20) P1 hides the
Princess 11.1pp more often than P2 (z = +4.8) and the Queen 11.3pp more (z = +4.9) — highly significant, and
a tempting story about hoarding royalty when you are behind. **It does not replicate.** At k50 — *the same
net, only a deeper search* — those gaps collapse to +1.1pp (z = +0.5) and +3.0pp (z = +1.3), and a different
card (Oathbound, +8.6pp, z = +5.1) becomes the "seat-dependent" one instead. The significance is real within
each corpus; the *card it attaches to* moves with the search budget. Never read a setup asymmetry off one
corpus.

**2. Hand composition is causal; ability usage is not.** A dealt hand is fixed by `rng.permutation` before
anyone has acted, so "Princess+Queen wins 74.2%" is a clean causal effect of the deal. But *whether a power
got used* is a decision made **from a position the bot was already in** — so "Sentry used → 38.8%, declined →
65.6%" almost certainly does not mean the Sentry is bad; it means the bot reaches for a self-disgracing power
when it is already losing. Same shape for "kept the Assassin unused → 11.6%": that is a losing player running
out of turns before the flip ever came, not proof that holding it is fatal. Treat `ability_usage.csv` and
`passives.csv` as *descriptions of the policy*, not as advice.

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
