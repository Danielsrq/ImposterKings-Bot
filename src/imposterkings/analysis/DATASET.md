# DATASET.md — self-play data collection for the NN phase

Status: **design spec** (no collector code yet — this doc defines the schema so it can be reviewed before
implementation). It specifies one dataset that serves three consumers from **one collection pass**:

1. a simple **MLP move-evaluator** — `f(state, action) → value`;
2. an **explainable attention model** — each card is a token, one self-attention block, a CLS token reads
   out the value, and `CLS→card` attention is an importance map ("which cards matter for this move");
3. **ground-truth exploration** — queryable stats over hidden/discarded cards, card-location correlations,
   and ability-usage rates.

The MLP and the attention model consume the **same featurizer** (§5); the exploration tables come from the
**same logs** (§6); and every logged game is **replayable in `ui.review`** for visual inspection (§7).

Related: scaling findings that motivate the target choices live in the `imposterkings-scaling-findings`
project memory. Analysis-package conventions are in `README.md` (this directory).

---

## 1. Principles

**Log once, derive many.** The engine is deterministic — `state.apply(action)` is a pure copy-on-write
transform with no RNG (`state.py:266-268` → `abilities.resolve`, `abilities.py:290`); randomness lives only
in the deal and the agents' search. So a game is fully reconstructable from `(deal_seed, ordered actions)`.
We therefore store a compact **replayable log** and *derive* training tensors and exploration tables by
replaying it through the engine. Consequence: when the feature encoder changes (it will), we re-derive in
seconds instead of regenerating games (hours of MCTS).

**Info-set for features, ground-truth for exploration.** Training features are computed from the
per-seat **information set** (`state.information_set(seat)`) — the model must never see hidden cards.
Exploration queries use the **omniscient** replayed `GameState`. The two are kept strictly separate.

**One featurizer, two adapters.** A single `featurize(view, state, action)` produces per-card + global
features; the MLP flattens them to a fixed vector, the transformer arranges them as a token set. Same
numbers, so the two models can never silently disagree about "the state."

---

## 2. Archival log — JSONL, one game per line

Extends the existing `record.py` `GameRecord` (which already carries `decisions`, `winner`, `rewards`)
with a **replayable header**. Because `play_game` fires `on_decision` on *every* ply including forced ones
(`arena.py:33-39`), the ordered `decisions[].chosen` **is** the action log — the game replays from
`GameState.deal(np.random.default_rng(deal_seed))`.

```jsonc
{
  "schema_version": 1,
  "engine_hash": "<git or config hash>",          // reproducibility
  "gen": { "spec": "hybrid-k20-l3", "temp_plies": 6, "temp": 1.0,
           "base_seed": 0, "self_play": true },
  "deal_seed": 12345,                              // GameState.deal(default_rng(deal_seed)) == initial state
  "starting_player": 0,                            // derivable from the dealt state; stored for convenience
  "winner": 0,
  "rewards": [1.0, -1.0],                          // per-seat, ±1 (rules.terminal_rewards)
  "decisions": [                                   // one per ply, in order; forced plies included
    { "seat": 0, "phase": "MAIN",
      "chosen": { "kind": "PLAY_CARD", "card": 6 },        // action_to_dict(move) -> replay log
      "candidates": [                                       // MCTS stats (empty on forced plies)
        { "move": {"kind":"PLAY_CARD","card":6}, "move_str": "Oathbound(6)#6",
          "visits": 812, "mean_q": 0.21, "visit_share": 0.40 }, ... ],
      "sims": 2048, "elapsed_ms": 190.0,
      "z": 1.0 }                                            // back-filled = rewards[seat], ±1
  ]
}
```

Only the header fields (`schema_version`, `engine_hash`, `gen`, `deal_seed`, `starting_player`) are new;
`decisions[]`/`winner`/`rewards`/`z` already exist in `record.py`. Logs are written in **shards**
(`games_00000.jsonl`, …) to an `--out-dir`, never into a canonical `results/` file.

---

## 3. Replay & determinism

To reconstruct any game:

```python
state = GameState.deal(np.random.default_rng(rec["deal_seed"]))
for d in rec["decisions"]:
    # ... derive features / table rows from `state` here ...
    state = state.apply(dict_to_action(d["chosen"]))
# state.winner == rec["winner"]
```

`apply` reconciles the knowledge frozensets (`hand_has`/`hand_lacks`) internally, so the info-set at each
replayed ply **exactly** reproduces what the acting agent saw (guess-revealed facts included). The engine's
`history` field is never populated — the collector is the *only* source of the ordered action log, which is
why it must capture every `on_decision` call. `dict_to_action` (the inverse of `record.action_to_dict`) is
the one helper `record.py` still lacks.

---

## 4. Training records — "action-in"

The primary model evaluates a *move*: `f(state, action) → value`. Each decision's search already scored
**every** legal action, so each decision expands to ~5.6 rows (the mean branching factor) — good and bad
moves with their values, which is exactly what a move-evaluator needs to learn:

| field | source | notes |
|---|---|---|
| `x` | `featurize(view, state, a)` | info-set features with `a` marked (§5) |
| `a` | `candidate.move` | the scored action |
| `target_q` | `candidate.mean_q` | MCTS action-value, **mover-relative, [−1,+1]** |
| `target_share` | `candidate.visit_share` | policy weight of `a` |
| `target_z` | `rewards[seat]` | terminal outcome, **±1**, same for all candidates of a decision |
| `is_chosen` | `a == chosen` | 1 for the played move |

**Scale convention.** `z`, `q` and the reward vector are all mover-relative on `[−1, +1]`
(`rules.terminal_rewards` returns ±1; `SearchResult.root_value()`/`MoveStat.mean_q` are on the same scale,
`mcts.py:93-99`). Keep them coherent so a value target `λ·z + (1−λ)·q` is well-defined. A `{0,1}` `win`
column is emitted alongside for human readability only.

**Why q over z.** The scaling study found ~75% of games are deal-decided, so `z` is a *noisy* per-move
label; `mean_q` conditions on the actual position and is the informative target. Expect small `λ`.

**Policy target.** The full visit distribution (`SearchResult.policy_target()`, `mcts.py:86-91`) is stored
per decision, enabling an optional policy head / distillation later.

Training tensors are materialized to `.npz` (numpy) by the build step (§10): `X` (features), `A` (action
encodings), `Q`, `PI`, `Z`, plus a row index back to `(game_id, ply, candidate)`.

---

## 5. The shared featurizer (`features.py`, future — numpy only)

A **located card** is the unit. The same card means different things in different zones, so a token is
`identity + zone + state + action-mark + belief`. Fixed width (~35–40 dims), identical for every token
including the CLS/global token.

| block | dims | contents |
|---|---|---|
| identity | 14 | one-hot card type (`CARD_NAMES`) |
| ability-category | ~8 | multihot: guess / optional-on-play / info-gather / stack-target / swap / mute-number / reaction / follow-up |
| value | 2 | base value + effective value (via `effective_stack_value`/`effective_hand_value`), normalized |
| zone | 10 | one-hot: `my_hand, opp_hand_known, my_hidden, leading, stack_below, antechamber_mine, antechamber_opp, discard_removed, muted, opp_unknown` |
| state | 5 | `disgraced, is_leading, is_legal_now, is_candidate_action, king_related` |
| belief | 2 | `identity_known` (vs inferred), `count` (duplicates in this (type,zone)) |

**Global / CLS token** — same width, an `is_global` flag, card slots zeroed; carries leading value, phase
one-hot (`StepKind`), whose-turn, both kings-flipped, hand sizes, muted-value bitmask. Its post-attention
embedding feeds the value head; `CLS→card` attention is the importance readout.

**Zone is the load-bearing feature.** It turns "a bag of cards" into "the game state," and it's where
imperfect info lives: opponent cards begin as `opp_unknown` tokens (identity zeroed, `count` =
`opp_hand_count`); a guess that lands flips one to `opp_hand_known` with identity filled (from
`opp_hand_has`/`opp_hand_lacks`, which are **name-based** — map names→ids via `card_ids_for_name`).

**Two adapters, same numbers.**
- **MLP:** flatten a `[zone × card_type]` count/feature grid + the global scalars → one fixed vector.
- **Transformer:** the token set + a key-padding mask (variable token count is native to attention);
  **no positional encoding** (it's a set — meaning comes from `zone`, not order); permutation-invariant.

**Action-in.** Set `is_candidate_action` on the involved token and read CLS → `Q(s,a)`; call once per
candidate. For non-card actions (guess-a-type, choose-number, declare/decline, flip-king, reactions) add a
transient "choice token" (e.g. a `guess`-zone token of the guessed type) so every action kind is markable.

**Duplicates as count, not repeats.** One token per `(card_type, zone)` with a `count` feature — repeated
identical tokens would split attention and muddy the importance map. The legal layer already dedupes by
name (`abilities.legal_play_cards`), so this matches engine semantics.

**Ability-category multihot** is built from existing groupings: `abilities._OPTIONAL_ONPLAY`,
`abilities._MANDATORY_GUESS`, `budget._HEAVY_ABILITIES`, `cards.Tag.REACTION`/`ROYALTY`, and the OATHBOUND
follow-up branch.

**v2 (additive, no re-collection): belief distribution.** Replace `opp_unknown` count tokens with a
per-type probability that the opponent holds each card (from the info-set's consistent-hand set /
determinization), letting attention reason about *likely* hidden cards. Deferred; the log already supports
it because features are re-derived by replay.

---

## 6. Exploration tables (CSV) + query cookbook

Derived by replaying logs; CSV so they're pandas-ready the moment pandas is installed.

**`card_locations.csv`** — one row per (game, card instance): `game_id, card_id, name, value, ability,
init_location ∈ {hand0, hand1, hidden0, hidden1, stack, muted, setup_discard0/1, leftover_faceup,
leftover_facedown}`.

**`ply_events.csv`** — one row per decision: `game_id, ply, seat, phase, action_kind, action_card,
guess_name, target, number, leading_card, source_card, blocked_kind, blocked_card, mover_z, mcts_q,
root_value, legal_count`. (`source_card` = the ability owner from `pending[-1].source`; `blocked_*` filled
for reaction plies by inspecting the omniscient stack.)

**`games.csv`** — one row per game: `game_id, deal_seed, spec, winner, rewards, num_plies, num_decisions,
hidden0, hidden1, kings0, kings1`, plus rolled-up ability-usage counts (`n_kingshand_used`,
`n_assassin_reveals`, `n_oathbound`, …).

Query cookbook (once `import pandas as pd`):

- **(a) hidden / discarded card distribution** —
  `cl[cl.init_location.isin(["hidden0","hidden1"])].name.value_counts(normalize=True)`.
- **(a2) `P(Queen in hand | Assassin hidden)`** — pivot `card_locations` to one row per game
  (`name → location`), then `g[g.Assassin.isin(["hidden0","hidden1"])].Queen.isin(["hand0","hand1"]).mean()`.
  Any card-location correlation is a groupby — this is why the ground-truth deal is stored.
- **(b) King's-Hand usage / target** — `kh = ply_events[ply_events.action_kind=="REVEAL_KINGSHAND"]`;
  rate = `len(kh)/n_games`; `kh.blocked_kind.value_counts()` (assassin-flip vs ability) and
  `kh.blocked_card.value_counts()` ("used on what").

---

## 7. Review-replayable format — `ui.review --replay <file>`

Every log is a reproducible game, so it drops straight into the **existing** review machinery:

```
GameRecord → initial_state = GameState.deal(default_rng(deal_seed))
           → moves = [dict_to_action(d["chosen"]) for d in decisions]
           → review.scripted_trajectory(initial_state, moves, ...)   # existing
           → run_review / render_review_frame                        # existing
```

CLI: `python -m imposterkings.ui.review --replay <file> [--game N] [--fast]`.

- **Default** re-searches each ply → a complete review (dual-eval graph + full recursive icicle/graft).
  Reproducible because `apply` is deterministic; the evals are freshly computed (not the exact
  generation-time RNG), which is fine for visual inspection.
- **`--fast`** skips re-search and synthesizes a per-ply eval strip + a shallow (root-level) icicle
  directly from the stored `candidates` (`visit_share`/`mean_q`). The deep recursive graft needs the full
  search tree, which is intentionally **not** serialized (far too heavy for a training corpus).

This is a key reason `record.py` keeps per-decision `candidates`: any self-play game is visually
inspectable without re-running the generator.

---

## 8. Generation policy

**Default: `hybrid-k20-l3` self-play** (~9.4k iters/game, ~13s/game self-play; 52.5% vs MCTS@500 in the
sweep) with a **temperature-sampled opening** (τ=1 for the first ~`temp_plies` plies, then argmax) for
state-space coverage. A thin exploration-agent wrapper samples the *played* move from
`result.policy_target()` while still recording the *true* search stats — so exploration diversifies the
states without corrupting the policy/Q targets (AlphaZero-style). `--spec` is fully overridable
(`fixed<N>` / `hybrid-k<k>-l<l>` / `branching-…`, via `budget_scaling.make_agent`).

**Root Dirichlet noise is deferred.** The current MCTS is prior-free UCT (`c=√2`), so there are no priors
to perturb; temperature sampling is the exploration knob until an NN supplies priors, at which point
Dirichlet becomes meaningful.

Rationale (from the scaling study): because outcomes are deal-noise-dominated, coverage (more diverse
games) matters as much as per-move strength; k20 is a deliberate balance of Q-target quality vs volume.

---

## 9. Formats & dependencies

- **Logs:** JSONL shards (human-readable, lossless, replayable).
- **Exploration tables:** CSV (numpy/`csv`, no new deps; pandas-ready).
- **Training tensors:** `.npz` (numpy).

All of the above are **numpy-only** — no new dependencies (pandas/pyarrow/torch are absent today and only
needed when you start exploring/training). Parquet is a documented optional upgrade for the tables once
pandas/pyarrow are installed.

---

## 10. Implementation roadmap (follow-up task — not built here)

- **`features.py`** (core, numpy-only): the shared featurizer of §5 — `featurize`, the MLP and token
  adapters, `FEATURE_SPEC` dim constants, `card_category(id)`, and index→label helpers for interpretability.
  Placed at package root so `analysis/` and a future `nn/` both import it (engine never imports it back).
- **`record.py`** (extend): add the replay header (`schema_version`, `engine_hash`, `gen`, `deal_seed`,
  `starting_player`) to `GameRecord`, and add `dict_to_action` (inverse of `action_to_dict`).
- **`analysis/datagen.py`** (new, runnable): chunked-parallel self-play collector reusing
  `budget_scaling.make_agent`/`spec_label`/`_cost_hook` + `play_game`'s `on_decision`; the
  temperature-exploration agent wrapper; writes JSONL shards to a non-clobbering `--out-dir`. CLI:
  `--games --spec --temp-plies --temp --workers --chunk --base-seed --shard-size --out-dir`.
- **`analysis/build_tables.py`** (new, runnable): replays logs → the three CSV tables + the training
  `.npz`, modeled on `merge_sweeps.py` / `eval_slice.py` (read raw, derive, write prefixed outputs, never
  clobber the source).
- **`ui/review.py`** (extend): the `--replay` path (load `GameRecord` → `scripted_trajectory`) + the
  `--fast` stats-only adapter.

---

## 11. Open items

- **Belief features (v2)** — per-type opponent-hand probabilities as token features (§5); additive.
- **First-mover advantage** — needs the collector to record **winner-by-seat** per game; the scaling run's
  `deal_outcomes.csv` records only `challenger_wins` (0/1/2) and can't answer "does the starter win the
  split deals". One extra field at collection time closes this.
- **Reproducibility** — settle `engine_hash` (git commit vs a hash of the rules/cards config) and bump
  `schema_version` on any log-format change.
