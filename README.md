# ImposterKings

A from-scratch Python rebuild of the 2-player hand-clearing card game **ImposterKings**: a headless
rules engine, a Single-Observer Information-Set MCTS bot with explainability, and a PyGame frontend.
Built mirroring the conventions of the sibling `bigtwo` project.

## Install

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,ui]"   # dev=pytest, ui=pygame; add ml for torch later
```

## Run

```bash
# Terminal play
python -m imposterkings.cli --p0 human --p1 random
python -m imposterkings.cli --p0 mcts  --p1 random --iters 1000 --explain   # show the search table

# PyGame window (click action buttons; opponent is a bot)
python -m imposterkings.ui.app --p1 mcts --iters 800
```

## Test

```bash
.venv\Scripts\python -m pytest -q              # fast suite
.venv\Scripts\python -m pytest -q -m slow      # MCTS-strength arena
```

## Architecture (`src/imposterkings/`)

The engine is pure and copy-on-write; agents only ever see an `InformationSet`, never the omniscient
`GameState`. A turn resolves as a **stack of micro-decisions** (`GameState.pending`), so compound
abilities and the nested reaction windows (King's Hand / Assassin) are ordinary searchable nodes —
the MCTS (ported near-verbatim from bigtwo) needs no special-casing.

| Module | Role |
|---|---|
| `cards` / `rules` / `actions` | card registry (18 instance ids), tunables + rewards, the `Action` type |
| `abilities` | the `resolve(state, action)` dispatcher and every card's semantics |
| `state` / `generate` | the resolution-stack state machine; `legal_moves` per decision point |
| `infoset` | projection + near-perfect-info `determinize` (the MCTS seam) |
| `mcts` / `agents` | SO-ISMCTS `search`; `RandomAgent`, `MCTSAgent` (`last_result` for explainability) |
| `arena` / `record` / `explain` / `cli` | game driver, JSONL (NN seam), formatters, terminal UI |
| `ui/` | PyGame frontend (`assets`, `render`, `app`) over the engine |

The neural-net dataset/training pipeline (goal 3) is deferred; the seams exist
(`record.play_and_record`, `SearchResult.policy_target`).
