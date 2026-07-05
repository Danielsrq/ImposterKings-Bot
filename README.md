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

## Scenario builder & headless testing

Instead of hunting for a seed that reaches a specific position, **build the board directly** — a debug/test
ground for both the rules and the post-game review screen.

**Interactive (wired into `ui.app`).** Launch straight into the setup screen, or press **`S`** / click the
**Scenario** button in-game:

```bash
python -m imposterkings.ui.app --setup
```

Pick a zone (P0 hand / P1 hand / Stack / hidden), click palette cards to fill it (stack top = leading),
set Turn / king toggles, then **Play** — vs the **Bot** (like normal play) or **Hotseat** (you drive *both*
sides to force an exact interaction, e.g. a King's-Hand counter). The end-of-game **Review** button opens
the icicle review of what you played.

**Headless programmatic mode.** Construct a position and drive it to a review-ready trajectory with no
window — for asserting rules interactions *and* rendering the board/review to PNG in tests:

```python
from imposterkings import scenario as sb
from imposterkings.ui import review, headless

st = sb.build(hand0=["Oathbound", "Inquisitor", "Queen"],
              hand1=["Elder", "KingsHand"], stack=["Sentry"], turn_player=0)   # stack top = leading
print(sb.show(st))                                                            # readable board dump

# a FIXED line (each ply searched, so the review icicle/graft populate)
traj = review.scripted_trajectory(st, [sb.play_card(sb.cid("Oathbound")),
                                       sb.play_card(sb.cid("Inquisitor")),
                                       sb.guess("Elder"), sb.REVEAL_KINGSHAND])
headless.review_png(traj, "review.png", cursor=len(traj) - 1)   # capture the review screen
headless.board_png(traj[-1].state, "board.png")

# or let the bots play from a built opening:
traj = review.build_trajectory(iters=200, seed=0, initial_state=st)
```

`scenario.build` also accepts `hidden=`, `kings=`, `discard=`, and an explicit `pending=` to start
mid-ability; `sb.cid("Oathbound", 1)` picks the 2nd instance of a duplicate.

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
| `arena` / `record` / `explain` / `cli` | game driver (`play_game(initial_state=)`), JSONL (NN seam), formatters, terminal UI |
| `scenario` | board-test builder: `build(...)` any position (cards by name or id) + `play`/`show`/action shorthands |
| `ui/` | PyGame frontend: `assets`, `render`, `app` (`--setup`), `review` (icicle + trajectory builders), `scenario_setup` (board builder), `headless` (PNG capture) |

The neural-net dataset/training pipeline (goal 3) is deferred; the seams exist
(`record.play_and_record`, `SearchResult.policy_target`).
