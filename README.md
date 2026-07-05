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

# Bot with a per-decision budget instead of fixed iterations:
#   hybrid    = clamp(k * eff_legal(l) * (1 + opp_cards), 64, 4096)   (branch + hand scaling)
#   branching = clamp(k * eff_legal(l), 64, 4096)
# k scales the budget; l = how many effective moves a sub-decision card (guess/select) counts as.
python -m imposterkings.ui.app --p1 hybrid --k 50 --l 3
python -m imposterkings.ui.review --p1 hybrid --k 50 --l 3 --seed 0   # post-game review of a bot-vs-bot game
```

The engine config (fixed iters vs branch vs branch+hand-scaling, and N/k/l) is also editable live via the
in-app **⚙ Settings** modal, and drives both the bot and the analysis/hint panels.

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

## Roadmap / TODO

1. **GodMode agent** — MCTS over the *fully-determined* game (an all-knowing observer). With perfect
   information there are no information sets and no determinization, so it is plain MCTS on the concrete
   `GameState`. A strong upper-bound baseline and a debugging oracle.
2. **Complexity analysis (redo)** — re-run the branching-factor / game-length / decision-count analysis;
   it predates a batch of engine bug-fixes (King's-Hand turn order, guess windows, …) so the old numbers
   are stale. Informs the budgets/dataset sizing below.
3. **NN datasets** — generate + analyze self-play datasets. A quick warm-start set from basic **MCTS@500**
   (or **MCTS@k=10**); sizing and the sampling budget informed by (2).
4. **Train NN models** — start with an **MLP** value/policy head (with room for richer architectures);
   this is the warm start for DQN self-play.
5. **NN-in-the-loop**
   - **5a** — use the trained NN as the **eval head for MCTS** (AlphaZero-style leaf evaluation, replacing
     random rollouts).
   - **5b** — **DQN self-play** to improve the model; benchmark by win-rate vs **MCTS@k=100** (or 50) —
     thresholds calibrated against (2).
   - **5c** — an **explainable Attention+MLP** model: inspect the attention to see *which cards* drive each
     decision.
