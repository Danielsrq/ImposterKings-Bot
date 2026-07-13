"""Asymmetric datagen: the primary agent vs a DIFFERENT opponent, to reach positions self-play never
visits. The contract that makes it safe:

- the opponent's moves ARE recorded (``chosen``) -- otherwise the game stops replaying, and replay is the
  whole point of the JSONL corpus;
- the opponent's moves carry NO candidates -- otherwise its (rollout-noise) ``mean_q`` becomes a training
  target. ``token_dataset`` builds rows only ``if cands``, so unlabelled decisions replay and vanish.

Engine + numpy only (no torch): the primary is a plain MCTS here; a checkpoint would only change its leaf
evaluator, not the recording contract.
"""
import numpy as np

from imposterkings.data_analysis.datagen import collect_game, primary_seat_for
from imposterkings.record import dict_to_action
from imposterkings.state import GameState

PRIMARY = ("hybrid", 4, 3)          # tiny budgets: this is a plumbing test, not a strength claim
OPPONENT = ("fixed", 24)


def _game(seed: int):
    return collect_game(PRIMARY, seed, temp_plies=0, base_seed=0, opp_spec=OPPONENT)


def test_opponent_moves_are_recorded_but_never_labelled():
    for seed in (0, 1):
        rec = _game(seed)
        seat = rec.gen["primary_seat"]
        assert seat == primary_seat_for(seed)
        opp = [d for d in rec.decisions if d.seat != seat]
        assert opp, "the opponent must actually move (and be recorded, or replay breaks)"
        # THE invariant: no opponent decision may ever carry a training target
        assert all(not d.candidates and d.sims == 0 for d in opp)
        assert all(d.chosen for d in opp)                        # ...but its action IS recorded
        labelled = [d for d in rec.decisions if d.candidates]
        assert labelled and all(d.seat == seat for d in labelled)
        # the primary's forced moves are unlabelled too (no search ran) -- that is pre-existing behaviour
        assert all(d.seat == seat for d in rec.decisions if d.candidates)


def test_mixed_games_still_replay_exactly():
    """deal_seed + the FULL ordered action list must reconstruct the game -- the regression a naive
    'only record my own turns' implementation would introduce."""
    for seed in (0, 1, 2):
        rec = _game(seed)
        st = GameState.deal(np.random.default_rng(rec.deal_seed))
        for d in rec.decisions:
            a = dict_to_action(d.chosen)
            assert a in st.legal_moves(), f"desync at seat {d.seat}"
            st = st.apply(a)
        assert st.is_terminal() and st.winner == rec.winner
        assert st.result() == rec.rewards


def test_primary_seat_mirrors_across_seeds():
    seats = [_game(s).gen["primary_seat"] for s in range(4)]
    assert seats == [0, 1, 0, 1]                                  # no seat bias in the corpus
    assert [primary_seat_for(s) for s in range(4)] == seats       # deterministic from the seed alone


def test_gen_meta_records_the_matchup():
    rec = _game(0)
    g = rec.gen
    assert g["self_play"] is False and g["record"] == "primary"
    assert g["spec"] == "hybrid-k4-l3" and g["opponent_spec"] == "fixed24"
    assert g["opponent_ckpt"] is None                              # vanilla rollout opponent


def test_self_play_is_unchanged_when_no_opponent_given():
    rec = collect_game(PRIMARY, 0, temp_plies=0, base_seed=0)     # no opp_spec -> the original path
    assert rec.gen["self_play"] is True
    assert rec.gen["opponent_spec"] is None and "primary_seat" not in rec.gen
    assert any(d.seat == 0 for d in rec.decisions if d.candidates)
    assert any(d.seat == 1 for d in rec.decisions if d.candidates)   # BOTH seats labelled in self-play


def test_token_dataset_builds_rows_only_from_the_primary(tmp_path):
    """The payoff: the opponent shapes WHICH positions we see, but never contributes a label."""
    from imposterkings.machine_learning import token_dataset as TD
    from imposterkings.record import write_jsonl

    recs = [_game(s) for s in range(4)]
    d = tmp_path / "mixed"
    d.mkdir()
    write_jsonl(str(d / "games_00000.jsonl"), recs)

    stats = TD.build(str(d), str(tmp_path / "t.npz"), feat="v1")
    expected = sum(1 for r in recs for dd in r.decisions if dd.candidates)
    assert stats["n_decisions"] == expected > 0                  # exactly the labelled (primary) decisions
    assert stats["n_games"] == 4                                 # ...and every game replayed (no desync)
    rows = TD.load(str(tmp_path / "t.npz"))
    assert len(rows) == stats["n_rows"] > 0
