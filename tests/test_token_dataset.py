"""token_dataset builder: replay the self-play logs -> ragged token rows. Verifies the stored tokens
exactly match a fresh tokenize (round-trip), the CSR structure, and row-count parity with the flat MLP
builder. Requires the committed corpus at datasets/selfplay_k20l3/."""
import glob
import os

import numpy as np
import pytest

from imposterkings.machine_learning import dataset as DS
from imposterkings.machine_learning import features as F
from imposterkings.machine_learning import token_dataset as TD
from imposterkings.record import dict_to_action, read_jsonl
from imposterkings.state import GameState

DATA = os.path.join("datasets", "selfplay_k20l3")
pytestmark = pytest.mark.skipif(not glob.glob(os.path.join(DATA, "*.jsonl")),
                                reason="self-play corpus not present")


def test_build_structure(tmp_path):
    out = str(tmp_path / "tok.npz")
    stats = TD.build(DATA, out, limit=3)
    rows = TD.load(out)
    assert len(rows) == stats["n_rows"] > 0
    assert rows.cards.shape[1] == F.CARD_DIM
    assert rows.board.shape[1] == F.BOARD_DIM
    assert rows.phase.shape[1] == F.PHASE_DIM
    assert rows.action.shape[1] == F.ACTION_DIM
    assert len(rows.y) == len(rows.board) == stats["n_rows"]
    # CSR offsets: monotonic, cover the whole cards array, len n_rows+1
    assert rows.card_offsets[0] == 0
    assert rows.card_offsets[-1] == rows.cards.shape[0] == stats["total_card_tokens"]
    assert len(rows.card_offsets) == stats["n_rows"] + 1
    assert np.all(np.diff(rows.card_offsets) >= 1)      # every row has >=1 card token


def test_roundtrip_tokens_match_fresh_tokenize(tmp_path):
    out = str(tmp_path / "tok.npz")
    TD.build(DATA, out, limit=1)
    rows = TD.load(out)
    # independently replay the FIRST game and compare its rows (stored in order) to fresh tokenize
    first = read_jsonl(sorted(glob.glob(os.path.join(DATA, "*.jsonl")))[0])[0]
    st = GameState.deal(np.random.default_rng(first["deal_seed"]))
    ri = 0
    for d in first["decisions"]:
        cands = d.get("candidates") or []
        if cands:
            view = st.information_set(d["seat"])
            for c in cands:
                t = F.tokenize(view, dict_to_action(c["move"]))
                cds, bd, ph, ac = rows.tokens(ri)
                assert np.array_equal(cds, t.cards)
                assert np.array_equal(bd, t.board)
                assert np.array_equal(ph, t.phase)
                assert np.array_equal(ac, t.action)
                assert rows.y[ri] == np.float32(c["mean_q"])
                assert rows.w[ri] == np.float32(c["visit_share"])
                ri += 1
        st = st.apply(dict_to_action(d["chosen"]))
    assert ri > 0


def test_row_count_parity_with_mlp_builder(tmp_path):
    tok = TD.build(DATA, str(tmp_path / "tok.npz"), limit=3)
    mlp = DS.build(DATA, str(tmp_path / "mlp.npz"), limit=3)
    assert tok["n_rows"] == mlp["n_rows"]               # same decisions -> same candidate rows
    assert tok["n_decisions"] == mlp["n_decisions"]
