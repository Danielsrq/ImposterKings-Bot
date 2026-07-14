"""Data paths must resolve BOTH from a source checkout and inside a frozen build.

Two habits break the moment the app is packaged, and both were live in the code:

* ``Path(__file__).parents[3] / "assets"`` encodes the SOURCE TREE's shape. Frozen, __file__ moves.
* ``"models/..."`` is relative to the WORKING DIRECTORY -- fine when you launch from the repo root, useless
  for an .exe someone starts from their Desktop.

Neither failure shows up in a normal test run (cwd IS the repo root, and nothing is frozen), so these tests
simulate both conditions explicitly. Without them the first symptom would be a black window on a stranger's
machine.
"""
import os
import sys
from pathlib import Path

import pytest

from imposterkings import paths


def test_source_layout_finds_the_real_assets_and_models():
    assert paths.asset_dir().is_dir(), "assets/ not found from a source checkout"
    assert (paths.asset_dir() / "99_back.jpg").exists()          # the card back the UI always needs
    assert paths.model_dir().name == "models"
    assert not paths.is_frozen()


def test_frozen_resolves_against_the_pyinstaller_bundle(monkeypatch, tmp_path):
    """Frozen, everything must hang off sys._MEIPASS -- where the spec's `datas` were unpacked."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert paths.is_frozen()
    assert paths.resource_dir() == tmp_path
    assert paths.asset_dir() == tmp_path / "assets"
    assert paths.model_dir() == tmp_path / "models"
    # A repo-relative literal lands inside the bundle, NOT in the cwd -- and at exactly the location the
    # spec's datas declare (`("../models/release", "models/release")` -> <bundle>/models/release).
    got = Path(paths.model_path("models/release/attn_d64_L2.npz"))
    assert got == tmp_path / "models" / "release" / "attn_d64_L2.npz"


def test_model_path_leaves_absolute_paths_alone(tmp_path):
    """A user pointing --nn at a checkpoint outside the bundle must still be obeyed."""
    p = str(tmp_path / "elsewhere.npz")
    assert paths.model_path(p) == p


def test_resolve_ckpt_prefers_what_exists_then_falls_back_to_the_bundle(monkeypatch, tmp_path):
    """The dev case (path exists relative to cwd) must win; only a MISSING path is re-resolved against the
    bundle. Getting this backwards would break every developer's --nn while 'fixing' the frozen one."""
    real = tmp_path / "here.npz"
    real.write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    assert paths.resolve_ckpt("here.npz") == "here.npz"           # exists as given -> untouched

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    got = paths.resolve_ckpt("models/release/attn_d64_L2.npz")    # missing -> re-resolved into the bundle
    assert Path(got) == tmp_path / "models" / "release" / "attn_d64_L2.npz"
    assert paths.resolve_ckpt(None) is None


def test_the_ui_reads_art_through_the_resolver_not_a_hardcoded_tree():
    """ui.assets must go through paths.asset_dir(); a reintroduced parents[N] hop would pass every other
    test and then ship a game with no card art."""
    pytest.importorskip("pygame")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from imposterkings.ui import assets
    assert assets.ASSETS_DIR == paths.asset_dir()


def test_checkpoint_discovery_is_not_cwd_relative(monkeypatch, tmp_path):
    """discover_ckpts() feeds the Settings model switcher. If it globs the cwd, the shipped game shows an
    EMPTY model list to anyone who launches the .exe from somewhere other than the install dir."""
    pytest.importorskip("pygame")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from imposterkings.ui.app import discover_ckpts

    from_repo = discover_ckpts()
    monkeypatch.chdir(tmp_path)                       # pretend the .exe was launched from the Desktop
    assert discover_ckpts() == from_repo, "discovery moved with the cwd -- it is still cwd-relative"


def test_the_same_file_under_different_separators_is_ONE_checkpoint(monkeypatch, tmp_path):
    """The Settings picker showed THREE options, two of them the same net under the same name.

    Cause: model_path() builds paths with pathlib (backslashes on Windows) while discover_ckpts() normalises
    to forward slashes, so `nn_ckpt not in nn_ckpts` was True for a file that was already in the list and it
    got inserted again. Paths must be compared canonically, never as strings."""
    a = r"C:\Some\Where\models\release\attn_d64_L2.npz"
    b = "C:/Some/Where/models/release/attn_d64_L2.npz"
    assert paths.canon(a) == paths.canon(b)
    assert paths.same_file(a, b)
    assert paths.same_file(a.upper(), b)              # Windows is case-insensitive; normcase folds it
    assert not paths.same_file(a, b.replace("attn_d64_L2", "mlp_256"))
    assert not paths.same_file(None, b) and not paths.same_file(a, None)


def test_the_default_checkpoint_is_not_listed_twice(monkeypatch, tmp_path):
    """End to end through app.run()'s list-building: the net the game boots with must be SELECTED in the
    picker, not appended as a phantom extra entry."""
    pytest.importorskip("pygame")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from imposterkings.ui.app import discover_ckpts

    models = tmp_path / "models" / "release"
    models.mkdir(parents=True)
    for n in ("attn_d64_L2.npz", "mlp_256.npz"):
        (models / n).write_bytes(b"x")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    elsewhere = tmp_path / "launched_from_here"                    # the .exe is started from the Desktop,
    elsewhere.mkdir()                                             # NOT from the install dir
    monkeypatch.chdir(elsewhere)

    found = discover_ckpts()
    assert len(found) == 2                                        # exactly the two shipped nets

    boot = paths.resolve_ckpt("models/release/attn_d64_L2.npz")   # what run_game.py hands to run()
    hits = [p for p in found if paths.same_file(p, boot)]
    assert len(hits) == 1, "the boot checkpoint is not exactly one of the discovered ones"

    # THE BUG: the raw string test says it is absent (backslashes vs forward slashes), so app.py used to
    # insert it a second time -> a third entry, identical in name to the first.
    assert boot not in found, "if this ever passes, the separator hazard is gone and `in` would be safe"
