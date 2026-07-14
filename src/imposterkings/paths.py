"""Where the game's DATA lives -- correct both in a source checkout and inside a frozen (PyInstaller) build.

Two habits break the moment the app is packaged, and both were present:

* ``Path(__file__).parents[3] / "assets"`` hard-codes the SOURCE TREE's shape (ui -> imposterkings -> src
  -> repo root). Frozen, ``__file__`` lives under the extraction dir and that 3-parent hop lands nowhere.
* ``"models/..."`` is relative to the WORKING DIRECTORY, so it resolves only when the app happens to be
  launched from the repo root -- i.e. never, for a shipped .exe someone starts from their Desktop.

Both now go through :func:`resource_dir`, which asks PyInstaller where it unpacked the bundle
(``sys._MEIPASS``) and otherwise walks up to the repo root. Anything read at runtime -- card art, model
weights -- must resolve through here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


def is_frozen() -> bool:
    """True inside a PyInstaller build (onedir or onefile)."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def resource_dir() -> Path:
    """The root that ``assets/`` and ``models/`` hang off.

    Frozen -> PyInstaller's extraction dir (``sys._MEIPASS``), where the spec's ``datas`` were placed.
    Source -> the repo root, four parents up from this file (src/imposterkings/paths.py -> repo)."""
    if is_frozen():
        return Path(sys._MEIPASS)                       # noqa: SLF001 -- PyInstaller's documented API
    return Path(__file__).resolve().parents[2]


def asset_dir() -> Path:
    """``assets/`` -- the 22 card/king JPGs."""
    return resource_dir() / "assets"


def model_dir() -> Path:
    """``models/`` -- checkpoints. The frozen build ships only ``models/release/*.npz`` (torch-free)."""
    return resource_dir() / "models"


def model_path(rel: str) -> str:
    """Resolve a repo-relative model path (``"models/release/attn_d64_L2.npz"``) against the bundle root.

    Accepts the historical CWD-relative strings verbatim, so callers keep their readable literals and the
    frozen build still finds the file. An ABSOLUTE path is returned untouched -- a user pointing --nn at a
    checkpoint outside the bundle must still work."""
    if os.path.isabs(rel):
        return rel
    p = rel.replace("\\", "/")
    if p.startswith("models/"):
        p = p[len("models/"):]
    return str(model_dir() / p)


def canon(p: str) -> str:
    """A path's canonical identity, for COMPARING two paths that may name the same file.

    Never compare checkpoint paths as raw strings. ``model_path`` builds them with ``pathlib`` (backslashes
    on Windows) while ``discover_ckpts`` normalises to forward slashes, so the same file compares unequal --
    which showed up as a phantom THIRD entry in the Settings model picker, two of whose options were the
    same net. normcase also folds case, which Windows does not distinguish."""
    return os.path.normcase(os.path.abspath(p))


def same_file(a: Optional[str], b: Optional[str]) -> bool:
    return bool(a) and bool(b) and canon(a) == canon(b)


def resolve_ckpt(p: Optional[str]) -> Optional[str]:
    """Resolve a checkpoint the USER named (``--nn`` / ``--attn``, or the frozen entry point's default).

    Prefer the path exactly as given -- a developer running from the repo root, or pointing at a checkpoint
    anywhere on disk, must keep working. Only if that does not exist do we re-resolve it against the bundle,
    which is the frozen case: the .exe is launched from the Desktop, so "models/release/x.npz" is relative
    to the wrong directory and would otherwise be silently reported as missing."""
    if not p:
        return p
    return p if os.path.exists(p) else model_path(p)
