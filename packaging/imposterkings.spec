# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the ImposterKings release build.

    py -m PyInstaller packaging/imposterkings.spec --noconfirm

Build it from the REPO ROOT, and export the weights first:

    py -m imposterkings.machine_learning.export_npz \
        models/gen1_v3c_v2feat/attn_d64_L2.pt models/mlp_256.pt --out-dir models/release

The two things this spec exists to get right:

1. **torch is excluded.** It is 4.24 GB and serves a 108,737-parameter net. The game reads weights from
   ``models/release/*.npz`` and runs the forward pass in numpy (``machine_learning/npz_infer``), so nothing
   in the shipped code path imports it. ``tests/test_torch_free.py`` proves that by blocking the import and
   playing a move -- if that test passes, this exclusion is safe.

2. **Data travels with the app.** ``assets/`` (22 JPGs) and the .npz weights are declared here and found at
   runtime through ``imposterkings.paths``, which asks PyInstaller where it unpacked them. Every other way
   of locating them (``__file__``-relative, cwd-relative) breaks once frozen.

onedir, not onefile: onefile re-extracts ~50 MB to a temp dir on EVERY launch, which is a visible stall for
a game that otherwise starts instantly. Ship the folder as a .zip.
"""
import os

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# Relative paths in a spec resolve against the WORKING DIRECTORY, not the spec file -- so "../src" silently
# became <parent-of-repo>/src when built from the repo root. Anchor everything to SPECPATH (which
# PyInstaller injects) so the build does not depend on where it was invoked from.
ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))  # noqa: F821 -- SPECPATH is injected by PyInstaller

# Everything heavy that the GAME never touches. torch is the point; the rest are training/analysis-only
# deps that would otherwise be dragged in by a stray transitive import and silently add tens of MB.
EXCLUDES = [
    "torch", "torchvision", "torchaudio",
    "scipy", "matplotlib", "pandas", "joblib", "tqdm", "sklearn",
    "IPython", "jupyter", "notebook", "pytest", "setuptools", "pip",
    "tkinter",                      # run_game's error box degrades gracefully without it
]

a = Analysis(
    [os.path.join(ROOT, "run_game.py")],
    pathex=[os.path.join(ROOT, "src")],
    binaries=[],
    datas=[
        (os.path.join(ROOT, "assets"), "assets"),                    # card art -> <bundle>/assets
        (os.path.join(ROOT, "models", "release"), "models/release"),  # the .npz nets -> <bundle>/models/release
    ],
    hiddenimports=collect_submodules("imposterkings"),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ImposterKings",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,                  # a game, not a CLI -- no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ImposterKings",
)
