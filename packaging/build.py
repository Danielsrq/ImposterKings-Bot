"""Build the ImposterKings release. Run from the repo root:

    py packaging/build.py

Does the whole thing: export the weights to torch-free .npz, freeze with PyInstaller, delete PyInstaller's
scratch dir, self-check the built .exe, and zip it.

The scratch-dir deletion is not tidiness. PyInstaller leaves a bare bootloader stub at
``build/imposterkings/ImposterKings.exe`` -- an exe with the SAME NAME as the real one but no ``_internal``
beside it, so double-clicking it dies with "Failed to load Python DLL ... python310.dll". Two identically
named executables, one of which cannot run, is a trap; this script removes it so ``dist/`` is the only
place an .exe exists.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
RELEASE = ROOT / "models" / "release"
CKPTS = [ROOT / "models" / "gen1_v3c_v2feat" / "attn_d64_L2.pt",   # the bot + the attention drawer
         ROOT / "models" / "mlp_256.pt"]                          # the fast alternative (Settings)


def run(*cmd, cwd=None, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd or ROOT, **kw)


def main() -> int:
    missing = [c for c in CKPTS if not c.exists()]
    if missing:
        print(f"missing checkpoints (models/ is gitignored -- they are not in the repo):\n  "
              + "\n  ".join(str(m) for m in missing))
        return 1

    # 1. weights -> .npz. The shipped game has no torch, so it cannot read a .pt (a torch pickle).
    RELEASE.mkdir(parents=True, exist_ok=True)
    run(sys.executable, "-m", "imposterkings.machine_learning.export_npz", *CKPTS, "--out-dir", RELEASE)

    # 2. freeze. workpath goes to a TEMP dir so no stray ImposterKings.exe is ever left in the repo.
    shutil.rmtree(DIST, ignore_errors=True)
    with tempfile.TemporaryDirectory() as work:
        run(sys.executable, "-m", "PyInstaller", "packaging/imposterkings.spec",
            "--noconfirm", "--distpath", DIST, "--workpath", work)

    app = DIST / "ImposterKings" / "ImposterKings.exe"
    assert app.exists(), app
    assert not (ROOT / "build").exists(), "PyInstaller scratch dir survived -- the decoy .exe is back"

    # 3. prove the BUILD works, not just the source: run its self-check from a DIFFERENT directory, which is
    #    what catches paths resolved against the cwd rather than the bundle.
    report = DIST / "selfcheck.txt"
    run(app, "--selfcheck", report, cwd=tempfile.gettempdir())
    print("\n" + report.read_text(encoding="utf-8"))
    if "FAIL" in report.read_text(encoding="utf-8"):
        return 1

    # 4. ship it
    zip_base = DIST / "ImposterKings-release_1"
    shutil.make_archive(str(zip_base), "zip", root_dir=DIST, base_dir="ImposterKings")
    size = (zip_base.with_suffix(".zip")).stat().st_size / 1e6
    print(f"\nOK  {zip_base.with_suffix('.zip')}  ({size:.1f} MB)")
    print(f"    run it with: {app}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
