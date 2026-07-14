"""ImposterKings -- the shipped game's entry point (what PyInstaller freezes into ImposterKings.exe).

Deliberately thin: no dev CLI. It seats you against the NN+MCTS bot on the v3c attention net and opens the
window. Everything else is in-game -- Settings switches the engine and cycles the model (the mlp_256 net is
the fast alternative), How to play explains the rules, and A opens the attention drawer.

The bot and the drawer both run in pure numpy (``machine_learning/npz_infer``) off ``models/release/*.npz``,
so this build carries no torch. See ``packaging/imposterkings.spec``.
"""
import multiprocessing
import sys

from imposterkings.ui.app import run

ATTN = "models/release/attn_d64_L2.npz"


def selfcheck(out_path: str) -> int:
    """``ImposterKings.exe --selfcheck <file>`` -- prove the BUILD works, and write the findings to a file.

    A windowed build has no console, so a launched-and-did-not-crash test cannot tell you whether the card
    art and the weights actually loaded or whether the app quietly degraded (the whole codebase is written
    to degrade rather than die). This resolves every runtime path, loads both nets, and plays a bot move --
    the same things the game does -- and reports each one. Also the first thing to ask a user for when they
    report that something looks wrong."""
    import numpy as np
    from imposterkings import paths
    from imposterkings.actions import StepKind
    from imposterkings.agents import MCTSAgent
    from imposterkings.machine_learning.benchmark import _evaluator_for
    from imposterkings.machine_learning.explain import explain
    from imposterkings.machine_learning.npz_infer import load as load_npz
    from imposterkings.state import GameState
    from imposterkings.ui.app import discover_ckpts

    lines, ok = [], True

    def check(label, fn):
        nonlocal ok
        try:
            lines.append(f"  OK    {label}: {fn()}")
        except Exception as e:                        # noqa: BLE001
            ok = False
            lines.append(f"  FAIL  {label}: {e!r}")

    lines.append(f"frozen={paths.is_frozen()}  root={paths.resource_dir()}")
    lines.append(f"cwd={__import__('os').getcwd()}   <- deliberately not the install dir when testing")
    check("torch absent (the point of the build)", lambda: "torch" not in sys.modules)
    check("card art", lambda: f"{len(list(paths.asset_dir().glob('*.jpg')))} jpgs")
    check("checkpoints found", lambda: [p.split('/')[-1] for p in discover_ckpts()])
    check("attention net loads", lambda: f"L={load_npz(paths.model_path(ATTN)).cfg.n_layers}")

    st = GameState.deal(np.random.default_rng(0), starting_player=0)
    while st.phase in (StepKind.SETUP_HIDE, StepKind.SETUP_DISCARD):
        st = st.apply(st.legal_moves()[0])
    view = st.information_set(st.to_play)

    def _move():
        agent = MCTSAgent(iterations=30, evaluator=_evaluator_for(paths.model_path(ATTN)))
        return agent.select_move(view, np.random.default_rng(0))

    check("bot picks a move (numpy NN+MCTS)", _move)
    check("attention drawer explains a move",
          lambda: f"q={explain(view, view.legal_moves()[0], load_npz(paths.model_path(ATTN)), attribution=True).q:+.3f}")

    report = "\n".join([f"ImposterKings self-check: {'PASS' if ok else 'FAIL'}"] + lines)
    print(report)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()      # PyInstaller: without this, any spawn would re-launch the game
    if len(sys.argv) > 2 and sys.argv[1] == "--selfcheck":
        sys.exit(selfcheck(sys.argv[2]))
    try:
        run(p1="hybrid", k=20, l=3, nn=ATTN, attn=ATTN)
    except Exception:                     # a frozen app has no console -> show the traceback rather than
        import traceback                  # vanishing silently, which is indistinguishable from a crash
        traceback.print_exc()
        try:
            import tkinter.messagebox as mb
            mb.showerror("ImposterKings", traceback.format_exc()[-1500:])
        except Exception:                 # noqa: BLE001 -- tkinter may not be bundled; the console print stands
            pass
        sys.exit(1)
