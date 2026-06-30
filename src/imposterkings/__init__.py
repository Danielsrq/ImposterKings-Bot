"""ImposterKings: a 2-player hand-clearing card game.

Layered like the sibling ``bigtwo`` project:
- engine core (pure, no I/O): :mod:`cards`, :mod:`rules`, :mod:`actions`, :mod:`abilities`,
  :mod:`state`, :mod:`generate`, :mod:`infoset`.
- search: :mod:`mcts`, :mod:`agents`.
- drivers / IO: :mod:`arena`, :mod:`record`, :mod:`explain`, :mod:`cli`, and the ``ui`` package.

The engine never imports the UI; agents only ever see an :class:`~imposterkings.infoset.InformationSet`.
"""
from __future__ import annotations

__version__ = "0.1.0"
