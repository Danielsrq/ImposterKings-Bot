"""Data-science layer: dataset generation, search/budget scaling studies, and their reporting.

Kept separate from the core game engine (state / rules / MCTS). Modules here import *up* into the
engine (``from ..state import ...``) but the engine never imports back down into ``analysis``.

    python -m imposterkings.data_analysis.search_scaling      # MCTS@N vs a fixed-N baseline
    python -m imposterkings.data_analysis.budget_scaling      # hybrid MCTS@k,l sweep vs fixed references
"""
