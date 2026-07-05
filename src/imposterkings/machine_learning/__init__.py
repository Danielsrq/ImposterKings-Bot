"""Machine-learning layer: the shared featurizer, corpus->tensor builder, and torch models/training.

Imports the engine (``from ..state import ...``) and the self-play corpus produced by
``imposterkings.data_analysis.datagen``; the engine never imports back into here. ``features`` and
``dataset`` are numpy-only (importable without torch); ``mlp``/``train`` require torch (declared under
the project's ``[ml]`` optional-deps). Kept torch-free at import time on purpose.

    python -m imposterkings.machine_learning.dataset --data datasets/selfplay_k20l3 --out datasets/tensors/k20l3.npz
    python -m imposterkings.machine_learning.train   --npz datasets/tensors/k20l3.npz --sweep "16;32;64"
"""
