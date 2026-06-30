"""Make ``src/`` importable in a fresh checkout even without an editable install.

Mirrors bigtwo's conftest: prepend the src-layout package root to ``sys.path`` so
``import imposterkings`` resolves when running ``pytest`` directly.
"""
import os
import sys

SRC = os.path.join(os.path.dirname(__file__), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
