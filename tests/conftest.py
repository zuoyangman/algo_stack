"""Make ``algo_stack`` importable from the source tree without installing.

Some CI environments don't run ``pip install -e .`` before tests; this file
adds the repo root to ``sys.path`` so ``import algo_stack`` succeeds either
way.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
