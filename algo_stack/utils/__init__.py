"""Shared utilities used by multiple algorithms.

Only put **algorithm-agnostic** helpers here. Anything specific to a single
algorithm belongs in that algorithm's folder.
"""

from algo_stack.utils.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
)
from algo_stack.utils.preprocessing import StandardScaler, train_test_split
from algo_stack.utils.validation import (
    check_array,
    check_random_state,
    check_X_y,
)

__all__ = [
    "accuracy_score",
    "mean_squared_error",
    "r2_score",
    "StandardScaler",
    "train_test_split",
    "check_array",
    "check_X_y",
    "check_random_state",
]
