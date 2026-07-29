"""Evaluation metrics package.

Re-exports the core scores from ``algo_stack.utils.metrics`` and adds
classification / clustering helpers. Prefer importing from here when writing
evaluation code::

    from algo_stack.metrics import accuracy_score, silhouette_score
"""

from algo_stack.metrics._scores import (
    confusion_matrix,
    log_loss,
    precision_recall_f1,
    silhouette_score,
)
from algo_stack.utils.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
)

__all__ = [
    "accuracy_score",
    "mean_squared_error",
    "r2_score",
    "precision_recall_f1",
    "confusion_matrix",
    "silhouette_score",
    "log_loss",
]
