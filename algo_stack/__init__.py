"""algo_stack — from-scratch ML/data-science algorithms.

Top-level conveniences. Import specific algorithms from their submodules, e.g.::

    from algo_stack.supervised.linear_regression import LinearRegression
    from algo_stack.unsupervised.kmeans import KMeans
"""

from algo_stack._base import (
    BaseEstimator,
    ClassifierMixin,
    ClusterMixin,
    RegressorMixin,
    TransformerMixin,
)

__all__ = [
    "BaseEstimator",
    "RegressorMixin",
    "ClassifierMixin",
    "ClusterMixin",
    "TransformerMixin",
]

__version__ = "0.1.0"
