import numpy as np
import pytest

from algo_stack.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    mean_squared_error,
    precision_recall_f1,
    r2_score,
    silhouette_score,
)
from algo_stack.utils.metrics import (
    accuracy_score as util_accuracy,
    mean_squared_error as util_mse,
    r2_score as util_r2,
)


def test_reexports_match_utils():
    y = np.array([1.0, 2.0, 3.0])
    assert mean_squared_error(y, y) == util_mse(y, y)
    assert r2_score(y, y) == util_r2(y, y)
    yt = np.array([0, 1, 1])
    yp = np.array([0, 0, 1])
    assert accuracy_score(yt, yp) == util_accuracy(yt, yp)


def test_precision_recall_f1_perfect():
    y = np.array([0, 1, 1, 0, 1])
    p, r, f1 = precision_recall_f1(y, y)
    assert p == 1.0 and r == 1.0 and f1 == 1.0


def test_precision_recall_f1_known():
    # TP=1, FP=1, FN=1 → P=R=F1=0.5
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    p, r, f1 = precision_recall_f1(y_true, y_pred)
    assert p == pytest.approx(0.5)
    assert r == pytest.approx(0.5)
    assert f1 == pytest.approx(0.5)


def test_confusion_matrix_binary():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # TN=2, FP=0, FN=1, TP=2
    np.testing.assert_array_equal(cm, [[2, 0], [1, 2]])


def test_silhouette_well_separated():
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(size=(40, 2)),
        rng.normal(size=(40, 2)) + 10.0,
    ])
    labels = np.array([0] * 40 + [1] * 40)
    score = silhouette_score(X, labels)
    assert score > 0.8


def test_silhouette_requires_two_clusters():
    X = np.random.default_rng(0).normal(size=(10, 2))
    with pytest.raises(ValueError):
        silhouette_score(X, np.zeros(10, dtype=int))


def test_log_loss_binary_perfect():
    y = np.array([0, 1, 1, 0])
    # Near-certain correct probs → very small loss
    p = np.array([0.01, 0.99, 0.99, 0.01])
    assert log_loss(y, p) < 0.02


def test_log_loss_multiclass():
    y = np.array([0, 1, 2])
    P = np.eye(3)
    assert log_loss(y, P) == pytest.approx(0.0, abs=1e-12)


def test_log_loss_worse_than_perfect():
    y = np.array([0, 1])
    good = log_loss(y, np.array([0.1, 0.9]))
    bad = log_loss(y, np.array([0.9, 0.1]))
    assert bad > good
