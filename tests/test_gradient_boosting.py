import numpy as np
import pytest

from algo_stack.supervised.gradient_boosting import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)


def test_gb_regressor_noisy_sine_r2():
    rng = np.random.default_rng(0)
    X = rng.uniform(-np.pi, np.pi, size=(250, 1))
    y = np.sin(X.ravel()) + 0.12 * rng.normal(size=250)
    reg = GradientBoostingRegressor(
        n_estimators=60, learning_rate=0.1, max_depth=3, random_state=0
    ).fit(X, y)
    assert reg.score(X, y) > 0.9
    assert len(reg.estimators_) == 60


def test_gb_classifier_blobs():
    rng = np.random.default_rng(1)
    X = np.vstack(
        [
            rng.normal([-2.0, 0.0], 0.55, size=(70, 2)),
            rng.normal([2.0, 0.0], 0.55, size=(70, 2)),
        ]
    )
    y = np.repeat([0, 1], 70)
    clf = GradientBoostingClassifier(
        n_estimators=40, learning_rate=0.1, max_depth=2, random_state=0
    ).fit(X, y)
    assert clf.score(X, y) > 0.95
    p = clf.predict_proba(X)
    assert p.shape == (X.shape[0], 2)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-8)


def test_gb_classifier_rejects_multiclass():
    X = np.zeros((6, 2))
    y = np.array([0, 1, 2, 0, 1, 2])
    with pytest.raises(ValueError, match="binary"):
        GradientBoostingClassifier(n_estimators=5).fit(X, y)
