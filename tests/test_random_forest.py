import numpy as np

from algo_stack.supervised.decision_tree import DecisionTreeClassifier
from algo_stack.supervised.random_forest import (
    RandomForestClassifier,
    RandomForestRegressor,
)


def _noisy_blobs(seed=0, noise=0.2, n_per=80):
    rng = np.random.default_rng(seed)
    X = np.vstack(
        [
            rng.normal([-2.0, 0.0], 0.9, size=(n_per, 2)),
            rng.normal([2.0, 0.0], 0.9, size=(n_per, 2)),
        ]
    )
    y = np.repeat([0, 1], n_per)
    flip = rng.random(len(y)) < noise
    y = np.where(flip, 1 - y, y)
    return X, y


def test_rf_better_or_more_stable_than_tree_on_noise():
    """RF should beat a single deep tree on noisy labels (test accuracy)."""
    X, y = _noisy_blobs(noise=0.25)
    # Hold out last 40% as a crude test split for stability comparison.
    n = X.shape[0]
    cut = int(0.6 * n)
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]

    tree_scores = []
    rf_scores = []
    for seed in range(5):
        tree = DecisionTreeClassifier(max_depth=None, random_state=seed).fit(Xtr, ytr)
        rf = RandomForestClassifier(
            n_estimators=25, max_depth=None, random_state=seed
        ).fit(Xtr, ytr)
        tree_scores.append(tree.score(Xte, yte))
        rf_scores.append(rf.score(Xte, yte))

    assert np.mean(rf_scores) >= np.mean(tree_scores) - 0.02
    # RF should also be at least as stable (not much higher std).
    assert np.std(rf_scores) <= np.std(tree_scores) + 0.05


def test_rf_proba_and_estimators():
    X, y = _noisy_blobs(noise=0.05, n_per=40)
    clf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=0).fit(X, y)
    assert len(clf.estimators_) == 10
    p = clf.predict_proba(X)
    assert p.shape == (X.shape[0], 2)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-8)
    assert clf.score(X, y) > 0.85


def test_rf_regressor_basic():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(120, 1))
    y = X.ravel() ** 2 + 0.05 * rng.normal(size=120)
    reg = RandomForestRegressor(
        n_estimators=20, max_depth=5, max_features=1, random_state=0
    ).fit(X, y)
    assert reg.score(X, y) > 0.85
