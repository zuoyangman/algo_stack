import numpy as np

from algo_stack.supervised.decision_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)


def _blobs(seed=0, n_per=60):
    rng = np.random.default_rng(seed)
    X = np.vstack(
        [
            rng.normal([-2.5, 0.0], 0.45, size=(n_per, 2)),
            rng.normal([2.5, 0.0], 0.45, size=(n_per, 2)),
        ]
    )
    y = np.repeat([0, 1], n_per)
    return X, y


def test_xor_needs_depth_two():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    shallow = DecisionTreeClassifier(max_depth=1).fit(X, y)
    deep = DecisionTreeClassifier(max_depth=2).fit(X, y)
    assert shallow.score(X, y) < 1.0
    assert deep.score(X, y) == 1.0
    np.testing.assert_array_equal(deep.predict(X), y)


def test_blobs_high_accuracy():
    X, y = _blobs()
    clf = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X, y)
    assert clf.score(X, y) > 0.95
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)


def test_regressor_piecewise():
    X = np.linspace(-1, 1, 40).reshape(-1, 1)
    y = np.where(X.ravel() < 0, -1.0, 1.0)
    reg = DecisionTreeRegressor(max_depth=2).fit(X, y)
    pred = reg.predict(X)
    assert np.mean(np.abs(pred - y) < 0.1) > 0.9


def test_tree_structure_attrs():
    X, y = _blobs(n_per=20)
    clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    root = clf.tree_
    assert root.n_samples == X.shape[0]
    assert hasattr(root, "feature")
    assert hasattr(root, "value")
