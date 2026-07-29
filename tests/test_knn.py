import numpy as np
import pytest

from algo_stack.supervised.knn import KNNClassifier, KNNRegressor


def _blobs(seed=0, n_per_class=60):
    rng = np.random.default_rng(seed)
    centres = np.array([[-3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    X = np.vstack(
        [c + rng.normal(scale=0.4, size=(n_per_class, 2)) for c in centres]
    )
    y = np.repeat(np.arange(3), n_per_class)
    return X, y


def test_classifier_simple_2d():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0]])
    y = np.array(["a", "a", "b", "b"])
    clf = KNNClassifier(n_neighbors=1).fit(X, y)
    np.testing.assert_array_equal(clf.predict([[0.1, 0.1], [10.5, 10.5]]), ["a", "b"])


def test_classifier_blobs_accuracy():
    X, y = _blobs()
    clf = KNNClassifier(n_neighbors=5).fit(X, y)
    assert clf.score(X, y) > 0.98
    p = clf.predict_proba(X)
    assert p.shape == (X.shape[0], 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-8)


def test_distance_weighting_changes_proba():
    X, y = _blobs()
    a = KNNClassifier(n_neighbors=5, weights="uniform").fit(X, y)
    b = KNNClassifier(n_neighbors=5, weights="distance").fit(X, y)
    pa = a.predict_proba(X)
    pb = b.predict_proba(X)
    # Distance-weighted prob for the (very close) own training point should be
    # at least as confident as uniform weighting.
    own_class = y
    assert np.mean(pb[np.arange(len(y)), own_class]) >= np.mean(
        pa[np.arange(len(y)), own_class]
    )


def test_regressor_recovers_function():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(300, 1))
    y = np.sin(X.ravel())
    reg = KNNRegressor(n_neighbors=5).fit(X, y)
    grid = np.linspace(-1.5, 1.5, 30).reshape(-1, 1)
    pred = reg.predict(grid)
    assert np.max(np.abs(pred - np.sin(grid.ravel()))) < 0.2


def test_invalid_metric_or_weights_raises():
    with pytest.raises(ValueError):
        KNNClassifier(weights="bogus").fit(np.zeros((3, 2)), np.array([0, 1, 0]))
    with pytest.raises(ValueError):
        KNNClassifier(metric="manhattan").fit(np.zeros((3, 2)), np.array([0, 1, 0]))


def test_k_larger_than_n_clamped():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1, 0])
    clf = KNNClassifier(n_neighbors=10).fit(X, y)
    # Should use all 3 neighbours -> majority vote -> class 0
    assert int(clf.predict([[1.5]])[0]) == 0
