import numpy as np
import pytest

from algo_stack.supervised.gaussian_nb import GaussianNB


def _blobs(seed=0, n=100):
    rng = np.random.default_rng(seed)
    centres = np.array([[-2.5, 0.0], [2.5, 0.0], [0.0, 2.5]])
    X = np.vstack([c + rng.normal(scale=0.6, size=(n, 2)) for c in centres])
    y = np.repeat(np.arange(3), n)
    return X, y


def test_multiclass_high_accuracy():
    X, y = _blobs()
    clf = GaussianNB().fit(X, y)
    assert clf.theta_.shape == (3, 2)
    assert clf.var_.shape == (3, 2)
    assert clf.class_prior_.shape == (3,)
    np.testing.assert_allclose(clf.class_prior_.sum(), 1.0, atol=1e-8)
    assert clf.score(X, y) > 0.95
    p = clf.predict_proba(X)
    assert p.shape == (X.shape[0], 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-8)


def test_means_near_centres():
    X, y = _blobs(seed=1)
    clf = GaussianNB().fit(X, y)
    expected = np.array([[-2.5, 0.0], [2.5, 0.0], [0.0, 2.5]])
    # Classes are ordered by unique labels 0,1,2 matching centres order.
    np.testing.assert_allclose(clf.theta_, expected, atol=0.25)


def test_string_labels():
    rng = np.random.default_rng(0)
    X0 = rng.normal(loc=-2, size=(50, 2))
    X1 = rng.normal(loc=+2, size=(50, 2))
    X = np.vstack([X0, X1])
    y = np.array(["neg"] * 50 + ["pos"] * 50)
    clf = GaussianNB().fit(X, y)
    assert set(clf.predict(X)) <= {"neg", "pos"}
    assert clf.score(X, y) > 0.9


def test_single_class_raises():
    X = np.ones((10, 2))
    y = np.zeros(10)
    with pytest.raises(ValueError):
        GaussianNB().fit(X, y)


def test_predict_feature_mismatch():
    X, y = _blobs()
    clf = GaussianNB().fit(X, y)
    with pytest.raises(ValueError):
        clf.predict(X[:, :1])
