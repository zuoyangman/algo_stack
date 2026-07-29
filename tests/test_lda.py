import numpy as np
import pytest

from algo_stack.supervised.lda import LinearDiscriminantAnalysis


def _blobs(seed=0, n=100):
    rng = np.random.default_rng(seed)
    centres = np.array([[-3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    X = np.vstack([c + rng.normal(scale=0.6, size=(n, 2)) for c in centres])
    y = np.repeat(np.arange(3), n)
    return X, y


def _binary(seed=0, n=120):
    rng = np.random.default_rng(seed)
    X0 = rng.normal(loc=(-2, 0), scale=0.7, size=(n, 2))
    X1 = rng.normal(loc=(+2, 0), scale=0.7, size=(n, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * n + [1] * n)
    return X, y


def test_multiclass_high_accuracy():
    X, y = _blobs()
    clf = LinearDiscriminantAnalysis().fit(X, y)
    assert clf.means_.shape == (3, 2)
    assert clf.cov_.shape == (2, 2)
    assert clf.priors_.shape == (3,)
    assert clf.coef_.shape == (3, 2)
    assert clf.intercept_.shape == (3,)
    assert clf.score(X, y) > 0.95
    p = clf.predict_proba(X)
    assert p.shape == (X.shape[0], 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-8)


def test_binary_high_accuracy():
    X, y = _binary()
    clf = LinearDiscriminantAnalysis().fit(X, y)
    assert clf.score(X, y) > 0.97


def test_custom_priors():
    X, y = _binary()
    clf = LinearDiscriminantAnalysis(priors=np.array([0.9, 0.1])).fit(X, y)
    np.testing.assert_allclose(clf.priors_, [0.9, 0.1])


def test_shrinkage():
    X, y = _blobs()
    clf = LinearDiscriminantAnalysis(shrinkage=0.5).fit(X, y)
    assert clf.score(X, y) > 0.9


def test_string_labels():
    X, y = _binary()
    y = np.array(["a"] * (len(y) // 2) + ["b"] * (len(y) // 2))
    clf = LinearDiscriminantAnalysis().fit(X, y)
    pred = clf.predict(X)
    assert set(pred) <= {"a", "b"}


def test_single_class_raises():
    X = np.ones((10, 2))
    y = np.zeros(10)
    with pytest.raises(ValueError):
        LinearDiscriminantAnalysis().fit(X, y)
