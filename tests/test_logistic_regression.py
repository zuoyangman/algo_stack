import numpy as np
import pytest

from algo_stack.supervised.logistic_regression import LogisticRegression


def _binary(seed=0, n=200):
    rng = np.random.default_rng(seed)
    X0 = rng.normal(loc=(-2, 0), scale=0.7, size=(n, 2))
    X1 = rng.normal(loc=(+2, 0), scale=0.7, size=(n, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * n + [1] * n)
    return X, y


def _three_class(seed=0, n=120):
    rng = np.random.default_rng(seed)
    centres = np.array([[-3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    X = np.vstack([c + rng.normal(scale=0.5, size=(n, 2)) for c in centres])
    y = np.repeat(np.arange(3), n)
    return X, y


def test_binary_classification_high_accuracy():
    X, y = _binary()
    clf = LogisticRegression(learning_rate=0.5, n_iter=500).fit(X, y)
    assert clf.coef_.shape == (2,)
    assert isinstance(clf.intercept_, float)
    assert clf.score(X, y) > 0.97
    p = clf.predict_proba(X)
    assert p.shape == (X.shape[0], 2)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-8)


def test_multinomial_classification_high_accuracy():
    X, y = _three_class()
    clf = LogisticRegression(learning_rate=0.5, n_iter=800).fit(X, y)
    assert clf.coef_.shape == (3, 2)
    assert clf.intercept_.shape == (3,)
    assert clf.score(X, y) > 0.97
    p = clf.predict_proba(X)
    assert p.shape == (X.shape[0], 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-8)


def test_l2_shrinks_weights():
    X, y = _binary()
    a = LogisticRegression(learning_rate=0.5, n_iter=500, l2=0.0).fit(X, y)
    b = LogisticRegression(learning_rate=0.5, n_iter=500, l2=10.0).fit(X, y)
    assert np.linalg.norm(b.coef_) < np.linalg.norm(a.coef_)


def test_loss_history_monotone_on_average():
    X, y = _binary()
    clf = LogisticRegression(learning_rate=0.5, n_iter=200).fit(X, y)
    losses = clf.loss_history_
    assert losses[-1] < losses[0]
    assert clf.n_iter_ == len(losses)


def test_predict_uses_class_labels():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
    y = np.array(["cat", "cat", "dog", "dog"])
    clf = LogisticRegression(learning_rate=0.5, n_iter=300).fit(X, y)
    pred = clf.predict(X)
    assert set(pred) <= {"cat", "dog"}
    assert clf.score(X, y) == 1.0


def test_single_class_raises():
    X = np.ones((10, 2))
    y = np.zeros(10)
    with pytest.raises(ValueError):
        LogisticRegression().fit(X, y)
