import numpy as np
import pytest

from algo_stack.supervised.perceptron import Perceptron


def _linear_blobs(seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal([-2, 0], 0.3, (50, 2)),
                   rng.normal([+2, 0], 0.3, (50, 2))])
    y = np.array([0] * 50 + [1] * 50)
    return X, y


def test_binary_separable():
    X, y = _linear_blobs()
    clf = Perceptron(learning_rate=1.0, n_iter=50, random_state=0).fit(X, y)
    assert clf.score(X, y) == 1.0
    assert clf.coef_.shape == (2,)
    assert clf.mistakes_history_[-1] == 0


def test_multiclass_ovr():
    rng = np.random.default_rng(0)
    centres = np.array([[-3, 0], [0, 3], [3, 0]], dtype=float)
    X = np.vstack([c + rng.normal(scale=0.3, size=(40, 2)) for c in centres])
    y = np.repeat(np.arange(3), 40)
    clf = Perceptron(n_iter=100, random_state=0).fit(X, y)
    assert clf.coef_.shape == (3, 2)
    assert clf.score(X, y) > 0.95


def test_xor_fails_or_partial():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    clf = Perceptron(n_iter=20).fit(X, y)
    assert clf.score(X, y) < 1.0


def test_single_class_raises():
    with pytest.raises(ValueError):
        Perceptron().fit(np.zeros((5, 2)), np.zeros(5))
