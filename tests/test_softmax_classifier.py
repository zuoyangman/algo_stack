import numpy as np
import pytest

from algo_stack.supervised.softmax_classifier import SoftmaxClassifier


def _blobs(seed=0, n=120):
    rng = np.random.default_rng(seed)
    centres = np.array([[-3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    X = np.vstack([c + rng.normal(scale=0.4, size=(n, 2)) for c in centres])
    y = np.repeat(np.arange(3), n)
    return X, y


@pytest.mark.parametrize("optimizer", ["sgd", "momentum", "adam"])
def test_multiclass_accuracy(optimizer):
    X, y = _blobs()
    clf = SoftmaxClassifier(
        optimizer=optimizer, learning_rate=0.1, n_epochs=300, random_state=0
    ).fit(X, y)
    assert clf.coef_.shape == (3, 2)
    assert clf.score(X, y) > 0.95
    p = clf.predict_proba(X)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-8)


def test_loss_decreases():
    X, y = _blobs()
    clf = SoftmaxClassifier(n_epochs=100, random_state=0).fit(X, y)
    assert clf.loss_curve_[-1] < clf.loss_curve_[0]


def test_l2_shrinks_weights():
    X, y = _blobs()
    a = SoftmaxClassifier(l2=0.0, n_epochs=100, random_state=0).fit(X, y)
    b = SoftmaxClassifier(l2=10.0, n_epochs=100, random_state=0).fit(X, y)
    assert np.linalg.norm(b.coef_) < np.linalg.norm(a.coef_)
