import numpy as np
import pytest

from algo_stack.supervised.mlp import MLPClassifier, MLPRegressor


def _xor_blobs(seed=0, n_per=80):
    rng = np.random.default_rng(seed)
    centres = np.array([[-2.0, -2.0], [2.0, 2.0], [-2.0, 2.0], [2.0, -2.0]])
    labels = np.array([0, 0, 1, 1])
    X = np.vstack([c + rng.normal(scale=0.3, size=(n_per, 2)) for c in centres])
    y = np.repeat(labels, n_per)
    return X, y


def test_classifier_solves_xor():
    X, y = _xor_blobs()
    clf = MLPClassifier(
        hidden_layer_sizes=(16, 16),
        activation="relu",
        learning_rate=0.05,
        n_epochs=200,
        batch_size=32,
        random_state=0,
    ).fit(X, y)
    assert clf.score(X, y) > 0.98
    proba = clf.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)


def test_classifier_layer_shapes():
    X, y = _xor_blobs(n_per=20)
    clf = MLPClassifier(
        hidden_layer_sizes=(8, 4),
        learning_rate=0.05,
        n_epochs=5,
        random_state=0,
    ).fit(X, y)
    assert len(clf.coefs_) == 3   # 2 hidden + 1 output
    assert clf.coefs_[0].shape == (2, 8)
    assert clf.coefs_[1].shape == (8, 4)
    assert clf.coefs_[2].shape == (4, 2)


def test_regressor_recovers_smooth_function():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(400, 1))
    y = np.sin(X.ravel() * 1.5)
    reg = MLPRegressor(
        hidden_layer_sizes=(32, 32),
        activation="tanh",
        learning_rate=0.01,
        momentum=0.9,
        n_epochs=300,
        batch_size=32,
        tol=1e-7,
        random_state=0,
    ).fit(X, y)
    assert reg.score(X, y) > 0.9
    assert reg.predict(X).shape == (X.shape[0],)


def test_loss_curve_decreases():
    X, y = _xor_blobs()
    clf = MLPClassifier(
        hidden_layer_sizes=(16,),
        learning_rate=0.05,
        n_epochs=50,
        random_state=0,
    ).fit(X, y)
    assert clf.loss_curve_[-1] < clf.loss_curve_[0]


def test_invalid_activation_raises():
    X, y = _xor_blobs(n_per=10)
    with pytest.raises(ValueError):
        MLPClassifier(activation="bogus").fit(X, y)


def test_reproducibility_via_random_state():
    X, y = _xor_blobs(n_per=30)
    a = MLPClassifier(
        hidden_layer_sizes=(8,), n_epochs=30, learning_rate=0.05, random_state=7
    ).fit(X, y)
    b = MLPClassifier(
        hidden_layer_sizes=(8,), n_epochs=30, learning_rate=0.05, random_state=7
    ).fit(X, y)
    for Wa, Wb in zip(a.coefs_, b.coefs_):
        np.testing.assert_allclose(Wa, Wb)


def test_classifier_label_strings():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-3, 0.5, size=(40, 2)), rng.normal(3, 0.5, size=(40, 2))])
    y = np.array(["neg"] * 40 + ["pos"] * 40)
    clf = MLPClassifier(
        hidden_layer_sizes=(8,), n_epochs=50, learning_rate=0.05, random_state=0
    ).fit(X, y)
    pred = clf.predict(X)
    assert set(pred) <= {"neg", "pos"}
    assert clf.score(X, y) > 0.95
