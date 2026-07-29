import numpy as np

from algo_stack.supervised.rnn import RNNClassifier


def _sequence_data(n=80, seq_len=10, n_features=4):
    rng = np.random.default_rng(0)
    X0, X1 = [], []
    for _ in range(n):
        trend = np.linspace(-1, 1, seq_len)[:, None]
        X0.append(rng.normal(size=(seq_len, n_features)) + trend)
        X1.append(rng.normal(size=(seq_len, n_features)) - trend)
    X = np.array(X0 + X1)
    y = np.array([0] * n + [1] * n)
    return X, y


def test_sequence_classification():
    X, y = _sequence_data()
    clf = RNNClassifier(
        hidden_size=16, n_epochs=80, learning_rate=0.05, random_state=0
    ).fit(X, y)
    assert clf.score(X, y) > 0.9
    assert clf.W_xh_.shape == (4, 16)
    assert clf.W_hh_.shape == (16, 16)


def test_predict_proba_valid():
    X, y = _sequence_data(n=30)
    clf = RNNClassifier(hidden_size=8, n_epochs=40, random_state=0).fit(X, y)
    p = clf.predict_proba(X)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-8)
