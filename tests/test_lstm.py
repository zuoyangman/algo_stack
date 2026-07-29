import numpy as np

from algo_stack.supervised.lstm import GRUClassifier, LSTMClassifier


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


def test_lstm_sequence_classification():
    X, y = _sequence_data()
    clf = LSTMClassifier(
        hidden_size=16, n_epochs=60, learning_rate=0.05, random_state=0
    ).fit(X, y)
    assert clf.score(X, y) > 0.9
    assert clf.W_x_.shape == (4, 64)
    assert clf.W_h_.shape == (16, 64)


def test_gru_sequence_classification():
    X, y = _sequence_data()
    clf = GRUClassifier(
        hidden_size=16, n_epochs=60, learning_rate=0.05, random_state=0
    ).fit(X, y)
    assert clf.score(X, y) > 0.9
    assert clf.W_xz_.shape == (4, 16)


def test_lstm_predict_proba_valid():
    X, y = _sequence_data(n=30)
    clf = LSTMClassifier(hidden_size=8, n_epochs=40, random_state=0).fit(X, y)
    p = clf.predict_proba(X)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-8)


def test_gru_predict_proba_valid():
    X, y = _sequence_data(n=30)
    clf = GRUClassifier(hidden_size=8, n_epochs=40, random_state=0).fit(X, y)
    p = clf.predict_proba(X)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-8)
