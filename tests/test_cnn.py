import numpy as np

from algo_stack.supervised.cnn import CNNClassifier


def _pattern_data(n=60, size=8):
    rng = np.random.default_rng(0)
    X0, X1 = [], []
    for _ in range(n):
        img0 = rng.normal(0, 0.1, size=(size, size))
        img0[size // 2, :] += 2.0
        X0.append(img0)
        img1 = rng.normal(0, 0.1, size=(size, size))
        img1[:, size // 2] += 2.0
        X1.append(img1)
    X = np.array(X0 + X1)[:, None, :, :]
    y = np.array([0] * n + [1] * n)
    return X, y


def test_learns_bar_patterns():
    X, y = _pattern_data()
    clf = CNNClassifier(
        num_filters=4, n_epochs=60, learning_rate=0.1, random_state=0
    ).fit(X, y)
    assert clf.score(X, y) > 0.95
    assert clf.conv_W_.shape == (4, 1, 3, 3)
    assert clf.fc_W_ is not None


def test_predict_proba_sums_to_one():
    X, y = _pattern_data(n=20)
    clf = CNNClassifier(n_epochs=30, random_state=0).fit(X, y)
    p = clf.predict_proba(X)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-8)
