import numpy as np
import pytest

from algo_stack.supervised.knn import KNNClassifier
from algo_stack.utils.neighbors import BallTree, KDTree


def _data(n=80, d=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    return X


def test_kdtree_matches_brute_force():
    X = _data()
    q = _data(n=10, seed=1)
    tree = KDTree(X)
    d_t, i_t = tree.query(q, k=5)
    # Brute force
    d2 = ((q[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    i_b = np.argsort(d2, axis=1)[:, :5]
    d_b = np.sqrt(np.take_along_axis(d2, i_b, axis=1))
    np.testing.assert_allclose(np.sort(d_t, axis=1), np.sort(d_b, axis=1), atol=1e-8)
    # Same neighbor sets (order may differ on ties)
    for i in range(q.shape[0]):
        assert set(i_t[i].tolist()) == set(i_b[i].tolist())


def test_balltree_matches_brute_force():
    X = _data()
    q = _data(n=10, seed=2)
    tree = BallTree(X, leaf_size=8)
    d_t, i_t = tree.query(q, k=4)
    d2 = ((q[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    i_b = np.argsort(d2, axis=1)[:, :4]
    d_b = np.sqrt(np.take_along_axis(d2, i_b, axis=1))
    np.testing.assert_allclose(np.sort(d_t, axis=1), np.sort(d_b, axis=1), atol=1e-8)
    for i in range(q.shape[0]):
        assert set(i_t[i].tolist()) == set(i_b[i].tolist())


@pytest.mark.parametrize("algorithm", ["brute", "kd_tree", "ball_tree", "auto"])
def test_knn_algorithms_agree(algorithm):
    rng = np.random.default_rng(0)
    centres = np.array([[-3.0, 0.0], [3.0, 0.0]])
    X = np.vstack([c + rng.normal(scale=0.4, size=(40, 2)) for c in centres])
    y = np.repeat([0, 1], 40)
    clf = KNNClassifier(n_neighbors=5, algorithm=algorithm).fit(X, y)
    assert clf.score(X, y) > 0.95
    assert hasattr(clf, "algorithm_")


def test_knn_invalid_algorithm():
    with pytest.raises(ValueError):
        KNNClassifier(algorithm="bogus").fit(np.zeros((3, 2)), np.array([0, 1, 0]))
