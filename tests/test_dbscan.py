import numpy as np
import pytest

from algo_stack.unsupervised.dbscan import DBSCAN


def test_two_blobs_and_noise():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(60, 2)) * 0.3 + np.array([-4.0, 0.0])
    B = rng.normal(size=(60, 2)) * 0.3 + np.array([4.0, 0.0])
    noise = np.array([[0.0, 8.0], [0.0, -8.0], [8.0, 8.0]])
    X = np.vstack([A, B, noise])
    db = DBSCAN(eps=0.9, min_samples=5).fit(X)
    labels = db.labels_
    clusters = set(labels) - {-1}
    assert len(clusters) == 2
    assert np.sum(labels == -1) >= 1
    assert len(db.core_sample_indices_) > 0


def test_all_noise_when_eps_tiny():
    X = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    db = DBSCAN(eps=0.1, min_samples=2).fit(X)
    np.testing.assert_array_equal(db.labels_, [-1, -1, -1])


def test_single_dense_cluster():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 2)) * 0.2
    db = DBSCAN(eps=1.0, min_samples=3).fit(X)
    assert set(db.labels_) == {0}


def test_fit_predict():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 2)) * 0.2
    db = DBSCAN(eps=1.0, min_samples=3)
    labels = db.fit_predict(X)
    np.testing.assert_array_equal(labels, db.labels_)


def test_invalid_eps():
    with pytest.raises(ValueError):
        DBSCAN(eps=0).fit(np.zeros((5, 2)))
