import numpy as np
import pytest

from algo_stack.unsupervised.kmeans import KMeans


def _three_blobs(seed=0, n=120):
    rng = np.random.default_rng(seed)
    centres = np.array([[-5.0, -5.0], [0.0, 5.0], [5.0, -5.0]])
    X = np.vstack([c + rng.normal(scale=0.4, size=(n, 2)) for c in centres])
    y = np.repeat(np.arange(3), n)
    return X, y, centres


def test_recovers_centres():
    X, y, centres_true = _three_blobs()
    km = KMeans(n_clusters=3, random_state=0, n_init=10).fit(X)
    assert km.cluster_centers_.shape == centres_true.shape

    # Match found centres to true centres greedily by distance.
    found = km.cluster_centers_.copy()
    matched_dists = []
    for c in centres_true:
        d = np.linalg.norm(found - c, axis=1)
        i = int(np.argmin(d))
        matched_dists.append(d[i])
        found[i] = np.array([np.inf, np.inf])  # mark as used
    assert max(matched_dists) < 0.5


def test_labels_partition_data():
    X, _, _ = _three_blobs()
    km = KMeans(n_clusters=3, random_state=0).fit(X)
    assert km.labels_.shape == (X.shape[0],)
    assert set(np.unique(km.labels_)) <= {0, 1, 2}


def test_inertia_decreases_with_more_clusters():
    X, _, _ = _three_blobs()
    inertias = [
        KMeans(n_clusters=k, random_state=0, n_init=5).fit(X).inertia_
        for k in (1, 2, 3, 5)
    ]
    assert all(a >= b - 1e-6 for a, b in zip(inertias, inertias[1:]))


def test_predict_assigns_consistently():
    X, _, _ = _three_blobs()
    km = KMeans(n_clusters=3, random_state=0).fit(X)
    np.testing.assert_array_equal(km.predict(X), km.labels_)


def test_fit_predict_returns_labels():
    X, _, _ = _three_blobs()
    km = KMeans(n_clusters=3, random_state=0)
    labels = km.fit_predict(X)
    np.testing.assert_array_equal(labels, km.labels_)


def test_n_clusters_too_large_raises():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError):
        KMeans(n_clusters=5).fit(X)


def test_random_init_works():
    X, _, _ = _three_blobs()
    km = KMeans(n_clusters=3, init="random", random_state=0, n_init=10).fit(X)
    assert km.cluster_centers_.shape == (3, 2)


def test_reproducibility_via_random_state():
    X, _, _ = _three_blobs()
    a = KMeans(n_clusters=3, random_state=42, n_init=4).fit(X)
    b = KMeans(n_clusters=3, random_state=42, n_init=4).fit(X)
    np.testing.assert_allclose(a.cluster_centers_, b.cluster_centers_)
    assert a.inertia_ == pytest.approx(b.inertia_)
