import numpy as np
import pytest

from algo_stack.unsupervised.minibatch_kmeans import MiniBatchKMeans


def _three_blobs(seed=0, n=100):
    rng = np.random.default_rng(seed)
    centres = np.array([[-5.0, -5.0], [0.0, 5.0], [5.0, -5.0]])
    X = np.vstack([c + rng.normal(scale=0.4, size=(n, 2)) for c in centres])
    return X, centres


def test_recovers_centres():
    X, centres_true = _three_blobs()
    mb = MiniBatchKMeans(
        n_clusters=3, batch_size=40, random_state=0, n_init=5, max_iter=80
    ).fit(X)
    found = mb.cluster_centers_.copy()
    matched = []
    for c in centres_true:
        d = np.linalg.norm(found - c, axis=1)
        i = int(np.argmin(d))
        matched.append(d[i])
        found[i] = np.inf
    assert max(matched) < 0.8


def test_predict_matches_labels():
    X, _ = _three_blobs()
    mb = MiniBatchKMeans(n_clusters=3, batch_size=50, random_state=1).fit(X)
    np.testing.assert_array_equal(mb.predict(X), mb.labels_)


def test_fit_predict():
    X, _ = _three_blobs()
    mb = MiniBatchKMeans(n_clusters=3, batch_size=50, random_state=0)
    labels = mb.fit_predict(X)
    np.testing.assert_array_equal(labels, mb.labels_)


def test_reproducibility():
    X, _ = _three_blobs()
    a = MiniBatchKMeans(n_clusters=3, batch_size=40, random_state=7, n_init=3).fit(X)
    b = MiniBatchKMeans(n_clusters=3, batch_size=40, random_state=7, n_init=3).fit(X)
    np.testing.assert_allclose(a.cluster_centers_, b.cluster_centers_)
    assert a.inertia_ == pytest.approx(b.inertia_)


def test_n_clusters_too_large_raises():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError):
        MiniBatchKMeans(n_clusters=5).fit(X)


def test_transform_shape():
    X, _ = _three_blobs()
    mb = MiniBatchKMeans(n_clusters=3, batch_size=50, random_state=0).fit(X)
    T = mb.transform(X)
    assert T.shape == (X.shape[0], 3)
