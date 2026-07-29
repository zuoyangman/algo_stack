import numpy as np
import pytest

from algo_stack.unsupervised.hierarchical import AgglomerativeClustering


def _three_blobs(seed=0, n=25):
    rng = np.random.default_rng(seed)
    centres = np.array([[-4.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    X = np.vstack([c + rng.normal(scale=0.25, size=(n, 2)) for c in centres])
    y = np.repeat(np.arange(3), n)
    return X, y


def _cluster_purity(labels, y):
    """Fraction of points whose cluster majority matches true label (greedy)."""
    correct = 0
    for c in np.unique(labels):
        mask = labels == c
        vals, counts = np.unique(y[mask], return_counts=True)
        correct += int(counts.max())
    return correct / len(y)


@pytest.mark.parametrize("linkage", ["ward", "average", "complete", "single"])
def test_recovers_three_blobs(linkage):
    X, y = _three_blobs()
    model = AgglomerativeClustering(n_clusters=3, linkage=linkage).fit(X)
    assert model.labels_.shape == (X.shape[0],)
    assert set(model.labels_) == {0, 1, 2}
    assert model.n_clusters_ == 3
    assert _cluster_purity(model.labels_, y) >= 0.9


def test_children_shape():
    X, _ = _three_blobs(n=10)
    model = AgglomerativeClustering(n_clusters=2, linkage="ward").fit(X)
    # n - n_clusters merges
    assert model.children_.shape == (X.shape[0] - 2, 2)


def test_fit_predict():
    X, _ = _three_blobs()
    model = AgglomerativeClustering(n_clusters=3, linkage="average")
    labels = model.fit_predict(X)
    np.testing.assert_array_equal(labels, model.labels_)


def test_invalid_linkage():
    X, _ = _three_blobs()
    with pytest.raises(ValueError):
        AgglomerativeClustering(linkage="median").fit(X)


def test_n_clusters_too_large():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError):
        AgglomerativeClustering(n_clusters=5).fit(X)
