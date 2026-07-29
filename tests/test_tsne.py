import numpy as np
import pytest

from algo_stack.unsupervised.tsne import TSNE


def test_embedding_shape_and_finite():
    rng = np.random.default_rng(0)
    X = np.vstack(
        [
            rng.normal([-2, 0], 0.3, size=(25, 2)),
            rng.normal([2, 0], 0.3, size=(25, 2)),
        ]
    )
    tsne = TSNE(
        n_components=2,
        perplexity=10,
        n_iter=250,
        learning_rate=100,
        random_state=0,
    )
    Y = tsne.fit_transform(X)
    assert Y.shape == (50, 2)
    assert np.all(np.isfinite(Y))
    assert np.isfinite(tsne.kl_divergence_)


def test_separates_two_blobs():
    rng = np.random.default_rng(1)
    X = np.vstack(
        [
            rng.normal([-5, 0, 0], 0.4, size=(30, 3)),
            rng.normal([5, 0, 0], 0.4, size=(30, 3)),
        ]
    )
    Y = TSNE(
        n_components=2,
        perplexity=12,
        n_iter=500,
        learning_rate=100,
        random_state=0,
    ).fit_transform(X)
    # Centroids of the two groups should be farther apart than within-group spread.
    c0, c1 = Y[:30].mean(axis=0), Y[30:].mean(axis=0)
    between = np.linalg.norm(c0 - c1)
    within = 0.5 * (
        np.mean(np.linalg.norm(Y[:30] - c0, axis=1))
        + np.mean(np.linalg.norm(Y[30:] - c1, axis=1))
    )
    assert between > 2.0 * within


def test_reproducibility():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 4))
    kwargs = dict(
        n_components=2, perplexity=10, n_iter=200, learning_rate=100, random_state=42
    )
    a = TSNE(**kwargs).fit_transform(X)
    b = TSNE(**kwargs).fit_transform(X)
    np.testing.assert_allclose(a, b)


def test_perplexity_too_large_raises():
    X = np.random.default_rng(0).normal(size=(10, 2))
    with pytest.raises(ValueError):
        TSNE(perplexity=10).fit_transform(X)


def test_transform_raises():
    X = np.random.default_rng(0).normal(size=(20, 2))
    tsne = TSNE(perplexity=5, n_iter=50, random_state=0).fit(X)
    with pytest.raises(RuntimeError):
        tsne.transform(X)
