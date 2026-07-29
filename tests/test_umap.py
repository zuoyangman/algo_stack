import numpy as np
import pytest

from algo_stack.unsupervised.umap import UMAP


def test_embedding_shape_and_finite():
    rng = np.random.default_rng(0)
    X = np.vstack(
        [
            rng.normal([-2, 0], 0.3, size=(25, 2)),
            rng.normal([2, 0], 0.3, size=(25, 2)),
        ]
    )
    Y = UMAP(
        n_components=2,
        n_neighbors=10,
        n_epochs=80,
        random_state=0,
    ).fit_transform(X)
    assert Y.shape == (50, 2)
    assert np.all(np.isfinite(Y))


def test_separates_two_blobs():
    rng = np.random.default_rng(1)
    X = np.vstack(
        [
            rng.normal([-5, 0, 0], 0.4, size=(30, 3)),
            rng.normal([5, 0, 0], 0.4, size=(30, 3)),
        ]
    )
    Y = UMAP(
        n_components=2,
        n_neighbors=10,
        n_epochs=150,
        learning_rate=1.0,
        random_state=0,
    ).fit_transform(X)
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
        n_components=2,
        n_neighbors=8,
        n_epochs=60,
        random_state=42,
        init="random",
    )
    a = UMAP(**kwargs).fit_transform(X)
    b = UMAP(**kwargs).fit_transform(X)
    np.testing.assert_allclose(a, b)


def test_transform_training_returns_embedding():
    X = np.random.default_rng(0).normal(size=(30, 3))
    umap = UMAP(n_neighbors=5, n_epochs=40, random_state=0).fit(X)
    Y = umap.transform(X)
    np.testing.assert_allclose(Y, umap.embedding_)


def test_transform_oos_raises():
    X = np.random.default_rng(0).normal(size=(20, 2))
    umap = UMAP(n_neighbors=5, n_epochs=30, random_state=0).fit(X)
    with pytest.raises(NotImplementedError):
        umap.transform(X[:5])


def test_n_neighbors_too_small_raises():
    X = np.random.default_rng(0).normal(size=(10, 2))
    with pytest.raises(ValueError):
        UMAP(n_neighbors=1).fit_transform(X)
