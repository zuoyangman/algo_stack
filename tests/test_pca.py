import numpy as np
import pytest

from algo_stack.unsupervised.pca import PCA


def test_explained_variance_ratio_sums_le_one():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    pca = PCA(n_components=3).fit(X)
    assert pca.components_.shape == (3, 4)
    assert pca.explained_variance_.shape == (3,)
    s = pca.explained_variance_ratio_.sum()
    assert 0 < s <= 1.0 + 1e-10


def test_reconstruction_full_rank():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3))
    pca = PCA(n_components=3).fit(X)
    X_hat = pca.inverse_transform(pca.transform(X))
    np.testing.assert_allclose(X_hat, X, atol=1e-10)


def test_fit_transform_matches_transform():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 5))
    pca = PCA(n_components=2)
    Z1 = pca.fit_transform(X)
    Z2 = pca.transform(X)
    np.testing.assert_allclose(Z1, Z2)


def test_mean_is_feature_mean():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 4)) + np.array([10.0, -2.0, 3.0, 0.5])
    pca = PCA(n_components=2).fit(X)
    np.testing.assert_allclose(pca.mean_, X.mean(axis=0))


def test_correlated_data_first_component_dominates():
    rng = np.random.default_rng(4)
    x = rng.normal(size=200)
    X = np.column_stack([x, 2 * x + 0.01 * rng.normal(size=200)])
    pca = PCA(n_components=2).fit(X)
    assert pca.explained_variance_ratio_[0] > 0.95


def test_invalid_n_components():
    X = np.random.default_rng(0).normal(size=(10, 3))
    with pytest.raises(ValueError):
        PCA(n_components=5).fit(X)
