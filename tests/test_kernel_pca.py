import numpy as np
import pytest

from algo_stack.unsupervised.kernel_pca import KernelPCA
from algo_stack.unsupervised.pca import PCA


def test_linear_kernel_matches_pca_up_to_sign():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    Z_pca = PCA(n_components=2).fit_transform(X)
    Z_k = KernelPCA(n_components=2, kernel="linear").fit_transform(X)
    # Align signs per component.
    for j in range(2):
        if np.dot(Z_pca[:, j], Z_k[:, j]) < 0:
            Z_k[:, j] *= -1
    # Same subspace up to scaling differences from eigenvalue convention —
    # correlations should be high.
    for j in range(2):
        corr = np.corrcoef(Z_pca[:, j], Z_k[:, j])[0, 1]
        assert abs(corr) > 0.99


def test_rbf_fit_transform_shape():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 2))
    kpca = KernelPCA(n_components=3, kernel="rbf", gamma=0.5).fit(X)
    Z = kpca.transform(X)
    assert Z.shape == (50, 3)
    assert kpca.lambdas_.shape == (3,)


def test_poly_kernel_runs():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 2))
    Z = KernelPCA(n_components=2, kernel="poly", degree=2, gamma=1.0).fit_transform(X)
    assert Z.shape == (30, 2)
    assert np.all(np.isfinite(Z))


def test_transform_new_points_finite():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 2))
    X_new = rng.normal(size=(10, 2))
    kpca = KernelPCA(n_components=2, kernel="rbf", gamma=1.0).fit(X)
    Z = kpca.transform(X_new)
    assert Z.shape == (10, 2)
    assert np.all(np.isfinite(Z))


def test_invalid_n_components():
    X = np.random.default_rng(0).normal(size=(5, 2))
    with pytest.raises(ValueError):
        KernelPCA(n_components=10).fit(X)
