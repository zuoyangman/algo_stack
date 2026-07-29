import numpy as np
import pytest

from algo_stack.supervised.kernel_ridge import KernelRidge


def test_rbf_recovers_sine():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2.5, 2.5, size=(120, 1))
    y = np.sin(X.ravel())
    reg = KernelRidge(alpha=0.05, kernel="rbf").fit(X, y)
    grid = np.linspace(-2, 2, 50).reshape(-1, 1)
    err = np.max(np.abs(reg.predict(grid) - np.sin(grid.ravel())))
    assert err < 0.2
    assert reg.score(X, y) > 0.95


def test_linear_matches_ridge_intuition():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    w = np.array([1.0, -2.0, 0.5])
    y = X @ w
    # Small alpha, linear kernel → near-perfect fit of a linear target
    reg = KernelRidge(alpha=1e-6, kernel="linear").fit(X, y)
    assert reg.score(X, y) > 0.999


def test_polynomial_kernel():
    rng = np.random.default_rng(2)
    X = rng.uniform(-1.5, 1.5, size=(80, 1))
    y = X.ravel() ** 2
    reg = KernelRidge(alpha=0.01, kernel="polynomial", degree=2).fit(X, y)
    assert reg.score(X, y) > 0.9


def test_attributes():
    X = np.linspace(-1, 1, 30).reshape(-1, 1)
    y = np.sin(X.ravel())
    reg = KernelRidge(alpha=0.1, kernel="rbf", gamma="scale").fit(X, y)
    assert reg.dual_coef_.shape == (30,)
    assert reg.X_fit_.shape == (30, 1)
    assert isinstance(reg.intercept_, float)
    assert reg.gamma_ > 0


def test_no_intercept():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    reg = KernelRidge(alpha=1e-3, kernel="linear", fit_intercept=False).fit(X, y)
    assert reg.intercept_ == 0.0
    pred = reg.predict(X)
    assert np.allclose(pred, y, atol=0.05)


def test_invalid_params():
    X = np.zeros((5, 2))
    y = np.arange(5, dtype=float)
    with pytest.raises(ValueError):
        KernelRidge(kernel="bogus").fit(X, y)
    with pytest.raises(ValueError):
        KernelRidge(alpha=-1.0).fit(X, y)
    with pytest.raises(ValueError):
        KernelRidge(degree=0).fit(X, y)
