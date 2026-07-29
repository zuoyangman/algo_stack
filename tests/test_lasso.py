import numpy as np
import pytest

from algo_stack.supervised.lasso import Lasso


def _sparse_data(seed=0, n=300, d=15):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.zeros(d)
    w[:3] = [3.0, -2.0, 1.5]
    y = X @ w + 0.1 * rng.normal(size=n)
    return X, y, w


def test_recovers_sparse_support():
    X, y, w = _sparse_data()
    model = Lasso(alpha=0.05, max_iter=3000, tol=1e-5).fit(X, y)
    # Non-zeros of truth should be recovered; most others near zero.
    assert np.all(np.abs(model.coef_[:3]) > 0.5)
    assert np.sum(np.abs(model.coef_[3:]) > 0.2) <= 2
    assert model.score(X, y) > 0.9


def test_larger_alpha_sparser():
    X, y, _ = _sparse_data()
    dense = Lasso(alpha=0.001, max_iter=2000).fit(X, y)
    sparse = Lasso(alpha=0.5, max_iter=2000).fit(X, y)
    assert np.sum(np.abs(sparse.coef_) > 1e-6) <= np.sum(np.abs(dense.coef_) > 1e-6)


def test_no_intercept():
    X, y, _ = _sparse_data()
    X_c = X - X.mean(axis=0)
    y_c = y - y.mean()
    model = Lasso(alpha=0.05, fit_intercept=False, max_iter=2000).fit(X_c, y_c)
    assert model.intercept_ == 0.0


def test_n_iter_recorded():
    X, y, _ = _sparse_data()
    model = Lasso(alpha=0.1, max_iter=50, tol=0.0).fit(X, y)
    assert model.n_iter_ == 50


def test_predict_validation():
    X, y, _ = _sparse_data()
    model = Lasso(alpha=0.1).fit(X, y)
    with pytest.raises(ValueError):
        model.predict(X[:, :2])


def test_negative_alpha_raises():
    X, y, _ = _sparse_data()
    with pytest.raises(ValueError):
        Lasso(alpha=-0.1).fit(X, y)
