import numpy as np
import pytest

from algo_stack.supervised.linear_regression import LinearRegression


def _make_data(seed=0, n=200, d=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.array([1.5, -2.0, 0.5])
    b = 4.0
    y = X @ w + b + 0.01 * rng.normal(size=n)
    return X, y, w, b


def test_recovers_known_coefficients():
    X, y, w, b = _make_data()
    model = LinearRegression().fit(X, y)
    np.testing.assert_allclose(model.coef_, w, atol=1e-2)
    assert abs(model.intercept_ - b) < 1e-2
    assert model.score(X, y) > 0.99


def test_normal_solver_matches_lstsq():
    X, y, _, _ = _make_data(seed=1)
    a = LinearRegression(solver="lstsq").fit(X, y)
    b = LinearRegression(solver="normal").fit(X, y)
    np.testing.assert_allclose(a.coef_, b.coef_, atol=1e-8)
    assert abs(a.intercept_ - b.intercept_) < 1e-8


def test_no_intercept():
    X, y, w, _ = _make_data()
    y_centered = y - y.mean()
    X_centered = X - X.mean(axis=0)
    model = LinearRegression(fit_intercept=False).fit(X_centered, y_centered)
    assert model.intercept_ == 0.0
    np.testing.assert_allclose(model.coef_, w, atol=5e-2)


def test_predict_validation():
    X, y, _, _ = _make_data()
    model = LinearRegression().fit(X, y)
    with pytest.raises(ValueError):
        model.predict(X[:, :2])  # wrong feature count


def test_unknown_solver_raises():
    X, y, _, _ = _make_data()
    with pytest.raises(ValueError):
        LinearRegression(solver="banana").fit(X, y)
