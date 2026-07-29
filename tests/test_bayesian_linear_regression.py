import numpy as np
import pytest

from algo_stack.supervised.bayesian_linear_regression import BayesianLinearRegression


def _make_data(seed=0, n=150, d=3, noise=0.2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.array([1.5, -2.0, 0.5])
    b = 2.0
    y = X @ w + b + noise * rng.normal(size=n)
    return X, y, w, b


def test_recovers_coefficients():
    X, y, w, b = _make_data()
    model = BayesianLinearRegression(alpha=1e-3, beta=25.0).fit(X, y)
    np.testing.assert_allclose(model.coef_, w, atol=0.15)
    assert abs(model.intercept_ - b) < 0.2
    assert model.score(X, y) > 0.95


def test_stores_precision_attrs():
    X, y, _, _ = _make_data()
    model = BayesianLinearRegression(alpha=2.0, beta=10.0).fit(X, y)
    assert model.alpha_ == 2.0
    assert model.beta_ == 10.0
    assert model.sigma_.shape == (3, 3)


def test_predict_std_positive():
    X, y, _, _ = _make_data()
    model = BayesianLinearRegression(alpha=1.0, beta=4.0).fit(X, y)
    mean = model.predict(X)
    std = model.predict_std(X)
    assert mean.shape == (X.shape[0],)
    assert std.shape == (X.shape[0],)
    assert np.all(std > 0)
    var = model.predict_var(X)
    np.testing.assert_allclose(var, std**2, atol=1e-12)


def test_larger_alpha_shrinks():
    X, y, _, _ = _make_data()
    small = BayesianLinearRegression(alpha=0.01, beta=10.0).fit(X, y)
    large = BayesianLinearRegression(alpha=100.0, beta=10.0).fit(X, y)
    assert np.linalg.norm(large.coef_) < np.linalg.norm(small.coef_)


def test_no_intercept():
    X, y, w, _ = _make_data()
    X_c = X - X.mean(axis=0)
    y_c = y - y.mean()
    model = BayesianLinearRegression(
        alpha=1e-3, beta=25.0, fit_intercept=False
    ).fit(X_c, y_c)
    assert model.intercept_ == 0.0
    np.testing.assert_allclose(model.coef_, w, atol=0.2)


def test_invalid_precision_raises():
    X, y, _, _ = _make_data()
    with pytest.raises(ValueError):
        BayesianLinearRegression(alpha=0.0).fit(X, y)
    with pytest.raises(ValueError):
        BayesianLinearRegression(beta=-1.0).fit(X, y)


def test_predict_validation():
    X, y, _, _ = _make_data()
    model = BayesianLinearRegression().fit(X, y)
    with pytest.raises(ValueError):
        model.predict(X[:, :2])
    with pytest.raises(ValueError):
        model.predict_std(X[:, :1])
