import numpy as np
import pytest

from algo_stack.supervised.ridge import Ridge


def _make_data(seed=0, n=200, d=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.array([1.5, -2.0, 0.5])
    b = 4.0
    y = X @ w + b + 0.01 * rng.normal(size=n)
    return X, y, w, b


def test_recovers_known_coefficients_small_alpha():
    X, y, w, b = _make_data()
    model = Ridge(alpha=1e-6).fit(X, y)
    np.testing.assert_allclose(model.coef_, w, atol=1e-2)
    assert abs(model.intercept_ - b) < 1e-2
    assert model.score(X, y) > 0.99


def test_larger_alpha_shrinks_weights():
    X, y, _, _ = _make_data()
    small = Ridge(alpha=0.01).fit(X, y)
    large = Ridge(alpha=100.0).fit(X, y)
    assert np.linalg.norm(large.coef_) < np.linalg.norm(small.coef_)


def test_no_intercept():
    X, y, w, _ = _make_data()
    y_c = y - y.mean()
    X_c = X - X.mean(axis=0)
    model = Ridge(alpha=1e-6, fit_intercept=False).fit(X_c, y_c)
    assert model.intercept_ == 0.0
    np.testing.assert_allclose(model.coef_, w, atol=5e-2)


def test_predict_validation():
    X, y, _, _ = _make_data()
    model = Ridge().fit(X, y)
    with pytest.raises(ValueError):
        model.predict(X[:, :2])


def test_negative_alpha_raises():
    X, y, _, _ = _make_data()
    with pytest.raises(ValueError):
        Ridge(alpha=-1.0).fit(X, y)


def test_alpha_zero_matches_ols_direction():
    X, y, w, _ = _make_data(seed=2)
    model = Ridge(alpha=0.0).fit(X, y)
    np.testing.assert_allclose(model.coef_, w, atol=2e-2)
