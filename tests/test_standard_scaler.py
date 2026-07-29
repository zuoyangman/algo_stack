import numpy as np
import pytest

from algo_stack.preprocessing.standard_scaler import StandardScaler
from algo_stack.utils.preprocessing import StandardScaler as UtilsStandardScaler


def test_zero_mean_unit_variance():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=3.0, size=(100, 3))
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-8)
    np.testing.assert_allclose(Z.std(axis=0), 1.0, atol=1e-8)


def test_inverse_round_trip():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 4)) * 7 + 2
    sc = StandardScaler()
    Z = sc.fit_transform(X)
    np.testing.assert_allclose(sc.inverse_transform(Z), X, atol=1e-10)


def test_with_mean_false():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    sc = StandardScaler(with_mean=False).fit(X)
    np.testing.assert_allclose(sc.mean_, 0.0)
    Z = sc.transform(X)
    np.testing.assert_allclose(Z, X / sc.scale_)


def test_with_std_false():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    sc = StandardScaler(with_std=False).fit(X)
    np.testing.assert_allclose(sc.scale_, 1.0)
    Z = sc.transform(X)
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-12)


def test_constant_feature_safe():
    X = np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]])
    Z = StandardScaler().fit_transform(X)
    assert np.all(np.isfinite(Z))
    np.testing.assert_allclose(Z[:, 0], 0.0)


def test_matches_utils_scaler():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 3))
    a = StandardScaler().fit(X)
    b = UtilsStandardScaler().fit(X)
    np.testing.assert_allclose(a.mean_, b.mean_)
    np.testing.assert_allclose(a.scale_, b.scale_)
    np.testing.assert_allclose(a.transform(X), b.transform(X))


def test_not_fitted_raises():
    with pytest.raises(RuntimeError):
        StandardScaler().transform(np.ones((2, 2)))


def test_keyword_only():
    with pytest.raises(TypeError):
        StandardScaler(True)  # type: ignore[misc]
