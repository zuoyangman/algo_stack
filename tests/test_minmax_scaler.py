import numpy as np
import pytest

from algo_stack.preprocessing.minmax_scaler import MinMaxScaler


def test_default_range_zero_one():
    X = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
    sc = MinMaxScaler().fit(X)
    Z = sc.transform(X)
    np.testing.assert_allclose(Z.min(axis=0), 0.0)
    np.testing.assert_allclose(Z.max(axis=0), 1.0)
    np.testing.assert_allclose(sc.data_min_, [0.0, 10.0])
    np.testing.assert_allclose(sc.data_max_, [10.0, 30.0])


def test_custom_feature_range():
    X = np.array([[0.0], [10.0]])
    sc = MinMaxScaler(feature_range=(-1, 1)).fit(X)
    Z = sc.fit_transform(X)
    np.testing.assert_allclose(Z.ravel(), [-1.0, 1.0])


def test_inverse_round_trip():
    rng = np.random.default_rng(0)
    X = rng.uniform(-5, 12, size=(60, 3))
    sc = MinMaxScaler(feature_range=(2, 5))
    Z = sc.fit_transform(X)
    np.testing.assert_allclose(sc.inverse_transform(Z), X, atol=1e-10)


def test_constant_feature():
    X = np.array([[3.0, 1.0], [3.0, 2.0], [3.0, 3.0]])
    Z = MinMaxScaler().fit_transform(X)
    np.testing.assert_allclose(Z[:, 0], 0.0)
    np.testing.assert_allclose(Z[:, 1], [0.0, 0.5, 1.0])


def test_invalid_feature_range():
    X = np.ones((5, 2))
    with pytest.raises(ValueError):
        MinMaxScaler(feature_range=(1, 0)).fit(X)


def test_attributes_shapes():
    X = np.random.default_rng(1).normal(size=(10, 4))
    sc = MinMaxScaler().fit(X)
    assert sc.min_.shape == (4,)
    assert sc.scale_.shape == (4,)
    assert sc.data_min_.shape == (4,)
    assert sc.data_max_.shape == (4,)


def test_keyword_only():
    with pytest.raises(TypeError):
        MinMaxScaler((0, 1))  # type: ignore[misc]
