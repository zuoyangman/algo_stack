import numpy as np
import pytest

from algo_stack.utils import (
    StandardScaler,
    accuracy_score,
    check_array,
    check_random_state,
    check_X_y,
    mean_squared_error,
    r2_score,
    train_test_split,
)


def test_check_array_rejects_1d_by_default():
    with pytest.raises(ValueError):
        check_array(np.array([1.0, 2.0, 3.0]))


def test_check_array_rejects_nan():
    with pytest.raises(ValueError):
        check_array(np.array([[1.0, np.nan]]))


def test_check_X_y_sample_mismatch():
    with pytest.raises(ValueError):
        check_X_y(np.zeros((5, 2)), np.zeros(4))


def test_check_random_state_types():
    assert isinstance(check_random_state(None), np.random.Generator)
    a = check_random_state(0)
    b = check_random_state(0)
    assert a.integers(0, 1_000_000) == b.integers(0, 1_000_000)
    g = np.random.default_rng(1)
    assert check_random_state(g) is g
    with pytest.raises(TypeError):
        check_random_state("not-a-seed")


def test_train_test_split_shapes_and_no_overlap():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    assert Xtr.shape == (7, 2) and Xte.shape == (3, 2)
    assert ytr.shape == (7,) and yte.shape == (3,)
    assert set(ytr.tolist()).isdisjoint(set(yte.tolist()))
    assert set(ytr.tolist()) | set(yte.tolist()) == set(range(10))


def test_standard_scaler_round_trip():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=3.0, size=(100, 3))
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-8)
    np.testing.assert_allclose(Z.std(axis=0), 1.0, atol=1e-8)
    np.testing.assert_allclose(sc.inverse_transform(Z), X, atol=1e-8)


def test_metrics_basic():
    y = np.array([1.0, 2.0, 3.0])
    assert mean_squared_error(y, y) == 0.0
    assert r2_score(y, y) == 1.0
    assert accuracy_score(np.array([0, 1, 1]), np.array([0, 0, 1])) == pytest.approx(2 / 3)
