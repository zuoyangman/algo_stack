import numpy as np
import pytest

from algo_stack.preprocessing.one_hot_encoder import OneHotEncoder


def test_1d_string_categories():
    X = np.array(["red", "blue", "red", "green"])
    enc = OneHotEncoder().fit(X)
    assert len(enc.categories_) == 1
    assert enc.categories_[0].tolist() == ["red", "blue", "green"]
    Z = enc.transform(X)
    assert Z.shape == (4, 3)
    assert Z.dtype == np.float64
    np.testing.assert_allclose(Z[0], [1, 0, 0])
    np.testing.assert_allclose(Z[1], [0, 1, 0])
    np.testing.assert_allclose(Z[3], [0, 0, 1])


def test_2d_numeric_categories():
    X = np.array([[0, 1], [1, 0], [0, 0]])
    Z = OneHotEncoder().fit_transform(X)
    # feature0 cats: 0,1 → 2 cols; feature1 cats: 1,0 → 2 cols
    assert Z.shape == (3, 4)
    assert np.all((Z == 0) | (Z == 1))


def test_drop_first():
    X = np.array([["a"], ["b"], ["a"], ["c"]])
    enc = OneHotEncoder(drop="first").fit(X)
    assert enc.categories_[0].tolist() == ["b", "c"]
    Z = enc.transform(X)
    assert Z.shape == (4, 2)
    # "a" was dropped → all-zero row
    np.testing.assert_allclose(Z[0], [0, 0])
    np.testing.assert_allclose(Z[1], [1, 0])


def test_unknown_category_all_zero():
    X_fit = np.array([["a"], ["b"]])
    enc = OneHotEncoder().fit(X_fit)
    Z = enc.transform(np.array([["c"], ["a"]]))
    np.testing.assert_allclose(Z[0], [0, 0])
    np.testing.assert_allclose(Z[1], [1, 0])


def test_fit_transform_matches_transform():
    X = np.array([[1], [2], [1], [3]])
    enc = OneHotEncoder()
    Z1 = enc.fit_transform(X)
    Z2 = enc.transform(X)
    np.testing.assert_allclose(Z1, Z2)


def test_invalid_drop():
    with pytest.raises(ValueError):
        OneHotEncoder(drop="last").fit(np.array([["a"]]))


def test_feature_mismatch():
    enc = OneHotEncoder().fit(np.array([["a"], ["b"]]))
    with pytest.raises(ValueError):
        enc.transform(np.array([["a", "b"]]))


def test_keyword_only():
    with pytest.raises(TypeError):
        OneHotEncoder("first")  # type: ignore[misc]
