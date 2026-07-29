import numpy as np
import pytest

from algo_stack.preprocessing.polynomial_features import PolynomialFeatures


def test_degree_two_with_bias():
    X = np.array([[2.0, 3.0]])
    poly = PolynomialFeatures(degree=2, include_bias=True).fit(X)
    Z = poly.transform(X)
    # [1, a, b, a^2, a*b, b^2]
    np.testing.assert_allclose(Z, [[1.0, 2.0, 3.0, 4.0, 6.0, 9.0]])
    assert poly.n_features_in_ == 2
    assert poly.n_output_features_ == 6
    assert poly.powers_.shape == (6, 2)


def test_no_bias():
    X = np.array([[2.0, 3.0]])
    Z = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
    np.testing.assert_allclose(Z, [[2.0, 3.0, 4.0, 6.0, 9.0]])


def test_interaction_only():
    X = np.array([[2.0, 3.0]])
    Z = PolynomialFeatures(degree=2, interaction_only=True).fit_transform(X)
    # [1, a, b, a*b]
    np.testing.assert_allclose(Z, [[1.0, 2.0, 3.0, 6.0]])


def test_degree_one_is_identity_plus_bias():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    Z = PolynomialFeatures(degree=1, include_bias=True).fit_transform(X)
    np.testing.assert_allclose(Z[:, 0], 1.0)
    np.testing.assert_allclose(Z[:, 1:], X)


def test_multiple_rows():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    Z = PolynomialFeatures(degree=2).fit_transform(X)
    assert Z.shape == (2, 6)
    np.testing.assert_allclose(Z[0], [1, 1, 2, 1, 2, 4])
    np.testing.assert_allclose(Z[1], [1, 3, 4, 9, 12, 16])


def test_invalid_degree():
    with pytest.raises(ValueError):
        PolynomialFeatures(degree=-1).fit(np.ones((3, 2)))


def test_feature_mismatch():
    poly = PolynomialFeatures().fit(np.ones((5, 2)))
    with pytest.raises(ValueError):
        poly.transform(np.ones((5, 3)))


def test_keyword_only():
    with pytest.raises(TypeError):
        PolynomialFeatures(2)  # type: ignore[misc]
