import numpy as np
import pytest

from algo_stack.unsupervised.gmm import GaussianMixture


def _two_gaussians(seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack(
        [
            rng.normal([-3.0, 0.0], 0.4, size=(80, 2)),
            rng.normal([3.0, 0.0], 0.4, size=(80, 2)),
        ]
    )
    y = np.repeat([0, 1], 80)
    return X, y


def test_recovers_two_components_diag():
    X, y = _two_gaussians()
    gmm = GaussianMixture(
        n_components=2, covariance_type="diag", random_state=0, max_iter=100
    ).fit(X)
    assert gmm.means_.shape == (2, 2)
    assert gmm.weights_.shape == (2,)
    assert np.isclose(gmm.weights_.sum(), 1.0)
    # Means should be near ±3 on first axis (order-invariant).
    xs = np.sort(gmm.means_[:, 0])
    assert xs[0] < -1.5 and xs[1] > 1.5


def test_full_covariance_runs():
    X, _ = _two_gaussians()
    gmm = GaussianMixture(
        n_components=2, covariance_type="full", random_state=1, max_iter=50
    ).fit(X)
    assert gmm.covariances_.shape == (2, 2, 2)
    assert gmm.predict(X).shape == (X.shape[0],)


def test_predict_proba_sums_to_one():
    X, _ = _two_gaussians()
    gmm = GaussianMixture(n_components=2, random_state=0).fit(X)
    proba = gmm.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(gmm.predict(X), np.argmax(proba, axis=1))


def test_score_samples_finite():
    X, _ = _two_gaussians()
    gmm = GaussianMixture(n_components=2, random_state=0).fit(X)
    ll = gmm.score_samples(X)
    assert ll.shape == (X.shape[0],)
    assert np.all(np.isfinite(ll))
    assert gmm.score(X) == pytest.approx(float(np.mean(ll)))


def test_reproducibility():
    X, _ = _two_gaussians()
    a = GaussianMixture(n_components=2, random_state=42).fit(X)
    b = GaussianMixture(n_components=2, random_state=42).fit(X)
    np.testing.assert_allclose(a.means_, b.means_)
    np.testing.assert_allclose(a.weights_, b.weights_)


def test_too_many_components_raises():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError):
        GaussianMixture(n_components=5).fit(X)
