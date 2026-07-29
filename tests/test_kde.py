import numpy as np
import pytest

from algo_stack.unsupervised.kde import KernelDensity


def test_score_samples_shape_and_finite():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    kde = KernelDensity(bandwidth=0.5).fit(X)
    ll = kde.score_samples(X)
    assert ll.shape == (100,)
    assert np.all(np.isfinite(ll))


def test_higher_density_near_data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 1))
    kde = KernelDensity(bandwidth=0.3).fit(X)
    centre = np.array([[0.0]])
    far = np.array([[6.0]])
    assert kde.score_samples(centre)[0] > kde.score_samples(far)[0]


def test_score_is_mean_loglik():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 2))
    kde = KernelDensity(bandwidth=1.0).fit(X)
    assert kde.score(X) == pytest.approx(float(np.mean(kde.score_samples(X))))


def test_standard_normal_1d_roughly_correct():
    """At 0, N(0,1) log-density is -0.5 log(2π) ≈ -0.918; KDE should be close."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(2000, 1))
    kde = KernelDensity(bandwidth=0.25).fit(X)
    log_p0 = kde.score_samples(np.array([[0.0]]))[0]
    true = -0.5 * np.log(2 * np.pi)
    assert abs(log_p0 - true) < 0.15


def test_invalid_bandwidth():
    with pytest.raises(ValueError):
        KernelDensity(bandwidth=0).fit(np.zeros((5, 1)))


def test_unsupported_kernel():
    with pytest.raises(ValueError):
        KernelDensity(kernel="tophat").fit(np.zeros((5, 1)))
