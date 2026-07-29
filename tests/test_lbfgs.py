import numpy as np
import pytest

from algo_stack.optimization.lbfgs import LBFGS, minimize_lbfgs, wolfe_line_search
from algo_stack.supervised.softmax_classifier import SoftmaxClassifier


def test_quadratic_bowl():
    def fun(z):
        return float(np.sum((z - 1.0) ** 2))

    def jac(z):
        return 2.0 * (z - 1.0)

    res = minimize_lbfgs(fun, np.zeros(4), jac=jac, tol=1e-8)
    assert res.success
    np.testing.assert_allclose(res.x, np.ones(4), atol=1e-5)
    assert res.fun < 1e-10


def test_rosenbrock():
    def fun(z):
        x, y = z
        return float((1 - x) ** 2 + 100 * (y - x**2) ** 2)

    def jac(z):
        x, y = z
        return np.array(
            [-2 * (1 - x) - 400 * x * (y - x**2), 200 * (y - x**2)],
            dtype=np.float64,
        )

    res = minimize_lbfgs(fun, np.array([-1.2, 1.0]), jac=jac, max_iter=300, tol=1e-8)
    assert res.fun < 1e-8
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-4)


def test_wolfe_line_search_descent():
    def fun(z):
        return float(np.sum(z**2))

    def jac(z):
        return 2 * z

    x = np.array([3.0, -4.0])
    g = jac(x)
    p = -g
    alpha, f_new, g_new, _, _ = wolfe_line_search(fun, jac, x, p, f_x=fun(x), g_x=g)
    assert alpha > 0
    assert f_new < fun(x)


def test_lbfgs_class_wrapper():
    opt = LBFGS(tol=1e-8)
    res = opt.minimize(
        lambda z: float(np.sum(z**2)),
        np.ones(3),
        jac=lambda z: 2 * z,
    )
    assert opt.result_ is not None
    np.testing.assert_allclose(res.x, 0.0, atol=1e-6)


def test_softmax_lbfgs_solver():
    rng = np.random.default_rng(0)
    centres = np.array([[-3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    X = np.vstack([c + rng.normal(scale=0.4, size=(80, 2)) for c in centres])
    y = np.repeat(np.arange(3), 80)
    clf = SoftmaxClassifier(optimizer="lbfgs", n_epochs=100, tol=1e-6, random_state=0).fit(
        X, y
    )
    assert clf.score(X, y) > 0.95
    assert len(clf.loss_curve_) >= 1
    assert clf.loss_curve_[-1] < clf.loss_curve_[0]
