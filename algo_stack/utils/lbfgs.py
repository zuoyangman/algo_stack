"""L-BFGS with strong Wolfe line search (NumPy reference).

Unlike the mini-batch ``Optimizer`` protocol in ``algo_stack.utils.optim``,
L-BFGS is a **full-batch** quasi-Newton method that minimises a scalar
objective ``f(θ)`` given its gradient ``∇f(θ)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class LBFGSResult:
    """Outcome of an L-BFGS run."""

    x: np.ndarray
    fun: float
    grad: np.ndarray
    n_iter: int
    n_fev: int
    n_gev: int
    success: bool
    message: str
    loss_curve: list[float]


def wolfe_line_search(
    fun: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    p: np.ndarray,
    *,
    f_x: float | None = None,
    g_x: np.ndarray | None = None,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iter: int = 20,
) -> tuple[float, float, np.ndarray, int, int]:
    """Strong Wolfe line search along direction ``p``.

    Returns
    -------
    alpha, f_new, g_new, n_fev, n_gev
    """

    if f_x is None:
        f_x = float(fun(x))
    if g_x is None:
        g_x = grad(x)
    g_x = np.asarray(g_x, dtype=np.float64)
    dphi0 = float(g_x @ p)
    if dphi0 >= 0:
        # Not a descent direction — fall back to tiny step.
        return 1e-8, f_x, g_x, 0, 0

    alpha = alpha0
    alpha_prev = 0.0
    f_prev = f_x
    n_fev = 0
    n_gev = 0

    for i in range(max_iter):
        x_new = x + alpha * p
        f_new = float(fun(x_new))
        n_fev += 1
        # Armijo sufficient decrease
        if f_new > f_x + c1 * alpha * dphi0 or (i > 0 and f_new >= f_prev):
            alpha, f_new, g_new, fev, gev = _zoom(
                fun, grad, x, p, alpha_prev, alpha, f_x, dphi0, f_prev, c1, c2
            )
            return alpha, f_new, g_new, n_fev + fev, n_gev + gev

        g_new = np.asarray(grad(x_new), dtype=np.float64)
        n_gev += 1
        dphi = float(g_new @ p)
        # Strong curvature condition
        if abs(dphi) <= -c2 * dphi0:
            return alpha, f_new, g_new, n_fev, n_gev
        if dphi >= 0:
            alpha, f_new, g_new, fev, gev = _zoom(
                fun, grad, x, p, alpha, alpha_prev, f_x, dphi0, f_new, c1, c2
            )
            return alpha, f_new, g_new, n_fev + fev, n_gev + gev

        alpha_prev = alpha
        f_prev = f_new
        alpha *= 2.0

    return alpha, f_new, g_new, n_fev, n_gev


def _zoom(
    fun: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    p: np.ndarray,
    alpha_lo: float,
    alpha_hi: float,
    f_x: float,
    dphi0: float,
    f_lo: float,
    c1: float,
    c2: float,
    max_iter: int = 20,
) -> tuple[float, float, np.ndarray, int, int]:
    n_fev = 0
    n_gev = 0
    f_new = f_lo
    g_new = grad(x + alpha_lo * p)
    n_gev += 1
    alpha = alpha_lo

    for _ in range(max_iter):
        alpha = 0.5 * (alpha_lo + alpha_hi)
        x_new = x + alpha * p
        f_new = float(fun(x_new))
        n_fev += 1
        if f_new > f_x + c1 * alpha * dphi0 or f_new >= f_lo:
            alpha_hi = alpha
        else:
            g_new = np.asarray(grad(x_new), dtype=np.float64)
            n_gev += 1
            dphi = float(g_new @ p)
            if abs(dphi) <= -c2 * dphi0:
                return alpha, f_new, g_new, n_fev, n_gev
            if dphi * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha
            f_lo = f_new

    return alpha, f_new, g_new, n_fev, n_gev


def _two_loop_recursion(
    grad: np.ndarray,
    s_list: list[np.ndarray],
    y_list: list[np.ndarray],
    rho_list: list[float],
) -> np.ndarray:
    """L-BFGS two-loop recursion → search direction ``H_k ∇f`` (ascent form negated)."""

    q = grad.copy()
    m = len(s_list)
    alphas = [0.0] * m
    for i in range(m - 1, -1, -1):
        alphas[i] = rho_list[i] * float(s_list[i] @ q)
        q = q - alphas[i] * y_list[i]

    if m > 0:
        ys = float(y_list[-1] @ s_list[-1])
        yy = float(y_list[-1] @ y_list[-1])
        gamma = ys / yy if yy > 0 else 1.0
    else:
        gamma = 1.0
    r = gamma * q

    for i in range(m):
        beta = rho_list[i] * float(y_list[i] @ r)
        r = r + s_list[i] * (alphas[i] - beta)
    return r


def minimize_lbfgs(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    jac: Callable[[np.ndarray], np.ndarray],
    max_iter: int = 200,
    m: int = 10,
    tol: float = 1e-6,
    c1: float = 1e-4,
    c2: float = 0.9,
) -> LBFGSResult:
    """Minimise ``fun`` starting from ``x0`` with L-BFGS + strong Wolfe search.

    Parameters
    ----------
    fun : callable
        Objective ``f(x) -> float``.
    x0 : ndarray
        Initial parameter vector (1-D).
    jac : callable
        Gradient ``∇f(x) -> ndarray`` with the same shape as ``x0``.
    max_iter : int
    m : int
        L-BFGS memory (number of ``(s, y)`` pairs retained).
    tol : float
        Stop when ``||∇f||_∞ < tol``.
    c1, c2 : float
        Wolfe constants.
    """

    x = np.asarray(x0, dtype=np.float64).copy().ravel()
    f = float(fun(x))
    g = np.asarray(jac(x), dtype=np.float64).ravel()
    n_fev = 1
    n_gev = 1
    loss_curve = [f]

    s_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    rho_list: list[float] = []

    for it in range(max_iter):
        if float(np.max(np.abs(g))) < tol:
            return LBFGSResult(
                x=x,
                fun=f,
                grad=g,
                n_iter=it,
                n_fev=n_fev,
                n_gev=n_gev,
                success=True,
                message="Gradient norm below tolerance.",
                loss_curve=loss_curve,
            )

        # Search direction: p = −H ∇f
        Hg = _two_loop_recursion(g, s_list, y_list, rho_list)
        p = -Hg

        alpha, f_new, g_new, fev, gev = wolfe_line_search(
            fun, jac, x, p, f_x=f, g_x=g, alpha0=1.0, c1=c1, c2=c2
        )
        n_fev += fev
        n_gev += gev
        g_new = np.asarray(g_new, dtype=np.float64).ravel()

        s = alpha * p
        y = g_new - g
        ys = float(y @ s)
        if ys > 1e-10:
            if len(s_list) >= m:
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)
            s_list.append(s)
            y_list.append(y)
            rho_list.append(1.0 / ys)

        x = x + s
        f = f_new
        g = g_new
        loss_curve.append(f)

    return LBFGSResult(
        x=x,
        fun=f,
        grad=g,
        n_iter=max_iter,
        n_fev=n_fev,
        n_gev=n_gev,
        success=float(np.max(np.abs(g))) < tol,
        message="Maximum iterations reached.",
        loss_curve=loss_curve,
    )


class LBFGS:
    """Sklearn-style wrapper around :func:`minimize_lbfgs`.

    Parameters
    ----------
    max_iter, m, tol, c1, c2
        Forwarded to :func:`minimize_lbfgs`.
    """

    def __init__(
        self,
        *,
        max_iter: int = 200,
        m: int = 10,
        tol: float = 1e-6,
        c1: float = 1e-4,
        c2: float = 0.9,
    ) -> None:
        self.max_iter = max_iter
        self.m = m
        self.tol = tol
        self.c1 = c1
        self.c2 = c2
        self.result_: LBFGSResult | None = None

    def minimize(
        self,
        fun: Callable[[np.ndarray], float],
        x0: np.ndarray,
        jac: Callable[[np.ndarray], np.ndarray],
    ) -> LBFGSResult:
        self.result_ = minimize_lbfgs(
            fun,
            x0,
            jac=jac,
            max_iter=self.max_iter,
            m=self.m,
            tol=self.tol,
            c1=self.c1,
            c2=self.c2,
        )
        return self.result_
