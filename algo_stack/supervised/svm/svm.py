"""Support Vector Machines: soft-margin C-SVC (SMO) and Kernel-Ridge-style SVR.

Binary SVC is trained with a didactic Sequential Minimal Optimisation (SMO)
solver. Multiclass uses One-vs-Rest. SVR uses the closed-form dual of Kernel
Ridge Regression (documented in PRINCIPLE.md) rather than ε-SVR SMO.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from algo_stack._base import BaseEstimator, ClassifierMixin, RegressorMixin
from algo_stack.utils.kernels import kernel_matrix, resolve_gamma
from algo_stack.utils.validation import check_array, check_X_y, check_random_state


# ---------------------------------------------------------------------------
# Binary C-SVM via simplified SMO
# ---------------------------------------------------------------------------


def _compute_b(
    alpha: np.ndarray,
    y: np.ndarray,
    K: np.ndarray,
    C: float,
    tol: float,
) -> float:
    """Average bias over unbound support vectors (0 < α < C)."""

    f = (alpha * y) @ K  # f_i without bias, shape (n,)
    mask = (alpha > tol) & (alpha < C - tol)
    if np.any(mask):
        # For unbound SVs: y_i f(x_i) = 1  ⇒  b = y_i - f_i
        return float(np.mean(y[mask] - f[mask]))
    # Fall back: average over all support vectors
    sv = alpha > tol
    if np.any(sv):
        return float(np.mean(y[sv] - f[sv]))
    return 0.0


def _smo_binary(
    K: np.ndarray,
    y: np.ndarray,
    *,
    C: float,
    tol: float,
    max_iter: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, int]:
    """Train a binary soft-margin SVM with simplified SMO.

    Parameters
    ----------
    K : (n, n) kernel matrix on the training set
    y : (n,) labels in {-1, +1}

    Returns
    -------
    alpha : (n,) dual coefficients
    b : intercept
    n_iter : number of outer passes performed
    """

    n = y.shape[0]
    alpha = np.zeros(n, dtype=np.float64)
    # Cache of f_i = Σ_j α_j y_j K_ij  (decision without bias)
    f = np.zeros(n, dtype=np.float64)
    b = 0.0

    def decision(i: int) -> float:
        return f[i] + b

    def take_step(i1: int, i2: int) -> bool:
        nonlocal b, f
        if i1 == i2:
            return False

        y1, y2 = y[i1], y[i2]
        a1_old, a2_old = alpha[i1], alpha[i2]
        E1 = decision(i1) - y1
        E2 = decision(i2) - y2

        s = y1 * y2
        if s < 0:
            L = max(0.0, a2_old - a1_old)
            H = min(C, C + a2_old - a1_old)
        else:
            L = max(0.0, a1_old + a2_old - C)
            H = min(C, a1_old + a2_old)
        if L >= H:
            return False

        k11 = K[i1, i1]
        k22 = K[i2, i2]
        k12 = K[i1, i2]
        eta = k11 + k22 - 2.0 * k12
        if eta > 1e-12:
            a2_new = a2_old + y2 * (E1 - E2) / eta
            a2_new = float(np.clip(a2_new, L, H))
        else:
            # Degenerate: evaluate objective at bounds and pick the better
            # (simplified: skip this pair)
            return False

        if abs(a2_new - a2_old) < tol * (a2_new + a2_old + tol):
            return False

        a1_new = a1_old + s * (a2_old - a2_new)
        # Clip a1 into [0, C] for numerical safety
        a1_new = float(np.clip(a1_new, 0.0, C))

        # Update cached f and alphas
        da1 = a1_new - a1_old
        da2 = a2_new - a2_old
        f += da1 * y1 * K[:, i1] + da2 * y2 * K[:, i2]
        alpha[i1] = a1_new
        alpha[i2] = a2_new

        # Recompute bias from unbound SVs
        b = _compute_b(alpha, y, K, C, tol)
        return True

    def examine_example(i2: int) -> int:
        y2 = y[i2]
        a2 = alpha[i2]
        E2 = decision(i2) - y2
        r2 = E2 * y2

        # KKT violation?
        if (r2 < -tol and a2 < C) or (r2 > tol and a2 > 0):
            # Prefer second alpha among non-bound examples
            nonbound = np.where((alpha > tol) & (alpha < C - tol))[0]
            if len(nonbound) > 1:
                # Heuristic: maximise |E1 - E2|
                E = f + b - y
                i1 = int(nonbound[np.argmax(np.abs(E[nonbound] - E2))])
                if take_step(i1, i2):
                    return 1

            # Loop over all non-bound, shuffled
            order = nonbound.copy()
            rng.shuffle(order)
            for i1 in order:
                if take_step(int(i1), i2):
                    return 1

            # Loop over all examples, shuffled
            all_idx = np.arange(n)
            rng.shuffle(all_idx)
            for i1 in all_idx:
                if take_step(int(i1), i2):
                    return 1
        return 0

    num_changed = 0
    examine_all = True
    n_iter = 0
    while (num_changed > 0 or examine_all) and n_iter < max_iter:
        num_changed = 0
        if examine_all:
            for i in range(n):
                num_changed += examine_example(i)
        else:
            nonbound = np.where((alpha > tol) & (alpha < C - tol))[0]
            for i in nonbound:
                num_changed += examine_example(int(i))
        if examine_all:
            examine_all = False
        elif num_changed == 0:
            examine_all = True
        n_iter += 1
        b = _compute_b(alpha, y, K, C, tol)

    return alpha, b, n_iter


# ---------------------------------------------------------------------------
# SVC
# ---------------------------------------------------------------------------


class SVC(BaseEstimator, ClassifierMixin):
    """Soft-margin C-Support Vector Classifier (binary SMO, multiclass OvR).

    Parameters
    ----------
    C : float, default 1.0
        Soft-margin penalty (larger → harder margin).
    kernel : {"linear", "rbf"}, default "rbf"
    gamma : {"scale"} | float, default "scale"
        RBF bandwidth. ``"scale"`` → ``1 / (n_features * X.var())``.
    max_iter : int, default 1000
        Maximum number of SMO outer passes.
    tol : float, default 1e-3
        KKT violation tolerance.
    random_state : int | Generator | None, default None
        Shuffle seed inside SMO.

    Attributes
    ----------
    classes_ : ndarray
    support_vectors_ : ndarray
        Training rows with nonzero dual coefficient (union over binary models
        for multiclass).
    dual_coef_ : ndarray
        For binary: shape ``(n_sv,)`` equal to ``α_i y_i`` on support vectors.
        For multiclass: list of per-class OvR dual coef arrays (stored as
        ``dual_coef_list_``); ``dual_coef_`` is the binary case only, or for
        multiclass the stacked OvR ``α y`` on the shared support-vector index
        set is not used — see ``estimators_``.
    intercept_ : float | ndarray
        Bias; scalar for binary, shape ``(n_classes,)`` for OvR.
    n_iter_ : int | list[int]
    """

    def __init__(
        self,
        *,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: Any = "scale",
        max_iter: int = 1000,
        tol: float = 1e-3,
        random_state: int | None = None,
    ) -> None:
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _validate(self) -> None:
        if self.C <= 0:
            raise ValueError("C must be > 0.")
        if self.kernel not in ("linear", "rbf"):
            raise ValueError("kernel must be 'linear' or 'rbf'.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        if self.tol <= 0:
            raise ValueError("tol must be > 0.")

    def _fit_binary(
        self,
        X: np.ndarray,
        y_pm: np.ndarray,
        gamma: float,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        K = kernel_matrix(X, kernel=self.kernel, gamma=gamma)
        alpha, b, n_iter = _smo_binary(
            K, y_pm, C=self.C, tol=self.tol, max_iter=self.max_iter, rng=rng
        )
        sv_mask = alpha > self.tol
        return {
            "alpha": alpha,
            "y_pm": y_pm,
            "b": b,
            "sv_mask": sv_mask,
            "support_vectors": X[sv_mask],
            "dual_coef": (alpha * y_pm)[sv_mask],
            "n_iter": n_iter,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVC":
        self._validate()
        X, y = check_X_y(X, y, y_numeric=False)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("SVC needs at least 2 classes.")

        gamma = resolve_gamma(self.gamma, X)
        self.gamma_ = gamma
        self.X_train_ = X
        rng = check_random_state(self.random_state)

        if n_classes == 2:
            y_pm = np.where(y_idx == 1, 1.0, -1.0)
            model = self._fit_binary(X, y_pm, gamma, rng)
            self._binary_ = True
            self.support_vectors_ = model["support_vectors"]
            self.dual_coef_ = model["dual_coef"]
            self.intercept_ = float(model["b"])
            self.n_iter_ = model["n_iter"]
            self._sv_idx_ = np.where(model["sv_mask"])[0]
            self._y_sv_ = model["y_pm"][model["sv_mask"]]
            self._alpha_sv_ = model["alpha"][model["sv_mask"]]
            # Keep full alpha for decision on training set if needed
            self._alpha_full_ = model["alpha"]
            self._y_pm_full_ = model["y_pm"]
        else:
            self._binary_ = False
            self._ovr_models_ = []
            n_iters = []
            # Collect union of support vectors for the public attribute
            sv_union_mask = np.zeros(X.shape[0], dtype=bool)
            intercepts = []
            for c in range(n_classes):
                y_pm = np.where(y_idx == c, 1.0, -1.0)
                model = self._fit_binary(X, y_pm, gamma, check_random_state(rng.integers(0, 2**31 - 1)))
                self._ovr_models_.append(model)
                sv_union_mask |= model["sv_mask"]
                intercepts.append(model["b"])
                n_iters.append(model["n_iter"])
            self.support_vectors_ = X[sv_union_mask]
            self.dual_coef_ = np.array(
                [m["dual_coef"] for m in self._ovr_models_], dtype=object
            )
            self.intercept_ = np.asarray(intercepts, dtype=np.float64)
            self.n_iter_ = n_iters
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["support_vectors_"])
        X = check_array(X)
        if self._binary_:
            if self.support_vectors_.shape[0] == 0:
                return np.full(X.shape[0], self.intercept_)
            K = kernel_matrix(
                X, self.support_vectors_, kernel=self.kernel, gamma=self.gamma_
            )
            return K @ self.dual_coef_ + self.intercept_
        # OvR: return (n_samples, n_classes) decision values
        scores = np.empty((X.shape[0], len(self.classes_)))
        for c, model in enumerate(self._ovr_models_):
            sv = model["support_vectors"]
            if sv.shape[0] == 0:
                scores[:, c] = model["b"]
            else:
                K = kernel_matrix(X, sv, kernel=self.kernel, gamma=self.gamma_)
                scores[:, c] = K @ model["dual_coef"] + model["b"]
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        if self._binary_:
            idx = (scores >= 0).astype(int)
            return self.classes_[idx]
        return self.classes_[np.argmax(scores, axis=1)]


# ---------------------------------------------------------------------------
# SVR — Kernel Ridge dual (closed form)
# ---------------------------------------------------------------------------


class SVR(BaseEstimator, RegressorMixin):
    """Support Vector Regressor via Kernel Ridge dual (closed form).

    **Design choice.** Rather than ε-insensitive SMO, this didactic SVR solves
    the Kernel Ridge dual

        α = (K + λ I)^{-1} y ,   λ = 1 / C ,
        ŷ(x) = K(x, X) α + b

    with optional intercept estimated as the mean residual on the training
    set when ``fit_intercept=True``. All training points are retained as
    support vectors (dense dual). See PRINCIPLE.md.

    Parameters
    ----------
    C : float, default 1.0
        Inverse ridge strength: ``λ = 1 / C``.
    kernel : {"linear", "rbf"}, default "rbf"
    gamma : {"scale"} | float, default "scale"
    fit_intercept : bool, default True
    """

    def __init__(
        self,
        *,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: Any = "scale",
        fit_intercept: bool = True,
    ) -> None:
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.fit_intercept = fit_intercept

    def _validate(self) -> None:
        if self.C <= 0:
            raise ValueError("C must be > 0.")
        if self.kernel not in ("linear", "rbf"):
            raise ValueError("kernel must be 'linear' or 'rbf'.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVR":
        self._validate()
        X, y = check_X_y(X, y)
        gamma = resolve_gamma(self.gamma, X)
        self.gamma_ = gamma
        self.X_train_ = X

        K = kernel_matrix(X, kernel=self.kernel, gamma=gamma)
        lam = 1.0 / self.C
        # Centering y for intercept (optional)
        if self.fit_intercept:
            y_mean = float(y.mean())
            y_c = y - y_mean
        else:
            y_mean = 0.0
            y_c = y

        alpha = np.linalg.solve(K + lam * np.eye(K.shape[0]), y_c)
        self.dual_coef_ = alpha
        self.support_vectors_ = X  # dense dual — all points
        # Intercept so that mean residual is zero on training data
        if self.fit_intercept:
            pred_no_b = K @ alpha
            self.intercept_ = float(y.mean() - pred_no_b.mean())
        else:
            self.intercept_ = 0.0
        self._y_mean_ = y_mean
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["dual_coef_"])
        X = check_array(X)
        K = kernel_matrix(
            X, self.support_vectors_, kernel=self.kernel, gamma=self.gamma_
        )
        return K @ self.dual_coef_ + self.intercept_
