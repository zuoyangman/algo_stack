"""Curated evaluation metrics (classification, regression, clustering)."""

from __future__ import annotations

import numpy as np

from algo_stack.utils.validation import check_array


def precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    pos_label: int | float | str = 1,
) -> tuple[float, float, float]:
    """Binary precision, recall, and F1 for the positive class.

    Returns
    -------
    precision, recall, f1 : float
    """

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred have different shapes: {y_true.shape} vs {y_pred.shape}."
        )

    tp = float(np.sum((y_true == pos_label) & (y_pred == pos_label)))
    fp = float(np.sum((y_true != pos_label) & (y_pred == pos_label)))
    fn = float(np.sum((y_true == pos_label) & (y_pred != pos_label)))

    precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
    denom = precision + recall
    f1 = 2.0 * precision * recall / denom if denom > 0.0 else 0.0
    return precision, recall, f1


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: np.ndarray | list | None = None,
) -> np.ndarray:
    """Compute a confusion matrix ``C`` where ``C[i, j]`` is the count of
    true label ``labels[i]`` predicted as ``labels[j]``.
    """

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred have different shapes: {y_true.shape} vs {y_pred.shape}."
        )

    if labels is None:
        labels_arr = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels_arr = np.asarray(labels)

    index = {lab: i for i, lab in enumerate(labels_arr.tolist())}
    n = labels_arr.size
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if t in index and p in index:
            cm[index[t], index[p]] += 1
    return cm


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient over all samples.

    Requires at least two distinct labels. Uses Euclidean distance.
    """

    X = check_array(X)
    labels = np.asarray(labels).ravel()
    if X.shape[0] != labels.shape[0]:
        raise ValueError(
            f"X and labels have different numbers of samples: "
            f"{X.shape[0]} vs {labels.shape[0]}."
        )

    unique = np.unique(labels)
    if unique.size < 2:
        raise ValueError("silhouette_score requires at least 2 distinct labels.")
    if unique.size >= X.shape[0]:
        raise ValueError(
            "silhouette_score requires fewer clusters than samples "
            "(each cluster needs room for intra-cluster distances)."
        )

    n = X.shape[0]
    # Pairwise squared distances via (a-b)^2 = a^2 + b^2 - 2 a·b
    gram = X @ X.T
    sq = np.sum(X * X, axis=1, keepdims=True)
    dists = np.sqrt(np.maximum(sq + sq.T - 2.0 * gram, 0.0))

    sil = np.empty(n, dtype=np.float64)
    for i in range(n):
        same = labels == labels[i]
        n_same = int(np.sum(same))
        if n_same <= 1:
            # Singleton cluster → silhouette defined as 0.
            sil[i] = 0.0
            continue

        a = float(np.sum(dists[i, same]) / (n_same - 1))
        b = np.inf
        for c in unique:
            if c == labels[i]:
                continue
            mask = labels == c
            b = min(b, float(np.mean(dists[i, mask])))
        denom = max(a, b)
        sil[i] = 0.0 if denom == 0.0 else (b - a) / denom

    return float(np.mean(sil))


def log_loss(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    eps: float = 1e-15,
) -> float:
    """Logarithmic loss (cross-entropy).

    Parameters
    ----------
    y_true : array-like
        Binary labels ``{0, 1}``, integer class indices, or one-hot rows.
    y_prob : array-like
        Predicted probabilities. Shape ``(n,)`` / ``(n, 1)`` for binary,
        or ``(n, n_classes)`` for multiclass.
    eps : float
        Clip probabilities into ``[eps, 1 − eps]`` for numerical stability.
    """

    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_true_arr = np.asarray(y_true)
    y_prob = np.clip(y_prob, eps, 1.0 - eps)

    if y_prob.ndim == 1 or (y_prob.ndim == 2 and y_prob.shape[1] == 1):
        p = y_prob.ravel()
        y = y_true_arr.ravel().astype(np.float64)
        if p.shape != y.shape:
            raise ValueError(
                f"y_true and y_prob have incompatible shapes: "
                f"{y_true_arr.shape} vs {y_prob.shape}."
            )
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    if y_prob.ndim != 2:
        raise ValueError(f"y_prob must be 1-D or 2-D, got shape {y_prob.shape}.")

    n_samples, n_classes = y_prob.shape
    if y_true_arr.ndim == 1:
        y_idx = y_true_arr.astype(int)
        if y_idx.shape[0] != n_samples:
            raise ValueError("y_true and y_prob sample counts differ.")
        if np.any(y_idx < 0) or np.any(y_idx >= n_classes):
            raise ValueError("y_true class index out of range for y_prob.")
        Y = np.zeros((n_samples, n_classes), dtype=np.float64)
        Y[np.arange(n_samples), y_idx] = 1.0
    elif y_true_arr.ndim == 2:
        Y = y_true_arr.astype(np.float64)
        if Y.shape != y_prob.shape:
            raise ValueError(
                f"one-hot y_true shape {Y.shape} != y_prob shape {y_prob.shape}."
            )
    else:
        raise ValueError(f"y_true must be 1-D or 2-D, got shape {y_true_arr.shape}.")

    return float(-np.mean(np.sum(Y * np.log(y_prob), axis=1)))
