"""Agglomerative (bottom-up) hierarchical clustering."""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, ClusterMixin
from algo_stack.utils.validation import check_array


def _squared_euclidean(X: np.ndarray) -> np.ndarray:
    xx = np.sum(X * X, axis=1)[:, None]
    d2 = xx + xx.T - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


class AgglomerativeClustering(BaseEstimator, ClusterMixin):
    """Naive O(n³) agglomerative clustering.

    Parameters
    ----------
    n_clusters : int, default 2
        Number of clusters to find.
    linkage : {"single", "complete", "average", "ward"}, default "ward"
        Ward linkage requires Euclidean geometry (always used here).

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
    n_clusters_ : int
    children_ : ndarray of shape (n_samples - 1, 2)
        Merge history; each row ``[i, j]`` merges clusters ``i`` and ``j``
        into a new cluster with id ``n_samples + row_index``.
    """

    def __init__(
        self,
        *,
        n_clusters: int = 2,
        linkage: str = "ward",
    ) -> None:
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> "AgglomerativeClustering":
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be >= 1.")
        if self.linkage not in ("single", "complete", "average", "ward"):
            raise ValueError(f"Unknown linkage={self.linkage!r}")
        X = check_array(X)
        n = X.shape[0]
        if self.n_clusters > n:
            raise ValueError(
                f"n_clusters={self.n_clusters} > n_samples={n}."
            )

        # Active cluster ids start as singleton sample indices.
        active = list(range(n))
        # Map cluster id -> member sample indices.
        members: dict[int, list[int]] = {i: [i] for i in range(n)}
        sizes = {i: 1 for i in range(n)}
        children: list[list[int]] = []

        if self.linkage == "ward":
            dist = _squared_euclidean(X)  # store squared distances
        else:
            dist = np.sqrt(_squared_euclidean(X))

        # Pairwise cluster distance matrix indexed by cluster id via a dict
        # of pairs; for simplicity use a dense matrix over current active set.
        # We rebuild a distance lookup among active clusters each merge step
        # via an (n_active x n_active) matrix — O(n³) overall.

        # Maintain D as a dict keyed by frozenset({a,b}) for a < b semantics.
        D: dict[tuple[int, int], float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                D[(i, j)] = float(dist[i, j])

        next_id = n
        while len(active) > self.n_clusters:
            # Find closest pair.
            best_key = None
            best_d = np.inf
            for a_i in range(len(active)):
                for a_j in range(a_i + 1, len(active)):
                    i, j = active[a_i], active[a_j]
                    key = (i, j) if i < j else (j, i)
                    d = D[key]
                    if d < best_d:
                        best_d = d
                        best_key = (i, j)
            assert best_key is not None
            i, j = best_key
            children.append([i, j])

            # New cluster.
            new = next_id
            next_id += 1
            members[new] = members[i] + members[j]
            sizes[new] = sizes[i] + sizes[j]

            # Distances from new cluster to remaining active clusters.
            for k in active:
                if k == i or k == j:
                    continue
                key_ik = (i, k) if i < k else (k, i)
                key_jk = (j, k) if j < k else (k, j)
                dik, djk = D[key_ik], D[key_jk]
                if self.linkage == "single":
                    dnk = min(dik, djk)
                elif self.linkage == "complete":
                    dnk = max(dik, djk)
                elif self.linkage == "average":
                    dnk = (sizes[i] * dik + sizes[j] * djk) / sizes[new]
                else:  # ward — Lance-Williams formula on *squared* distances
                    ni, nj, nk = sizes[i], sizes[j], sizes[k]
                    key_ij = (i, j) if i < j else (j, i)
                    dij = D[key_ij]
                    dnk = (
                        ((ni + nk) * dik + (nj + nk) * djk - nk * dij)
                        / (ni + nj + nk)
                    )
                key_nk = (new, k) if new < k else (k, new)
                D[key_nk] = float(dnk)

            # Drop old pairs involving i or j.
            for k in list(D.keys()):
                if i in k or j in k:
                    del D[k]

            active.remove(i)
            active.remove(j)
            active.append(new)
            del members[i], members[j]
            del sizes[i], sizes[j]

        # Assign labels 0..n_clusters-1 to remaining clusters.
        label_map = {cid: lab for lab, cid in enumerate(sorted(active))}
        labels = np.empty(n, dtype=np.int64)
        for cid, samples in members.items():
            lab = label_map[cid]
            for s in samples:
                labels[s] = lab

        self.labels_ = labels
        self.n_clusters_ = self.n_clusters
        self.children_ = np.asarray(children, dtype=np.int64)
        return self
