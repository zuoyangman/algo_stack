"""Spatial neighbour indexes: KD-Tree and Ball-Tree (NumPy, didactic).

Both structures accelerate exact nearest-neighbour search in moderate
dimensions. For ``d ≳ 20`` the curse of dimensionality usually makes
brute-force matrix multiply competitive again — callers should pick via
``algorithm="auto"``.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from algo_stack.utils.validation import check_array


def _push_neighbor(
    heap: list[tuple[float, int]], dist: float, idx: int, k: int
) -> None:
    """Maintain a max-heap of size ≤ k keyed by distance (farthest on top).

    ``heapq`` is a min-heap, so we store ``(-dist, idx)``.
    """

    item = (-dist, idx)
    if len(heap) < k:
        heapq.heappush(heap, item)
    elif dist < -heap[0][0]:
        heapq.heapreplace(heap, item)


def _heap_to_sorted(heap: list[tuple[float, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Convert neighbour max-heap to ascending (distances, indices)."""

    ordered = sorted(((-neg_d, idx) for neg_d, idx in heap), key=lambda t: t[0])
    dists = np.array([d for d, _ in ordered], dtype=np.float64)
    idxs = np.array([i for _, i in ordered], dtype=np.int64)
    return dists, idxs


# ---------------------------------------------------------------------------
# KD-Tree
# ---------------------------------------------------------------------------


@dataclass
class _KDNode:
    point_idx: int
    point: np.ndarray
    axis: int
    left: "_KDNode | None" = None
    right: "_KDNode | None" = None


class KDTree:
    """Axis-aligned binary space partition for Euclidean nearest neighbours.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training points (copied and stored).
    leaf_size : int, default 1
        Reserved for future leaf packing; currently one point per node.
    """

    def __init__(self, X: np.ndarray, *, leaf_size: int = 1) -> None:
        self.data_ = check_array(X).copy()
        self.leaf_size = leaf_size
        self.n_features_ = self.data_.shape[1]
        indices = np.arange(self.data_.shape[0])
        self.root_ = self._build(indices, depth=0)

    def _build(self, indices: np.ndarray, depth: int) -> _KDNode | None:
        if indices.size == 0:
            return None
        axis = depth % self.n_features_
        order = indices[np.argsort(self.data_[indices, axis])]
        mid = order.size // 2
        node = _KDNode(
            point_idx=int(order[mid]),
            point=self.data_[order[mid]],
            axis=axis,
        )
        node.left = self._build(order[:mid], depth + 1)
        node.right = self._build(order[mid + 1 :], depth + 1)
        return node

    def query(
        self, X: np.ndarray, *, k: int = 1, return_distance: bool = True
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Query the ``k`` nearest neighbours of each row in ``X``."""

        X = check_array(X)
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X.shape[1]}."
            )
        k = min(k, self.data_.shape[0])
        n_q = X.shape[0]
        dists = np.empty((n_q, k), dtype=np.float64)
        idxs = np.empty((n_q, k), dtype=np.int64)
        for i in range(n_q):
            heap: list[tuple[float, int]] = []
            self._search(self.root_, X[i], k, heap)
            d_row, i_row = _heap_to_sorted(heap)
            dists[i] = d_row
            idxs[i] = i_row
        if return_distance:
            return dists, idxs
        return idxs

    def _search(
        self,
        node: _KDNode | None,
        q: np.ndarray,
        k: int,
        heap: list[tuple[float, int]],
    ) -> None:
        if node is None:
            return
        d = float(np.linalg.norm(q - node.point))
        _push_neighbor(heap, d, node.point_idx, k)

        axis = node.axis
        diff = float(q[axis] - node.point[axis])
        near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
        self._search(near, q, k, heap)
        radius = -heap[0][0] if len(heap) >= k else np.inf
        if abs(diff) <= radius:
            self._search(far, q, k, heap)


# ---------------------------------------------------------------------------
# Ball-Tree
# ---------------------------------------------------------------------------


@dataclass
class _BallNode:
    center: np.ndarray
    radius: float
    point_indices: np.ndarray
    left: "_BallNode | None" = None
    right: "_BallNode | None" = None
    is_leaf: bool = False


class BallTree:
    """Metric ball-tree for Euclidean nearest neighbours.

    Each node stores a bounding ball ``(center, radius)`` covering its
    subtree. The far child can be pruned when
    ``||q − center|| − radius >= current_worst``.
    """

    def __init__(self, X: np.ndarray, *, leaf_size: int = 16) -> None:
        self.data_ = check_array(X).copy()
        self.leaf_size = max(1, int(leaf_size))
        self.n_features_ = self.data_.shape[1]
        indices = np.arange(self.data_.shape[0])
        self.root_ = self._build(indices)

    def _build(self, indices: np.ndarray) -> _BallNode:
        pts = self.data_[indices]
        center = pts.mean(axis=0)
        radius = (
            float(np.max(np.linalg.norm(pts - center, axis=1))) if len(pts) else 0.0
        )
        if indices.size <= self.leaf_size:
            return _BallNode(
                center=center,
                radius=radius,
                point_indices=indices.copy(),
                is_leaf=True,
            )
        spreads = pts.max(axis=0) - pts.min(axis=0)
        axis = int(np.argmax(spreads))
        order = indices[np.argsort(self.data_[indices, axis])]
        mid = max(1, order.size // 2)
        if mid >= order.size:
            mid = order.size - 1
        left = self._build(order[:mid])
        right = self._build(order[mid:])
        return _BallNode(
            center=center,
            radius=radius,
            point_indices=np.empty(0, dtype=np.int64),
            left=left,
            right=right,
            is_leaf=False,
        )

    def query(
        self, X: np.ndarray, *, k: int = 1, return_distance: bool = True
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        X = check_array(X)
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X.shape[1]}."
            )
        k = min(k, self.data_.shape[0])
        n_q = X.shape[0]
        dists = np.empty((n_q, k), dtype=np.float64)
        idxs = np.empty((n_q, k), dtype=np.int64)
        for i in range(n_q):
            heap: list[tuple[float, int]] = []
            self._search(self.root_, X[i], k, heap)
            d_row, i_row = _heap_to_sorted(heap)
            dists[i] = d_row
            idxs[i] = i_row
        if return_distance:
            return dists, idxs
        return idxs

    def _search(
        self,
        node: _BallNode | None,
        q: np.ndarray,
        k: int,
        heap: list[tuple[float, int]],
    ) -> None:
        if node is None:
            return
        dist_center = float(np.linalg.norm(q - node.center))
        worst = -heap[0][0] if len(heap) >= k else np.inf
        if dist_center - node.radius >= worst:
            return

        if node.is_leaf:
            for idx in node.point_indices:
                d = float(np.linalg.norm(q - self.data_[int(idx)]))
                _push_neighbor(heap, d, int(idx), k)
            return

        assert node.left is not None and node.right is not None
        d_left = float(np.linalg.norm(q - node.left.center))
        d_right = float(np.linalg.norm(q - node.right.center))
        first, second = (
            (node.left, node.right) if d_left <= d_right else (node.right, node.left)
        )
        self._search(first, q, k, heap)
        self._search(second, q, k, heap)
