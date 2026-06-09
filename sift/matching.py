"""
Feature matching.

Implements:
- kd-tree with Best-Bin-First (BBF) approximate nearest-neighbour search
  (Lowe 2004, §5 — 200 bin-checks, L2 distance)
- Lowe's ratio test (d1/d2 < 0.8)
"""

from __future__ import annotations

import heapq
import itertools

import numpy as np
from numpy.typing import NDArray


RATIO_THRESHOLD = 0.8   # Lowe (2004) §7.1
BBF_MAX_CHECKS  = 200   # Lowe (2004) §5

_LEAF_SIZE = 10         # max descriptors per leaf node (implementation choice)


# ---------------------------------------------------------------------------
# kd-tree
# ---------------------------------------------------------------------------

class _KDNode:
    """A single node in the kd-tree (internal or leaf)."""

    __slots__ = ("indices", "split_dim", "split_val", "left", "right")

    def __init__(self) -> None:
        self.indices:   NDArray[np.intp]  = np.empty(0, dtype=np.intp)
        self.split_dim: int               = 0
        self.split_val: float             = 0.0
        self.left:  _KDNode | None        = None
        self.right: _KDNode | None        = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None


def _build(data: NDArray[np.float64], indices: NDArray[np.intp]) -> _KDNode:
    """Recursively build a kd-tree over data[indices]."""
    node = _KDNode()

    if len(indices) <= _LEAF_SIZE:
        node.indices = indices
        return node

    subset = data[indices]                          # (n, D)
    spreads = subset.max(axis=0) - subset.min(axis=0)
    dim = int(np.argmax(spreads))                   # split on widest dimension
    median = float(np.median(subset[:, dim]))

    left_mask  = subset[:, dim] <= median
    right_mask = ~left_mask

    if not left_mask.any() or not right_mask.any():
        # Degenerate: all values identical → make a leaf
        node.indices = indices
        return node

    node.split_dim = dim
    node.split_val = median
    node.left  = _build(data, indices[left_mask])
    node.right = _build(data, indices[right_mask])
    return node


class KDTree:
    """kd-tree wrapper built over a fixed set of descriptor vectors."""

    def __init__(self, data: NDArray[np.float64]) -> None:
        self._data = data
        self._root = _build(data, np.arange(len(data), dtype=np.intp))

    def bbf_knn(
        self,
        query: NDArray[np.float64],
        k: int = 2,
        max_checks: int = BBF_MAX_CHECKS,
    ) -> list[tuple[float, int]]:
        """Best-Bin-First approximate k-NN search (Lowe 2004, §5).

        Returns up to *k* (distance, index) pairs sorted ascending.
        Visits at most *max_checks* leaf entries before stopping.
        """
        data = self._data

        # Min-heap entries: (dist_to_hyperplane², tie-breaker, node).
        # The tie-breaker counter prevents Python from comparing _KDNode objects.
        ctr: itertools.count[int] = itertools.count()
        heap: list[tuple[float, int, _KDNode]] = []
        heapq.heappush(heap, (0.0, next(ctr), self._root))

        best: list[tuple[float, int]] = []   # (squared_dist, global_idx)
        checks = 0

        while heap and checks < max_checks:
            _, _, node = heapq.heappop(heap)

            # Descend to a leaf, pushing unexplored branches onto the heap
            while not node.is_leaf:
                diff = query[node.split_dim] - node.split_val
                if diff <= 0.0:
                    near, far = node.left, node.right
                else:
                    near, far = node.right, node.left
                # Far child may be closer than current best — queue it
                heapq.heappush(heap, (diff * diff, next(ctr), far))
                node = near

            # Evaluate all descriptors in this leaf
            for idx in node.indices:
                d = data[idx] - query
                best.append((float(d @ d), int(idx)))
                checks += 1
                if checks >= max_checks:
                    break

        best.sort(key=lambda x: x[0])
        return [(float(np.sqrt(d2)), idx) for d2, idx in best[:k]]


# ---------------------------------------------------------------------------
# Public matching API
# ---------------------------------------------------------------------------

def match_descriptors(
    desc1: NDArray[np.float64],
    desc2: NDArray[np.float64],
    ratio_threshold: float = RATIO_THRESHOLD,
    max_checks: int = BBF_MAX_CHECKS,
) -> list[tuple[int, int, float]]:
    """Match descriptors using BBF approximate NN search + Lowe ratio test.

    Args:
        desc1: Query descriptors, shape (N, 128).
        desc2: Database descriptors, shape (M, 128).
        ratio_threshold: Lowe ratio test threshold (paper: 0.8).
        max_checks: BBF bin-check limit (paper: 200).

    Returns:
        List of (idx1, idx2, distance) for passing matches.
    """
    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return []
    if desc2.shape[0] < 2:
        return []

    tree = KDTree(desc2)
    matches: list[tuple[int, int, float]] = []

    for i, q in enumerate(desc1):
        nns = tree.bbf_knn(q, k=2, max_checks=max_checks)
        if len(nns) < 2:
            continue
        d0, j0 = nns[0]
        d1, _  = nns[1]
        if d1 > 1e-12 and d0 < ratio_threshold * d1:
            matches.append((int(i), int(j0), float(d0)))

    return matches
