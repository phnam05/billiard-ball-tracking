"""Optimal detection-to-track assignment.

The old tracker had no association step at all: it took the largest contour and
assumed it was the cue ball.  With more than one ball on the table that is a
coin flip, and after a collision it reliably swaps identities.

Globally optimal assignment (Hungarian / Jonker-Volgenant) fixes this by
choosing the set of pairings that minimises *total* cost, instead of letting an
early greedy choice steal the detection a later track needed.

``scipy.optimize.linear_sum_assignment`` is used when SciPy is installed; the
pure-NumPy implementation below is a drop-in fallback so the package has no hard
SciPy dependency.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _scipy_lsa  # type: ignore

    HAVE_SCIPY = True
except Exception:  # pragma: no cover - environment dependent
    _scipy_lsa = None
    HAVE_SCIPY = False


def _hungarian(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Jonker-Volgenant shortest-augmenting-path assignment, O(n^3).

    Returns row and column index arrays, as ``scipy`` does.  Requires a finite
    cost matrix: callers substitute a large finite sentinel for "forbidden" and
    drop those pairs afterwards.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    transposed = False
    if cost.shape[0] > cost.shape[1]:
        cost = cost.T
        transposed = True

    n, m = cost.shape
    INF = np.inf

    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(m + 1, dtype=np.float64)
    p = np.zeros(m + 1, dtype=np.int64)  # p[j] = row matched to column j (1-based)
    way = np.zeros(m + 1, dtype=np.int64)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF, dtype=np.float64)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            free = ~used[1:]
            if np.any(free):
                cur = cost[i0 - 1, :] - u[i0] - v[1:]
                better = free & (cur < minv[1:])
                if np.any(better):
                    minv[1:][better] = cur[better]
                    way[1:][better] = j0
                candidates = np.where(free, minv[1:], INF)
                j1 = int(np.argmin(candidates)) + 1
                delta = float(candidates[j1 - 1])
            if not np.isfinite(delta):
                break
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0 != 0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    rows = []
    cols = []
    for j in range(1, m + 1):
        if p[j] > 0:
            rows.append(p[j] - 1)
            cols.append(j - 1)

    r = np.asarray(rows, dtype=int)
    c = np.asarray(cols, dtype=int)
    order = np.argsort(r if not transposed else c)
    r, c = r[order], c[order]
    if transposed:
        return c, r
    return r, c


def solve(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Minimum-cost assignment; SciPy when available, NumPy otherwise."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    if HAVE_SCIPY:
        r, c = _scipy_lsa(cost)
        return np.asarray(r, dtype=int), np.asarray(c, dtype=int)
    return _hungarian(cost)


#: Cost used for pairs that must never be matched.  Finite so the solver stays
#: well defined, but far above any real cost.
FORBIDDEN = 1e6


def associate(
    cost: np.ndarray, forbidden_threshold: float = FORBIDDEN / 2.0
) -> Tuple[list, list, list]:
    """Assign rows to columns and report what went unmatched.

    Returns ``(matches, unmatched_rows, unmatched_cols)`` where ``matches`` is a
    list of ``(row, col)`` pairs whose cost was below ``forbidden_threshold``.
    """
    cost = np.asarray(cost, dtype=np.float64)
    n_rows, n_cols = (cost.shape if cost.ndim == 2 else (0, 0))
    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols))

    rows, cols = solve(cost)
    matches = []
    matched_rows = set()
    matched_cols = set()
    for r, c in zip(rows, cols):
        if cost[r, c] < forbidden_threshold:
            matches.append((int(r), int(c)))
            matched_rows.add(int(r))
            matched_cols.add(int(c))

    unmatched_rows = [i for i in range(n_rows) if i not in matched_rows]
    unmatched_cols = [j for j in range(n_cols) if j not in matched_cols]
    return matches, unmatched_rows, unmatched_cols
