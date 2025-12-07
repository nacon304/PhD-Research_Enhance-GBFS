"""
pagerank_utils.py

Python reimplementation of Rcpp function v2aux_pagerank:

// [[Rcpp::export]]
arma::vec v2aux_pagerank(arma::mat& A) { ... }
"""

from __future__ import annotations
import numpy as np


def v2aux_pagerank(A: np.ndarray) -> np.ndarray:
    """
    PageRank implementation equivalent to Rcpp v2aux_pagerank.

    Parameters
    ----------
    A : ndarray, shape (N, N)
        Adjacency matrix of the graph. A[i, j] > 0 means an edge
        from node i to node j (row i -> column j).

    Returns
    -------
    R : ndarray, shape (N,)
        PageRank scores.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("v2aux_pagerank: A must be a square (N x N) matrix.")

    N = A.shape[0]
    NN = float(N)
    d = 0.85  # damping factor

    maxiter = 100
    thr = 1e-3  # convergence threshold (L2 norm)

    # ------------------------------------------------------------------
    # prepare
    # sumA: row sums
    sumA = A.sum(axis=1)  # (N,)
    Kvec = np.zeros(N, dtype=float)

    # Rcpp code:
    # if (sumA(n) > 0) Kvec(n) = 1.0 / sumA(n);
    Kvec[sumA > 0] = 1.0 / sumA[sumA > 0]

    # Kinv is diag(Kvec)
    Kinv = np.diag(Kvec)       # (N, N)
    M = (Kinv @ A).T           # (N, N)  # transpose như R: arma::trans(Kinv*A)

    # ------------------------------------------------------------------
    # iterate
    scterm = np.full(N, (1.0 - d) / NN, dtype=float)
    Rold = np.full(N, 1.0 / NN, dtype=float)
    Rnew = np.zeros(N, dtype=float)

    for it in range(maxiter):
        # update
        Rnew = d * (M @ Rold) + scterm
        Rinc = np.linalg.norm(Rnew - Rold, ord=2)
        Rold = Rnew

        # điều kiện dừng: it > 5 & Rinc < thr
        if (Rinc < thr) and (it > 5):
            break

    return Rold
