"""
Unsupervised Graph-based Feature Selection (UGFS)

Python reimplementation of R function do.ugfs from Rdimtools::feature_UGFS.

UGFS is an unsupervised feature selection method with two parameters `nbdk`
and `varthr`. It constructs an affinity graph using local variance computation
and scores variables based on the PageRank algorithm.
"""

from typing import Literal, Dict, Any, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors

# -------------------------------------------------------------------------
# External helpers (will be provided in other files)
# -------------------------------------------------------------------------
# You said: "hàm nào thiếu thì cứ import tôi sẽ gửi file sau"
# nên ở đây mình chỉ import, không tự implement.

from UGFS.auxiliary import (
    aux_typecheck,
    check_ndim,
    aux_preprocess_hidden,
    aux_featureindicator,
)
from UGFS.pagerank_utils import v2aux_pagerank  # bạn sẽ tạo file này sau


# -------------------------------------------------------------------------
# Main function: do_ugfs
# -------------------------------------------------------------------------
def do_ugfs(
    X: np.ndarray,
    ndim: int = 2,
    nbdk: int = 5,
    varthr: float = 2.0,
    preprocess: Literal["null", "center", "scale", "cscale", "whiten", "decorrelate"] = "null",
) -> Dict[str, Any]:
    """
    Unsupervised Graph-based Feature Selection (UGFS).

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data matrix whose rows are observations and columns are features.
    ndim : int, optional (default=2)
        Target number of selected features (embedding dimension).
    nbdk : int, optional (default=5)
        Size of neighborhood for local variance computation (k in KNN).
    varthr : float, optional (default=2.0)
        Threshold value for affinity graph construction. If too small so that
        the variable-graph is not constructed, an error is raised.
    preprocess : {"null","center","scale","cscale","whiten","decorrelate"}
        Preprocessing option. Implemented inside `aux_preprocess_hidden`.

    Returns
    -------
    result : dict
        {
          "Y": (n_samples, ndim) embedded data (selected features after preprocessing),
          "A": (p, p) affinity matrix between features,
          "prscore": (p,) PageRank scores for each feature,
          "featidx": (ndim,) indices (0-based) of selected features,
          "trfinfo": preprocessing transform info,
          "projection": (p, ndim) feature indicator (basis for projection)
        }
    """
    # ------------------------------------------------------------------
    # PREPROCESSING
    # 1. type check
    aux_typecheck(X)
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    # 2. ndim
    ndim = int(ndim)
    if not check_ndim(ndim, p):
        raise ValueError("* do_ugfs : 'ndim' must be a positive integer in [1, #(covariates)].")

    # 3. preprocess
    algpreprocess = preprocess if preprocess is not None else "null"

    # 4. extra parameters
    myk = int(round(nbdk))
    mythr = float(varthr)

    # ------------------------------------------------------------------
    # COMPUTATION : Preliminary
    #  Preprocessing
    tmplist = aux_preprocess_hidden(X, type=algpreprocess, algtype="linear")
    trfinfo = tmplist["info"]
    pX = tmplist["pX"]  # (n, p)

    #  Nearest-neighbor (equivalent to RANN::nn2 with k = myk + 1)
    #  nn_indices: shape (n, myk+1), first column is self, then neighbors
    knn = NearestNeighbors(n_neighbors=myk + 1, algorithm="auto")
    knn.fit(pX)
    nn_indices = knn.kneighbors(return_distance=False)

    # ------------------------------------------------------------------
    # COMPUTATION : Main
    # Step 1. compute local variance
    Varji = np.zeros((n, p), dtype=float)

    # Trong R code:
    #   tgtvec = pX[i, ]
    #   tgtmat = pX[as.vector(knnnbd$nn.idx[, 2:(myk+1)]), ]
    # Lưu ý: tgtmat KHÔNG phụ thuộc vào i, chỉ tgtvec thay đổi.
    # Họ dùng toàn bộ neighbor của tất cả điểm làm "mat".
    neighbor_block = nn_indices[:, 1 : (myk + 1)]  # (n, myk)
    neighbor_flat = neighbor_block.reshape(-1)      # length = n * myk
    tgtmat_global = pX[neighbor_flat, :]           # (n*myk, p)

    for i in range(n):
        tgtvec = pX[i, :]  # (p,)
        Varji[i, :] = ugfs_var(tgtvec, tgtmat_global)

    print(
        f"* do_ugfs : range of variances is "
        f"[{Varji.min():.2f}, {Varji.max():.2f}]."
    )

    # Step 2. binarize
    VarBin = np.zeros((n, p), dtype=int)
    VarBin[Varji <= mythr] = 1

    if np.all(VarBin == 0):
        raise ValueError(
            "* do_ugfs : 'varthr' is too small; graph may not be constructed. Use larger value."
        )

    # Step 3. graph construction (A: p x p)
    A = np.zeros((p, p), dtype=float)
    for i in range(n):
        # Sp = { j | VarBin[i,j] > 0.5 }
        Sp = np.where(VarBin[i, :] > 0.5)[0]
        if Sp.size > 0:
            # A[Sp, Sp] = 1 (full clique on these features)
            A[np.ix_(Sp, Sp)] = 1.0

    # Remove self-loops
    np.fill_diagonal(A, 0.0)

    # Step 4. compute the score via PageRank algorithm
    pgscore = np.asarray(v2aux_pagerank(A)).ravel()
    # Nếu muốn dùng bản nội bộ:
    # pgscore = my_pagerank(A)

    # Step 5. select the largest ones and find the projection
    idxvec = np.argsort(pgscore)[::-1][:ndim]  # 0-based indices, descending score
    projection = aux_featureindicator(p, ndim, idxvec)

    # ------------------------------------------------------------------
    # RETURN
    Y = pX @ projection  # (n, ndim)

    result = {
        "Y": Y,
        "A": A,
        "prscore": pgscore,
        "featidx": idxvec,
        "trfinfo": trfinfo,
        "projection": projection,
    }
    return result


# -------------------------------------------------------------------------
# Auxiliary functions (translated from R)
# -------------------------------------------------------------------------
def ugfs_var(vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """
    Compute local variance for all variables at once.

    Parameters
    ----------
    vec : (p,)
        Feature values at one sample (target point).
    mat : (k, p)
        Feature values at neighbor points.

    Returns
    -------
    output : (p,)
        Local variance per feature.
    """
    vec = np.asarray(vec, dtype=float).ravel()
    mat = np.asarray(mat, dtype=float)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)

    k = mat.shape[0]
    # R code:
    # output[i] = sum((tgtval - tgtvec)^2) / k
    # Vectorized version:
    diff = mat - vec[None, :]      # (k, p)
    output = (diff ** 2).sum(axis=0) / float(k)
    return output


def my_pagerank(A: np.ndarray, d: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """
    Simple PageRank implementation (R's mypagerank).

    Parameters
    ----------
    A : (N, N) adjacency matrix
    d : float
        Damping factor.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance on L2 distance.

    Returns
    -------
    R : (N,) PageRank scores.
    """
    A = np.asarray(A, dtype=float)
    N = A.shape[0]

    # row sums (out-degrees)
    k = A.sum(axis=1)
    # tránh chia cho 0: nếu hàng toàn 0 thì đặt k=1 (dangling node)
    k_safe = np.where(k == 0, 1.0, k)

    # M = t(diag(1/k) %*% A)  trong R
    # diag(1/k) %*% A -> scale từng hàng
    M = (A / k_safe[:, None]).T  # (N, N) transpose để tương ứng

    R_old = np.full(N, 1.0 / N)
    teleport = (1.0 - d) / N

    for _ in range(max_iter):
        R_new = d * (M @ R_old) + teleport
        diff = R_old - R_new
        R_inc = np.linalg.norm(diff)
        R_old = R_new
        if R_inc < tol:
            break

    return R_old
