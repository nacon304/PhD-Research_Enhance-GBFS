# graph_fs.py
# ---------------------------------------------------------
# Graph-based feature selection wrappers:
#   - InfFS (PyIFS)
#   - UGFS (UGFS.feature_ugfs.do_ugfs)
#
# Fix:
#   - InfFS sometimes returns complex WEIGHT (numerical artifacts).
#   - DO NOT cast complex -> float directly (causes ComplexWarning).
#   - Convert complex safely:
#       * if imag is tiny (numerical noise): take real part
#       * else: use magnitude abs(.) for scoring
# ---------------------------------------------------------

from __future__ import annotations

import importlib
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler

import PyIFS
from UGFS.feature_ugfs import do_ugfs


# =====================================================
# 0. Patch cho PyIFS.InfFS (bắt buộc)
# =====================================================
_m = importlib.import_module("PyIFS.InfFS")
_m.math = math
_m.stats = stats


# =====================================================
# 0. Helpers
# =====================================================
def _safe_real_scores(
    v: Any,
    *,
    name: str = "scores",
    verbose: int = 0,
    imag_tol_rel: float = 1e-10,
) -> np.ndarray:
    """
    Convert a score vector to a real-valued float array safely.

    - If complex with tiny imaginary part (numerical noise): take real part.
    - If complex with significant imaginary part: take magnitude abs(.).
    - Replace non-finite values with 0.0 (so they rank last under abs-based ranking).
    """
    arr = np.asarray(v).ravel()

    if np.iscomplexobj(arr):
        imag_max = float(np.max(np.abs(arr.imag))) if arr.size else 0.0
        real_max = float(np.max(np.abs(arr.real))) if arr.size else 0.0

        # tiny imag => treat as numerical noise
        if imag_max <= imag_tol_rel * max(1.0, real_max):
            if verbose:
                print(f"[{name}] complex detected but imag is tiny "
                      f"(imag_max={imag_max:.3e}, real_max={real_max:.3e}) -> take real")
            arr = arr.real
        else:
            if verbose:
                print(f"[{name}] complex detected with non-trivial imag "
                      f"(imag_max={imag_max:.3e}, real_max={real_max:.3e}) -> take abs")
            arr = np.abs(arr)

    arr = np.asarray(arr, dtype=float)

    # sanitize non-finite
    bad = ~np.isfinite(arr)
    if np.any(bad):
        if verbose:
            print(f"[{name}] non-finite values found: {int(np.sum(bad))} -> set to 0.0")
        arr[bad] = 0.0

    return arr


def _rank_by_abs(scores: np.ndarray) -> np.ndarray:
    """
    Rank features by descending abs(score), stable for NaNs/Infs already handled.
    """
    abs_s = np.abs(scores)
    # abs_s should already be finite; still keep safe
    abs_s[~np.isfinite(abs_s)] = 0.0
    return np.argsort(-abs_s)


# =====================================================
# 1. InfFS: ranking & scores
# =====================================================
def inffs_rank(
    X,
    y,
    alpha: float = 0.5,
    supervision: int = 1,
    verbose: int = 0,
    standardize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Chạy InfFS để lấy score và ranking cho toàn bộ feature.

    Returns
    -------
    scores : np.ndarray, shape (n_features,)
        Score từng feature (real float, đã sanitize).
    ranking : np.ndarray, shape (n_features,)
        Chỉ số feature (0-based) sắp xếp giảm dần theo abs(score).
    extra : dict
        {"RANKED": RANKED, "WEIGHT_raw": WEIGHT_raw, "WEIGHT": WEIGHT_sanitized}
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int).ravel()

    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    inf = PyIFS.InfFS()
    RANKED, WEIGHT = inf.infFS(X, y, alpha, supervision, verbose)

    WEIGHT_raw = np.asarray(WEIGHT).ravel()
    WEIGHT_safe = _safe_real_scores(WEIGHT_raw, name="InfFS/WEIGHT", verbose=verbose)

    # ranking dùng abs(score) để consistent với build_fronts_from_scores
    ranking = _rank_by_abs(WEIGHT_safe)

    extra = {
        "RANKED": np.asarray(RANKED, dtype=int).ravel(),
        "WEIGHT_raw": WEIGHT_raw,
        "WEIGHT": WEIGHT_safe,
    }
    return WEIGHT_safe, ranking, extra


# =====================================================
# 2. UGFS: ranking & scores
# =====================================================
def ugfs_rank(
    X,
    y=None,  # ignored (UGFS unsupervised)
    ndim: Optional[int] = None,
    nbdk: int = 5,
    varthr: float = 2.0,
    preprocess: str = "cscale",
    standardize: bool = True,
    verbose: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Chạy UGFS để lấy PageRank score & ranking.

    Returns
    -------
    scores : np.ndarray, shape (n_features,)
        PageRank scores (real float, sanitized).
    ranking : np.ndarray, shape (n_features,)
        Chỉ số feature (0-based) sắp xếp giảm dần theo abs(score).
    extra : dict
        Kết quả đầy đủ của do_ugfs.
    """
    X = np.asarray(X, dtype=float)
    n_features = X.shape[1]

    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    if ndim is None:
        ndim = min(20, max(1, n_features // 2))

    result = do_ugfs(
        X,
        ndim=ndim,
        nbdk=nbdk,
        varthr=varthr,
        preprocess=preprocess,
    )

    prscore_raw = np.asarray(result.get("prscore", []))
    prscore = _safe_real_scores(prscore_raw, name="UGFS/prscore", verbose=verbose)

    ranking = _rank_by_abs(prscore)

    # keep featidx if needed
    result["featidx"] = np.asarray(result.get("featidx", []), dtype=int).ravel()
    result["prscore"] = prscore

    return prscore, ranking, result


# =====================================================
# 3. Hàm tổng: dùng cho code compare
# =====================================================
def run_graph_fs_methods(
    X_train,
    y_train,
    k: int = 20,
    use_inffs: bool = True,
    use_ugfs: bool = True,
    inffs_params: Optional[Dict[str, Any]] = None,
    ugfs_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Chạy các phương pháp graph-based FS (InfFS, UGFS) trên X_train, y_train.

    Returns
    -------
    scores_dict : dict[name] -> score vector shape (n_features,)
    selected_dict : dict[name] -> np.ndarray chỉ số feature (0-based) top-k
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int).ravel()
    n_features = X_train.shape[1]

    if k is None or k > n_features:
        k = n_features
    k = max(1, int(k))

    scores_dict: Dict[str, np.ndarray] = {}
    selected_dict: Dict[str, np.ndarray] = {}

    # --- InfFS ---
    if use_inffs:
        params = dict(inffs_params) if inffs_params else {}
        scores_inffs, ranking_inffs, _ = inffs_rank(X_train, y_train, **params)
        topk = ranking_inffs[:k].astype(int)

        scores_dict["InfFS"] = scores_inffs
        selected_dict["InfFS"] = topk

    # --- UGFS ---
    if use_ugfs:
        params = dict(ugfs_params) if ugfs_params else {}
        # UGFS ignores y_train but keep signature consistent
        scores_ugfs, ranking_ugfs, _ = ugfs_rank(X_train, y_train, **params)
        topk = ranking_ugfs[:k].astype(int)

        scores_dict["UGFS"] = scores_ugfs
        selected_dict["UGFS"] = topk

    return scores_dict, selected_dict


# =====================================================
# 4. Quick self-test (optional)
# =====================================================
if __name__ == "__main__":
    from myinputdatasetXD import myinputdatasetXD

    dataset, labels, datasetName = myinputdatasetXD(1)
    X = dataset[:, 1:].astype(float)
    y = labels.astype(int).ravel()

    print(f"[TEST] Dataset: {datasetName}, X shape = {X.shape}")

    scores_dict, selected_dict = run_graph_fs_methods(
        X, y,
        k=20,
        use_inffs=True,
        use_ugfs=True,
        inffs_params={"alpha": 0.5, "supervision": 1, "verbose": 1, "standardize": True},
        ugfs_params={"ndim": None, "nbdk": 5, "varthr": 2.0, "preprocess": "cscale", "standardize": True, "verbose": 1},
    )

    for name in scores_dict:
        print(f"{name}: scores shape = {scores_dict[name].shape}, "
              f"top-k idx shape = {selected_dict[name].shape}")
