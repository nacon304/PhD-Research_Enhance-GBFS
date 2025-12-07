import importlib
import math
import numpy as np
import scipy.stats as stats

from sklearn.preprocessing import StandardScaler

import PyIFS
from UGFS.feature_ugfs import do_ugfs   # bạn đã có sẵn trong code UGFS


# =====================================================
# 0. Patch cho PyIFS.InfFS (bắt buộc)
# =====================================================
_m = importlib.import_module("PyIFS.InfFS")
_m.math = math
_m.stats = stats


# =====================================================
# 1. InfFS: ranking & scores
# =====================================================

def inffs_rank(
    X,
    y,
    alpha=0.5,
    supervision=1,
    verbose=0,
    standardize=True,
):
    """
    Chạy InfFS để lấy score và ranking cho toàn bộ feature.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
    y : array, shape (n_samples,)
    standardize : bool
        Nếu True, chuẩn hóa X bằng StandardScaler trước khi chạy InfFS.

    Returns
    -------
    scores : np.ndarray, shape (n_features,)
        WEIGHT từ InfFS (score từng feature).
    ranking : np.ndarray, shape (n_features,)
        Chỉ số feature (0-based) sắp xếp giảm dần theo score.
    extra : dict
        {"RANKED": RANKED, "WEIGHT": WEIGHT}
    """
    X = np.asarray(X, float)
    y = np.asarray(y, int).ravel()

    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    inf = PyIFS.InfFS()
    RANKED, WEIGHT = inf.infFS(X, y, alpha, supervision, verbose)
    WEIGHT = np.asarray(WEIGHT, float).ravel()
    RANKED = np.asarray(RANKED, int).ravel()  # đã là ranking giảm dần

    # ranking dùng abs(score) để consistent với build_fronts_from_scores
    ranking = np.argsort(-np.abs(WEIGHT))

    scores = WEIGHT
    extra = {
        "RANKED": RANKED,
        "WEIGHT": WEIGHT,
    }
    return scores, ranking, extra


# =====================================================
# 2. UGFS: ranking & scores
# =====================================================

def ugfs_rank(
    X,
    y=None,
    ndim=None,
    nbdk=5,
    varthr=2.0,
    preprocess="cscale",
    standardize=True,
):
    """
    Chạy UGFS để lấy PageRank score & ranking.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
    y : ignored (UGFS là unsupervised)
    ndim : int or None
        Số chiều nhúng. Nếu None, lấy min(20, p//2).
        (prscore và featidx vẫn được tính cho toàn bộ features)
    nbdk, varthr, preprocess : tham số UGFS
    standardize : bool
        Nếu True, chuẩn hóa X bằng StandardScaler trước khi chạy UGFS.

    Returns
    -------
    scores : np.ndarray, shape (n_features,)
        PageRank scores (prscore) cho từng feature.
    ranking : np.ndarray, shape (n_features,)
        Chỉ số feature (0-based) sắp xếp giảm dần theo score.
    extra : dict
        Kết quả đầy đủ của do_ugfs (Y, A, prscore, featidx, projection).
    """
    X = np.asarray(X, float)
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

    prscore = np.asarray(result["prscore"], float).ravel()
    featidx = np.asarray(result["featidx"], int).ravel()

    # ranking theo abs(score)
    scores = prscore
    ranking = np.argsort(-np.abs(scores))

    extra = result
    return scores, ranking, extra


# =====================================================
# 3. Hàm tổng: dùng cho code compare
# =====================================================

def run_graph_fs_methods(
    X_train,
    y_train,
    k=20,
    use_inffs=True,
    use_ugfs=True,
    inffs_params=None,
    ugfs_params=None,
):
    """
    Chạy các phương pháp graph-based FS (InfFS, UGFS) trên X_train, y_train.

    Parameters
    ----------
    X_train : array, shape (n_samples, n_features)
    y_train : array, shape (n_samples,)
    k : int
        Số feature tối đa để tạo subset top-k (analog với run_all_filters).
    use_inffs, use_ugfs : bool
        Bật/tắt từng phương pháp.
    inffs_params : dict or None
        Tham số override cho inffs_rank (alpha, supervision, verbose, standardize).
    ugfs_params : dict or None
        Tham số override cho ugfs_rank (ndim, nbdk, varthr, preprocess, standardize).

    Returns
    -------
    scores_dict : dict[name] -> score vector shape (n_features,)
    selected_dict : dict[name] -> np.ndarray chỉ số feature (0-based) top-k
    """
    X_train = np.asarray(X_train, float)
    y_train = np.asarray(y_train, int).ravel()
    n_features = X_train.shape[1]

    if k is None or k > n_features:
        k = n_features

    scores_dict = {}
    selected_dict = {}

    # --- InfFS ---
    if use_inffs:
        if inffs_params is None:
            inffs_params = {}
        scores_inffs, ranking_inffs, _ = inffs_rank(
            X_train,
            y_train,
            **inffs_params,
        )
        topk_inffs = ranking_inffs[:k]

        scores_dict["InfFS"] = scores_inffs
        selected_dict["InfFS"] = topk_inffs

    # --- UGFS ---
    if use_ugfs:
        if ugfs_params is None:
            ugfs_params = {}
        scores_ugfs, ranking_ugfs, _ = ugfs_rank(
            X_train,
            y_train,
            **ugfs_params,
        )
        topk_ugfs = ranking_ugfs[:k]

        scores_dict["UGFS"] = scores_ugfs
        selected_dict["UGFS"] = topk_ugfs

    return scores_dict, selected_dict


# =====================================================
# 4. Quick self-test (optional)
# =====================================================
if __name__ == "__main__":
    # Ví dụ test nhanh với myinputdatasetXD nếu bạn muốn
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
        inffs_params={"alpha": 0.5, "supervision": 1, "verbose": 0, "standardize": True},
        ugfs_params={"ndim": None, "nbdk": 5, "varthr": 2.0, "preprocess": "cscale", "standardize": True},
    )

    for name in scores_dict:
        print(f"{name}: scores shape = {scores_dict[name].shape}, "
              f"top-k idx shape = {selected_dict[name].shape}")
