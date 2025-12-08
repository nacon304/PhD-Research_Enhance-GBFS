import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

import gbfs_globals as GG
from kshell_2 import kshell_2


def evaluate_objective_f(M, individual_z):
    """
    Node-based evaluation with 2-phase k-shell (Redundancy + Complementary).

    Parameters
    ----------
    M : int
        Number of objectives (should be 2).
    individual_z : array-like, shape (featNum,)
        Binary chromosome on NODES:
        z_i = 1 nếu feature i thuộc S(z).

    Returns
    -------
    f : np.ndarray, shape (M,)
        Objective values [f1, f2], với:
          f1 = mean(1 - accuracy) over 5-fold CV trên tập F(z),
          f2 = |F(z)| / featNum.
    featIdx_full : np.ndarray, shape (featNum,)
        Boolean mask cho tập feature cuối F(z) trong không gian full.
    """
    # ========= Kiểm tra global =========
    if GG.data is None:
        raise ValueError("evaluate_objective_f: global 'data' chưa được set.")
    if GG.trData is None or GG.trLabel is None:
        raise ValueError(
            "evaluate_objective_f: 'trData' hoặc 'trLabel' chưa được set."
        )
    if GG.Zout is None:
        raise ValueError(
            "evaluate_objective_f: GG.Zout (redundancy graph) chưa được khởi tạo."
        )

    d = GG.data.shape[1]
    z = np.asarray(individual_z, dtype=int).ravel()

    if z.size != d:
        raise ValueError(
            f"evaluate_objective_f: kích thước chromosome z={z.size} "
            f"không khớp số feature d={d}."
        )

    # Nếu subset rỗng -> trả về worst
    if z.sum() == 0:
        f = np.array([1.0, 1.0], dtype=float)
        featIdx_full = np.zeros(d, dtype=bool)
        return f, featIdx_full

    # ========= Pha 1: K-shell trên đồ thị Redundancy =========
    W_red = np.asarray(GG.Zout, dtype=float)
    if GG.vWeight is not None:
        w_red = np.asarray(GG.vWeight, dtype=float).ravel()
    else:
        w_red = np.ones(d, dtype=float)

    # S(z) = các node được bật trong chromosome
    S = np.where(z != 0)[0]
    if S.size == 0:
        # không nên xảy ra do check ở trên, nhưng cẩn thận
        f = np.array([1.0, 1.0], dtype=float)
        featIdx_full = np.zeros(d, dtype=bool)
        return f, featIdx_full

    # core1 = tập node được chọn từ đồ thị redundancy
    core1 = kshell_2(W_red, nodes=S, node_weights=w_red)
    core1 = np.asarray(core1, dtype=int).ravel()

    # ========= Pha 2: K-shell trên đồ thị Complementary =========
    all_idx = np.arange(d, dtype=int)
    U = np.setdiff1d(all_idx, core1, assume_unique=True)

    core2 = np.array([], dtype=int)
    if GG.compMat is not None and U.size > 0:
        W_comp = np.asarray(GG.compMat, dtype=float)

        # base relevance: dùng GG.vWeight làm w0 (hoặc bạn có thể tách thêm fisherBase nếu muốn)
        w0 = w_red

        alpha = getattr(GG, "alphaComp", 0.5)

        if core1.size > 0:
            comp_to_core = W_comp[np.ix_(U, core1)]
            # max complementary đến core cho từng node trong U
            max_comp = comp_to_core.max(axis=1)
        else:
            max_comp = np.zeros(U.size, dtype=float)

        # w_comp_full: vector weight kích thước d, chỉ sửa tại U
        w_comp_full = w0.copy()
        w_comp_full[U] = alpha * w0[U] + (1.0 - alpha) * max_comp

        core2 = kshell_2(W_comp, nodes=U, node_weights=w_comp_full)
        core2 = np.asarray(core2, dtype=int).ravel()

    # ========= Gộp tập feature cuối F(z) =========
    F_idx = np.unique(np.concatenate([core1, core2]))
    F_idx = F_idx[(F_idx >= 0) & (F_idx < d)]

    # Nếu vì lý do gì đó vẫn rỗng -> fallback
    if F_idx.size == 0:
        f = np.array([1.0, 1.0], dtype=float)
        featIdx_full = np.zeros(d, dtype=bool)
        return f, featIdx_full

    # ========= Đánh giá 5-fold CV trên tập F(z) =========
    X = np.asarray(GG.trData, dtype=float)[:, F_idx]
    y = np.asarray(GG.trLabel).ravel()

    Nfold = 5
    kf = KFold(n_splits=Nfold, shuffle=False)
    fold_losses = []

    for train_index, test_index in kf.split(X):
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_test = X[test_index, :]
        y_test = y[test_index]

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        plabel = knn.predict(X_test)

        acc = np.mean(plabel == y_test)
        fold_losses.append(1.0 - acc)  # 1 - accuracy (để minimize)

    f1 = float(np.mean(fold_losses))
    f2 = float(F_idx.size / d)  # tỷ lệ số feature

    f = np.array([f1, f2], dtype=float)

    # ========= map sang mask full =========
    featIdx_full = np.zeros(d, dtype=bool)
    featIdx_full[F_idx] = True

    if f.size != M:
        raise ValueError(
            "The number of objective values does not match your input M. "
            "Kindly check evaluate_objective_f."
        )

    return f, featIdx_full
