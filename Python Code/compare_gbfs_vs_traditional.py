# compare_gbfs_vs_traditional.py

import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from myinputdatasetXD import myinputdatasetXD

# import lại các hàm / globals từ code GBFS và traditional
from fisherScore import fisherScore
from newtry_ms import newtry_ms
import gbfs_globals as GG

from traditional_fs import (
    run_all_filters,
    run_embedded_methods,
    run_wrapper_methods,
    evaluate_with_knn_holdout,
)


def _mapminmax_zero_one(X):
    """
    Regularize each column to [0, 1].
    (copy từ code GBFS chính để dùng lại)
    """
    X = np.asarray(X, dtype=float)
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)

    denom = X_max - X_min
    denom[denom == 0] = 1.0
    return (X - X_min) / denom


def run_gbfs_on_split(
    X_raw,
    y,
    tr_mask,
    delt,
    omega,
    kNeigh=5,
    max_feat=20,
):
    """
    Chạy GBFS trên MỘT split train/test cố định (70/30).

    Parameters
    ----------
    X_raw : np.ndarray, shape (n_samples, n_features)
    y     : np.ndarray, shape (n_samples,)
    tr_mask : bool array, shape (n_samples,)
        True = train, False = test
    delt, omega : tham số GBFS (DELT, OMEGA)
    kNeigh : số kNN trên graph feature
    max_feat : số feature tối đa yêu cầu (tham số N trong newtry_ms, nếu bạn muốn chỉnh thêm)

    Returns
    -------
    result : dict
      {
        "acc": float,
        "fnum": int,
        "fset": np.ndarray,
        "time": float,
        "assiNum": float,
      }
    """
    tic = perf_counter()

    # 1) Chuẩn hóa [0,1] toàn bộ dữ liệu (giống code cũ)
    zData = _mapminmax_zero_one(X_raw)
    y = np.asarray(y).ravel().astype(int)

    # 2) Gán globals để newtry_ms dùng
    GG.DELT = delt
    GG.OMEGA = omega
    GG.data = zData
    GG.label = y
    GG.featNum = zData.shape[1]
    GG.kNeigh = kNeigh

    GG.trIdx = tr_mask
    GG.trData = zData[tr_mask, :]
    GG.trLabel = y[tr_mask]
    GG.teData = zData[~tr_mask, :]
    GG.teLabel = y[~tr_mask]

    GG.assiNumInside = []

    # 3) Fisher score trên tập train -> vWeight
    _, vWeight0 = fisherScore(GG.trData, GG.trLabel)
    GG.vWeight = 1.0 + _mapminmax_zero_one(vWeight0.reshape(-1, 1)).ravel()
    GG.vWeight1 = GG.vWeight

    # 4) Xây similarity graph dựa trên correlation (giống code chính)
    adj = 1.0 - pdist(GG.trData.T, metric="correlation")
    adj = np.nan_to_num(adj, nan=0.0)
    adj = np.abs(adj)
    GG.Weight = squareform(adj)
    GG.Zout = squareform(adj)

    # 5) Chỉ giữ k láng giềng gần nhất cho mỗi feature
    GG.kNeiMatrix = np.zeros((GG.featNum, GG.kNeigh), dtype=int)
    kNeiZoutMode = np.zeros_like(GG.Zout)
    for i in range(GG.Zout.shape[0]):
        idx_sorted = np.argsort(-GG.Zout[i, :])
        idx_topk = idx_sorted[:GG.kNeigh]
        GG.kNeiMatrix[i, :] = idx_topk
        kNeiZoutMode[i, idx_topk] = 1

    GG.kNeiZout = GG.Zout * (kNeiZoutMode != 0)
    kNeiAdj = squareform(GG.kNeiZout, force='tovector', checks=False)

    # 6) Chạy thuật toán newtry_ms (GBFS)
    #    20 = số cá thể, 50 = số generation → bạn có thể đưa thành tham số nếu muốn
    featIdx = np.asarray(newtry_ms(kNeiAdj, 20, 50))
    selected_features = np.where(featIdx != 0)[0]
    selected_num = selected_features.size

    # 7) Đánh giá bằng KNN trên cùng split
    if selected_num == 0:
        acc = 0.0
    else:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(GG.trData[:, selected_features], GG.trLabel)
        predLabel = knn.predict(GG.teData[:, selected_features])
        acc = np.mean(predLabel == GG.teLabel)

    runtime = perf_counter() - tic
    assiNum = float(np.sum(getattr(GG, "assiNumInside", [])))

    return {
        "acc": float(acc),
        "fnum": int(selected_num),
        "fset": selected_features,
        "time": float(runtime),
        "assiNum": assiNum,
    }

def generate_splits(n_samples, runs=20, train_ratio=0.7, random_state=42):
    """
    Sinh danh sách bool mask cho RUNS lần random 70/30.
    Dùng chung cho GBFS + tất cả traditional FS.
    """
    rng = np.random.default_rng(random_state)
    splits = []
    n_train = int(round(train_ratio * n_samples))

    for _ in range(runs):
        perm = rng.permutation(n_samples)
        train_idx = perm[:n_train]
        tr_mask = np.zeros(n_samples, dtype=bool)
        tr_mask[train_idx] = True
        splits.append(tr_mask)

    return splits

def compare_gbfs_vs_traditional(
    data_index=2,
    runs=20,
    delt=10.0,
    omega=0.8,
    kNeigh=5,
    max_k_fs=None,
):
    # --- Load dữ liệu ---
    dataset, labels, datasetName = myinputdatasetXD(data_index)
    X_raw = dataset[:, 1:].astype(float)
    y = np.asarray(labels, dtype=int).ravel()

    n_samples, n_features = X_raw.shape
    print(f"Dataset: {datasetName}")
    print(f"X shape = {X_raw.shape}, y shape = {y.shape}")

    if max_k_fs is None:
        max_k_fs = min(20, max(1, n_features // 2))  # tránh > #features

    # --- Sinh splits 70/30 chung ---
    splits = generate_splits(n_samples, runs=runs, train_ratio=0.7, random_state=42)

    # Lưu kết quả
    accs_traditional = {}   # name -> list[acc]
    gbfs_accs = []
    gbfs_fnums = []

    for r, tr_mask in enumerate(splits, start=1):
        print(f"\n========== RUN {r}/{runs} ==========")

        X_train_raw = X_raw[tr_mask, :]
        X_test_raw  = X_raw[~tr_mask, :]
        y_train     = y[tr_mask]
        y_test      = y[~tr_mask]

        # --- Baseline KNN với ALL features trên split này ---
        scaler_all = StandardScaler()
        X_train_all = scaler_all.fit_transform(X_train_raw)
        X_test_all  = scaler_all.transform(X_test_raw)

        clf_all = KNeighborsClassifier(n_neighbors=5)
        clf_all.fit(X_train_all, y_train)
        y_pred_all = clf_all.predict(X_test_all)
        acc_all = accuracy_score(y_test, y_pred_all)
        print(f"[RUN {r}] Baseline ALL features acc = {acc_all:.4f}")

        # =============================
        # 1) Traditional Feature Selection
        # =============================
        # FILTERS
        f_scores, f_selected = run_all_filters(X_train_raw, y_train, k=max_k_fs)

        # EMBEDDED
        e_scores, e_selected = run_embedded_methods(X_train_raw, y_train, k=max_k_fs)

        # WRAPPERS (cẩn thận runtime nếu dataset rất lớn)
        w_selected = run_wrapper_methods(X_train_raw, y_train, k=max_k_fs, cv=5)

        # Gộp tất cả subset
        all_selected = {}
        all_selected.update(f_selected)
        all_selected.update(e_selected)
        all_selected.update(w_selected)
        all_selected["ALL_features"] = np.arange(n_features)

        # Đánh giá trên cùng split
        accs = evaluate_with_knn_holdout(
            X_train_raw, y_train,
            X_test_raw,  y_test,
            all_selected,
            n_neighbors=5,
        )
        # ALL_features dùng baseline acc_all để đảm bảo thống nhất
        accs["ALL_features"] = acc_all

        # Lưu acc theo từng phương pháp
        for name, acc in accs.items():
            accs_traditional.setdefault(name, []).append(acc)

        # =============================
        # 2) GBFS trên cùng split
        # =============================
        gb_result = run_gbfs_on_split(
            X_raw, y, tr_mask,
            delt=delt,
            omega=omega,
            kNeigh=kNeigh,
            max_feat=max_k_fs,
        )
        gbfs_accs.append(gb_result["acc"])
        gbfs_fnums.append(gb_result["fnum"])

        print(f"[RUN {r}] GBFS acc = {gb_result['acc']:.4f}, "
              f"#feat = {gb_result['fnum']}")

    # =============================
    # Tính mean / std cho từng phương pháp
    # =============================
    summary = []

    # Traditional
    for name, arr in accs_traditional.items():
        arr = np.asarray(arr, dtype=float)
        mean_acc = arr.mean()
        std_acc  = arr.std()

        if name == "ALL_features":
            fnum = n_features
        else:
            # Trong code trên, k cố định = max_k_fs cho tất cả FS
            fnum = max_k_fs

        fRatio = fnum / n_features
        eRate  = 1.0 - mean_acc

        summary.append({
            "method": name,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "num_features": fnum,
            "fRatio": fRatio,
            "eRate": eRate,
        })

    # GBFS
    gbfs_accs = np.asarray(gbfs_accs, dtype=float)
    gbfs_fnums = np.asarray(gbfs_fnums, dtype=float)

    if gbfs_fnums.size == 0:
        gb_fmean = 0.0
    else:
        gb_fmean = gbfs_fnums.mean()

    gb_summary = {
        "method": "GBFS",
        "mean_acc": gbfs_accs.mean() if gbfs_accs.size > 0 else 0.0,
        "std_acc": gbfs_accs.std() if gbfs_accs.size > 0 else 0.0,
        "num_features": gb_fmean,
        "fRatio": gb_fmean / n_features if n_features > 0 else 0.0,
        "eRate": 1.0 - (gbfs_accs.mean() if gbfs_accs.size > 0 else 0.0),
    }
    summary.append(gb_summary)

    # Sắp xếp cho đẹp khi vẽ (theo mean_acc giảm dần)
    summary = sorted(summary, key=lambda d: -d["mean_acc"])

    print("\n===== SUMMARY (mean ± std acc, #feat) =====")
    for s in summary:
        print(f"{s['method']:18s}: "
              f"acc = {s['mean_acc']:.4f} ± {s['std_acc']:.4f}, "
              f"#feat ≈ {s['num_features']:.1f} "
              f"(fRatio = {s['fRatio']:.3f}, eRate = {s['eRate']:.3f})")

    # =============================
    # Vẽ hình minh họa
    # =============================
    methods = [s["method"] for s in summary]
    mean_acc = [s["mean_acc"] for s in summary]
    std_acc  = [s["std_acc"]  for s in summary]
    fRatios  = [s["fRatio"]   for s in summary]
    eRates   = [s["eRate"]    for s in summary]

    # ---- Hình 1: Bar chart accuracy ----
    plt.figure(figsize=(8, 4))
    x = np.arange(len(methods))
    plt.bar(x, mean_acc, yerr=std_acc, capsize=3)
    plt.xticks(x, methods, rotation=45, ha="right")
    plt.ylabel("Test accuracy (KNN)")
    plt.title(f"Accuracy comparison on {datasetName} (70/30, {runs} runs)")
    plt.tight_layout()

    # ---- Hình 2: Bar chart fRatio (#feature / tổng) ----
    plt.figure(figsize=(8, 4))
    plt.bar(x, fRatios)
    plt.xticks(x, methods, rotation=45, ha="right")
    plt.ylabel("fRatio = #selected features / #total features")
    plt.title(f"Feature ratio comparison on {datasetName}")
    plt.tight_layout()

    # ---- Hình 3: Scatter fRatio vs eRate (giống front đơn giản) ----
    plt.figure(figsize=(5, 4))
    for s in summary:
        m = s["method"]
        fr = s["fRatio"]
        er = s["eRate"]
        if m == "GBFS":
            plt.scatter(fr, er, s=80, marker="*", label=m)   # highlight GBFS
        else:
            plt.scatter(fr, er, s=40, marker="o", label=m)

    # Tránh duplicate legend label
    handles, labels = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    plt.legend(uniq.values(), uniq.keys(), fontsize=8)

    plt.xlabel("fRatio (smaller is sparser)")
    plt.ylabel("eRate = 1 - accuracy (smaller is better)")
    plt.title(f"fRatio vs eRate on {datasetName}")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()

    plt.show()

    return summary

if __name__ == "__main__":
    # Ví dụ: dataset Glass, index = 1
    summary = compare_gbfs_vs_traditional(
        data_index=1,
        runs=20,       # số lần lặp split 70/30
        delt=10.0,     # tinh chỉnh theo best setting của bạn
        omega=0.8,
        kNeigh=5,
        max_k_fs=None,  # nếu None → auto = min(20, n_feat/2)
    )
