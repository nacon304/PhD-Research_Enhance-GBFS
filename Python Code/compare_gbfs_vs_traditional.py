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
from sklearn.model_selection import StratifiedShuffleSplit

from traditional_fs import (
    run_all_filters,
    run_embedded_methods,
    run_wrapper_methods,
)

from graph_fs import (
    run_graph_fs_methods,
)

# =====================================================
# 0. Helper chung
# =====================================================

def _mapminmax_zero_one(X):
    """
    Regularize each column to [0, 1].
    """
    X = np.asarray(X, dtype=float)
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)

    denom = X_max - X_min
    denom[denom == 0] = 1.0
    return (X - X_min) / denom


def _hv_max2d(xs, ys):
    """
    Hypervolume cho 2D maximization với reference = (0,0),
    mỗi điểm là (x, y) >= 0, rectangle từ (0,0) đến (x,y).
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    pts = np.column_stack([xs, ys])

    keep = np.ones(len(pts), bool)
    for i in range(len(pts)):
        if not keep[i]:
            continue
        for j in range(len(pts)):
            if j == i or not keep[j]:
                continue
            if (pts[j, 0] >= pts[i, 0] and pts[j, 1] >= pts[i, 1] and
                (pts[j, 0] > pts[i, 0] or pts[j, 1] > pts[i, 1])):
                keep[i] = False
                break

    pts = pts[keep]
    if pts.size == 0:
        return 0.0

    order = np.argsort(-pts[:, 0])
    pts = pts[order]

    hv = 0.0
    max_y = 0.0
    for x, y in pts:
        if y > max_y:
            hv += x * (y - max_y)
            max_y = y
    return hv

def hypervolume_min2d(fr, er, ref=(1.0, 1.0)):
    """
    Hypervolume for minimize 2D (fRatio, eRate).
    - fr: array fRatio
    - er: array eRate
    - ref: reference point (fRef, eRef), (1, 1) by default.

    Idea:
      - Convert to maximization: x' = ref_x - fr, y' = ref_y - er
      - Then use _hv_max2d.
    """
    fr = np.asarray(fr, float)
    er = np.asarray(er, float)

    xs = np.clip(ref[0] - fr, 0, None)
    ys = np.clip(ref[1] - er, 0, None)

    return _hv_max2d(xs, ys)

def filter_nondominated_min(fr, er):
    """
    Keep only non-dominated points for 2D minimization:
    objectives = (fr, er).

    Returns:
        fr_nd, er_nd  (both sorted by fr ascending)
    """
    fr = np.asarray(fr, float)
    er = np.asarray(er, float)

    pts = np.column_stack([fr, er])
    n = len(pts)
    if n == 0:
        return fr, er

    keep = np.ones(n, dtype=bool)

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            # j dominates i in minimization?
            if (pts[j, 0] <= pts[i, 0] and pts[j, 1] <= pts[i, 1] and
                (pts[j, 0] < pts[i, 0] or pts[j, 1] < pts[i, 1])):
                keep[i] = False
                break

    pts_nd = pts[keep]
    if pts_nd.size == 0:
        return np.array([], float), np.array([], float)

    # sort by fr ascending for nicer plotting
    order = np.argsort(pts_nd[:, 0])
    pts_nd = pts_nd[order]

    return pts_nd[:, 0], pts_nd[:, 1]

# =====================================================
# 1. GBFS on one split + build Pareto front (fRatio, eRate)
# =====================================================

def run_gbfs_on_split(
    X_raw,
    y,
    tr_mask,
    delt,
    omega,
    kNeigh=5,
    pop=20,
    times=50,
):
    """
    Run GBFS on ONE fixed train/test split (70/30).

    Parameters
    ----------
    X_raw : np.ndarray, shape (n_samples, n_features)
    y     : np.ndarray, shape (n_samples,)
    tr_mask : bool array, shape (n_samples,)
        True = train, False = test
    delt, omega : parameters for GBFS (DELT, OMEGA)
    kNeigh : number of kNN graph features
    pop, times : parameters for newtry_ms

    Returns
    -------
    result_best : dict
      {
        "acc": float,
        "fnum": int,
        "fset": np.ndarray,
        "time": float,
        "assiNum": float,
      }

    gbfs_front : dict
      {
        "fRatio": np.ndarray,
        "eRate": np.ndarray
      }
      with each element corresponding to a solution on the Pareto front
      (evaluated by KNN on the same split).
    """
    tic = perf_counter()

    # 1) Normalize all data to [0,1]
    zData = _mapminmax_zero_one(X_raw)
    y = np.asarray(y).ravel().astype(int)

    # 2) Set globals for newtry_ms
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

    # 3) Fisher score on train set -> vWeight
    _, vWeight0 = fisherScore(GG.trData, GG.trLabel)
    GG.vWeight = 1.0 + _mapminmax_zero_one(vWeight0.reshape(-1, 1)).ravel()
    GG.vWeight1 = GG.vWeight

    # 4) Build similarity graph based on correlation
    adj = 1.0 - pdist(GG.trData.T, metric="correlation")
    adj = np.nan_to_num(adj, nan=0.0)
    adj = np.abs(adj)
    GG.Weight = squareform(adj)
    GG.Zout = squareform(adj)

    # 5) Keep only k nearest neighbors for each feature
    GG.kNeiMatrix = np.zeros((GG.featNum, GG.kNeigh), dtype=int)
    kNeiZoutMode = np.zeros_like(GG.Zout)
    for i in range(GG.Zout.shape[0]):
        idx_sorted = np.argsort(-GG.Zout[i, :])
        idx_topk = idx_sorted[:GG.kNeigh]
        GG.kNeiMatrix[i, :] = idx_topk
        kNeiZoutMode[i, idx_topk] = 1

    GG.kNeiZout = GG.Zout * (kNeiZoutMode != 0)
    kNeiAdj = squareform(GG.kNeiZout, force='tovector', checks=False)

    # 6) Run newtry_ms algorithm (GBFS)
    #    newtry_ms must return: featidx_best, pareto_masks, pareto_objs
    out = newtry_ms(kNeiAdj, pop, times)

    pareto_masks = None
    if isinstance(out, tuple) and len(out) >= 2:
        featidx_best = np.asarray(out[0])
        pareto_masks = np.asarray(out[1])
        pareto_objs = np.asarray(out[2]) if len(out) >=3 else None
    else:
        featidx_best = np.asarray(out)
    
    # pareto_indices = {}
    # if pareto_objs is not None and pareto_objs.size > 0:
    #     for i in range(pareto_masks.shape[0]):
    #         mask = pareto_masks[i]
    #         obj = pareto_objs[i]

    #         frate = obj[1]
    #         erate = obj[0]
    #         frate_2 = sum(mask != 0)

    #         if frate_2 != frate:
    #             continue

    #         if pareto_indices.get(frate) is None or erate < pareto_indices[frate][1]:
    #             pareto_indices[frate] = (mask, erate)
    #     pareto_masks = []
    #     pareto_objs = []
    #     for frate, (mask, erate) in pareto_indices.items():
    #         pareto_masks.append(mask)
    #         pareto_objs.append([erate, frate])
    #     pareto_masks = np.asarray(pareto_masks)
    #     pareto_objs = np.asarray(pareto_objs)

    selected_features = np.where(featidx_best != 0)[0]
    selected_num = selected_features.size

    # 7) Evaluate "best" solution by KNN on the same split
    if selected_num == 0:
        acc_best = 0.0
    else:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(GG.trData[:, selected_features], GG.trLabel)
        predLabel = knn.predict(GG.teData[:, selected_features])
        acc_best = np.mean(predLabel == GG.teLabel)

    runtime = perf_counter() - tic
    assiNum = float(np.sum(getattr(GG, "assiNumInside", [])))

    result_best = {
        "acc": float(acc_best),
        "fnum": int(selected_num),
        "fset": selected_features,
        "time": float(runtime),
        "assiNum": assiNum,
    }

    # 8) Build GBFS front from all Pareto solutions (if any)
    fRatios_gb = []
    eRates_gb = []

    if pareto_masks is not None and pareto_masks.size > 0:
        pareto_masks = np.asarray(pareto_masks)
        if pareto_masks.ndim == 1:
            pareto_masks = pareto_masks.reshape(1, -1)
        
        for mask in pareto_masks:
            idx = np.where(mask != 0)[0]
            k = idx.size
            if k == 0:
                continue

            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(GG.trData[:, idx], GG.trLabel)
            pred = knn.predict(GG.teData[:, idx])
            acc = np.mean(pred == GG.teLabel)

            fRatios_gb.append(k / GG.featNum)
            eRates_gb.append(1.0 - acc)

    # remove pareto solution has same feature ratio but worse error rate
    gbfs_indx = {}
    if len(fRatios_gb) > 0:
        for fr, er in zip(fRatios_gb, eRates_gb):
            if (fr not in gbfs_indx) or (er < gbfs_indx[fr]):
                gbfs_indx[fr] = er
        fr_list = []
        er_list = []
        for fr, er in gbfs_indx.items():
            fr_list.append(fr)
            er_list.append(er)

    sort_idx = np.argsort(fr_list)
    gbfs_front = {
        "fRatio": np.asarray(np.array(fr_list)[sort_idx], float),
        "eRate": np.asarray(np.array(er_list)[sort_idx], float),
    }
    # gbfs_front = {
    #     "fRatio": np.asarray(np.array(fRatios_gb)[sort_idx], float),
    #     "eRate": np.asarray(np.array(eRates_gb)[sort_idx], float),
    # }
    print(gbfs_front)

    return result_best, gbfs_front


# =====================================================
# 2. Traditional FS: build curve (fRatio, eRate) based on RATIOS
# =====================================================

RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def build_fronts_from_scores(scores_dict, X_train, y_train, X_test, y_test):
    """
    scores_dict: dict[name] -> score vector (n_features,)
    Return:
      fronts: dict[name] -> (fRatio_array, eRate_array)
    """
    n_features = X_train.shape[1]
    fronts = {}

    for name, s in scores_dict.items():
        s = np.asarray(s).ravel()
        # Handle NaN if any
        s = np.nan_to_num(s, nan=0.0)

        # Ranking in descending order by |score|
        ranking = np.argsort(-np.abs(s))

        fRatios = []
        eRates = []

        for r in RATIOS:
            k = int(round(r * n_features))
            k = max(1, min(k, n_features))
            idx = ranking[:k]

            Xtr = X_train[:, idx]
            Xte = X_test[:, idx]

            clf = KNeighborsClassifier(n_neighbors=5)
            clf.fit(Xtr, y_train)
            y_pred = clf.predict(Xte)
            acc = accuracy_score(y_test, y_pred)

            fRatios.append(k / n_features)
            eRates.append(1.0 - acc)

        fronts[name] = (np.asarray(fRatios), np.asarray(eRates))

    return fronts

# =====================================================
# 3. Compare fronts: Traditional curves vs GBFS Pareto (1 split)
# =====================================================

def build_fronts_from_wrapper_methods(X_train, X_test, y_train, y_test, cv=5):
    fronts = {}
    for ratio in RATIOS:
        k = int(round(ratio * X_train.shape[1]))
        k = max(1, min(k, X_train.shape[1]))
        selected = run_wrapper_methods(X_train, y_train, k=k, cv=cv)
        for name, idx in selected.items():
            Xtr = X_train[:, idx]
            Xte = X_test[:, idx]

            clf = KNeighborsClassifier(n_neighbors=5)
            clf.fit(Xtr, y_train)
            y_pred = clf.predict(Xte)
            acc = accuracy_score(y_test, y_pred)

            if name not in fronts:
                fronts[name] = ([], [])

            fronts[name][0].append(len(idx) / X_train.shape[1])  # fRatio
            fronts[name][1].append(1.0 - acc)                        # eRate
    return fronts

def compare_fronts_median_hv(
    data_index=1,
    delt=10.0,
    omega=0.8,
    kNeigh=5,
    runs=5,
    hv_ref=(1.0, 1.0),
):
    # --- Load data ---
    dataset, labels, datasetName = myinputdatasetXD(data_index)
    X_raw = dataset[:, 1:].astype(float)
    y = np.asarray(labels, dtype=int).ravel()

    n_samples, n_features = X_raw.shape
    print(f"Dataset: {datasetName}")
    print(f"X shape = {X_raw.shape}, y shape = {y.shape}")

    # --- fixed 70/30 split for all runs ---
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.3,
        random_state=42
    )
    for train_idx, test_idx in sss.split(X_raw, y):
        tr_mask = np.zeros(n_samples, dtype=bool)
        tr_mask[train_idx] = True

    X_train_raw = X_raw[tr_mask, :]
    X_test_raw  = X_raw[~tr_mask, :]
    y_train     = y[tr_mask]
    y_test      = y[~tr_mask]

    # --- Baseline ALL features ---
    scaler_all = StandardScaler()
    Xtr_all = scaler_all.fit_transform(X_train_raw)
    Xte_all = scaler_all.transform(X_test_raw)

    clf_all = KNeighborsClassifier(n_neighbors=5)
    clf_all.fit(Xtr_all, y_train)
    y_pred_all = clf_all.predict(Xte_all)
    acc_all = accuracy_score(y_test, y_pred_all)
    print(f"[BASELINE] ALL features acc = {acc_all:.4f}")

    # Prepare structures to store fronts per RUN
    trad_fronts_runs = []   # list[ dict[name] -> (fr, er) ]
    gb_fronts_runs   = []   # list[ dict("fRatio","eRate") ]

    # For HV + acc best
    hv_trad = {}            # name -> list[HV per run]
    hv_gb   = []            # list[HV per run]
    acc_trad = {}           # name -> list[best acc per run]
    fratio_best_trad = {}   # name -> list[best fRatio per run]
    gb_accs = []            # list[best acc per run for GBFS]
    gb_fratios_best = []    # list[best fRatio per run for GBFS]

    # =============================
    # Multi-run
    # =============================
    f_scores, _ = run_all_filters(Xtr_all, y_train, k=n_features)
    g_scores, _ = run_graph_fs_methods(
        X_train_raw, y_train,
        k=n_features,
        use_inffs=True,
        use_ugfs=True,
    )
    for r in range(runs):
        print(f"\n========== FRONT RUN {r+1}/{runs} ==========")

        # 1) Traditional FS: get scores + build fronts
        e_scores, _ = run_embedded_methods(Xtr_all, y_train, k=n_features)
        # w_fronts = build_fronts_from_wrapper_methods(
        #     Xtr_all, Xte_all, y_train, y_test, cv=3
        # )

        all_scores = {}
        all_scores.update(f_scores)
        all_scores.update(e_scores)
        all_scores.update(g_scores)

        trad_fronts_r = build_fronts_from_scores(
            all_scores, Xtr_all, y_train, Xte_all, y_test
        )
        # trad_fronts_r.update(w_fronts)
        trad_fronts_runs.append(trad_fronts_r)

        # 2) GBFS on the same split
        gb_best_r, gb_front_r = run_gbfs_on_split(
            X_raw, y, tr_mask,
            delt=delt,
            omega=omega,
            kNeigh=kNeigh,
            pop=20,
            times=50,
        )

        print(f"[GBFS run {r+1}] acc = {gb_best_r['acc']:.4f}, "
              f"#feat = {gb_best_r['fnum']}")

        gb_fronts_runs.append(gb_front_r)

        # ----- CALCULATE HV + BEST ACC FOR THIS RUN -----

        # Traditional methods
        for name, (fr, er) in trad_fronts_r.items():
            fr = np.asarray(fr, float)
            er = np.asarray(er, float)
            # Hypervolume
            hv_val = hypervolume_min2d(fr, er, ref=hv_ref)
            hv_trad.setdefault(name, []).append(hv_val)

            # Best acc = 1 - min(eRate)
            if er.size > 0:
                idx_best = int(np.argmin(er))
                best_e = float(er[idx_best])
                best_acc = 1.0 - best_e
                best_fr = float(fr[idx_best])
            else:
                best_acc = 0.0
                best_fr = 0.0

            acc_trad.setdefault(name, []).append(best_acc)
            fratio_best_trad.setdefault(name, []).append(best_fr)

        # GBFS
        fr_gb_r = gb_front_r["fRatio"]
        er_gb_r = gb_front_r["eRate"]
        if fr_gb_r.size > 0:
            hv_val_gb = hypervolume_min2d(fr_gb_r, er_gb_r, ref=hv_ref)
            best_acc_gb = float(gb_best_r["acc"])
            best_fr_gb = float(gb_best_r["fnum"] / n_features)
        else:
            hv_val_gb = 0.0
            best_acc_gb = 0.0
            best_fr_gb = 0.0

        hv_gb.append(hv_val_gb)
        gb_accs.append(best_acc_gb)
        gb_fratios_best.append(best_fr_gb)

    # Convert lists to np.array
    for name in hv_trad:
        hv_trad[name] = np.asarray(hv_trad[name], float)
        acc_trad[name] = np.asarray(acc_trad[name], float)
        fratio_best_trad[name] = np.asarray(fratio_best_trad[name], float)

    hv_gb = np.asarray(hv_gb, float)
    gb_accs = np.asarray(gb_accs, float)
    gb_fratios_best = np.asarray(gb_fratios_best, float)

    # =============================
    # Select median HV front for each method
    # =============================
    median_trad_fronts = {}

    for name, hv_arr in hv_trad.items():
        median_val = np.median(hv_arr)
        idx = int(np.argsort(np.abs(hv_arr - median_val))[0])
        
        # get the original front for the chosen run
        fr, er = trad_fronts_runs[idx][name]

        # keep only non-dominated points (minimization)
        fr_nd, er_nd = filter_nondominated_min(fr, er)
        median_trad_fronts[name] = (fr_nd, er_nd)

        print(f"[TRAD {name}] median HV = {median_val:.6f}, "
            f"choose run {idx+1} (HV = {hv_arr[idx]:.6f})")

    # GBFS
    if hv_gb.size > 0:
        median_gb = np.median(hv_gb)
        idx_gb = int(np.argsort(np.abs(hv_gb - median_gb))[0])
        median_gb_front = gb_fronts_runs[idx_gb]

        # filter non-dominated points for GBFS front
        fr_gb = median_gb_front["fRatio"]
        er_gb = median_gb_front["eRate"]
        fr_gb_nd, er_gb_nd = filter_nondominated_min(fr_gb, er_gb)
        median_gb_front = {
            "fRatio": fr_gb_nd,
            "eRate": er_gb_nd,
        }

        print(f"[GBFS] median HV = {median_gb:.6f}, "
              f"choose run {idx_gb+1} (HV = {hv_gb[idx_gb]:.6f})")
    else:
        median_gb_front = {"fRatio": np.array([]), "eRate": np.array([])}
        idx_gb = -1
        median_gb = 0.0

    # =============================
    # SUMMARY mean ± std ACC (with best point on front)
    # =============================
    summary = []

    # Traditional
    for name in sorted(hv_trad.keys()):
        acc_arr = acc_trad[name]
        fr_arr  = fratio_best_trad[name]

        mean_acc = acc_arr.mean()
        std_acc  = acc_arr.std()
        mean_fRatio = fr_arr.mean()
        mean_fnum   = mean_fRatio * n_features

        eRate_mean = 1.0 - mean_acc

        summary.append({
            "method": name,
            "mean_acc": float(mean_acc),
            "std_acc": float(std_acc),
            "num_features": float(mean_fnum),
            "fRatio": float(mean_fRatio),
            "eRate": float(eRate_mean),
        })

    # GBFS
    if gb_accs.size > 0:
        mean_acc_gb = gb_accs.mean()
        std_acc_gb  = gb_accs.std()
        mean_fRatio_gb = gb_fratios_best.mean()
        mean_fnum_gb   = mean_fRatio_gb * n_features
    else:
        mean_acc_gb = std_acc_gb = mean_fRatio_gb = mean_fnum_gb = 0.0

    summary.append({
        "method": "GBFS",
        "mean_acc": float(mean_acc_gb),
        "std_acc": float(std_acc_gb),
        "num_features": float(mean_fnum_gb),
        "fRatio": float(mean_fRatio_gb),
        "eRate": float(1.0 - mean_acc_gb),
    })

    summary = sorted(summary, key=lambda d: -d["mean_acc"])

    print("\n===== SUMMARY (best acc per run, mean ± std, #feat) =====")
    for s in summary:
        print(f"{s['method']:18s}: "
              f"acc = {s['mean_acc']:.4f} ± {s['std_acc']:.4f}, "
              f"#feat ≈ {s['num_features']:.1f} "
              f"(fRatio = {s['fRatio']:.3f}, eRate = {s['eRate']:.3f})")

    # =============================
    # Plot: use median HV front of EACH method
    # =============================
    plt.figure(figsize=(6, 5))

    # Traditional: plot median curves
    for name, (fr, er) in median_trad_fronts.items():
        plt.plot(fr, er, marker="o", linewidth=1, markersize=3, label=name)

    # GBFS: plot median HV front, connect into a line
    fr_gb_plot = median_gb_front["fRatio"]
    er_gb_plot = median_gb_front["eRate"]
    if fr_gb_plot.size > 0:
        order = np.argsort(fr_gb_plot)
        plt.plot(
            fr_gb_plot[order],
            er_gb_plot[order],
            marker="*",
            linestyle="-",
            linewidth=1.5,
            markersize=6,
            label=f"GBFS Pareto (median HV)",
        )

    # Baseline ALL features (1 point)
    plt.scatter(
        [1.0], [1.0 - acc_all],
        s=60, marker="s", label="ALL features"
    )

    plt.xlabel("fRatio = #selected features / #total features")
    plt.ylabel("eRate = 1 - accuracy (KNN)")
    plt.title(
        f"Median-HV fronts on {datasetName} "
        f"(single 70/30 split, {runs} runs per method)"
    )
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    return median_trad_fronts, median_gb_front, hv_trad, hv_gb, summary

# =====================================================
# 5. Main
# =====================================================

if __name__ == "__main__":
    median_trad_fronts, median_gb_front, hv_trad, hv_gb, summary = compare_fronts_median_hv(
        data_index=2,
        delt=10.0,
        omega=0.8,
        kNeigh=5,
        runs=5,
        hv_ref=(1.0, 1.0),
    )