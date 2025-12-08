import os
import shutil
import numpy as np
import pandas as pd
from time import perf_counter
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KNeighborsClassifier

from myinputdatasetXD import myinputdatasetXD
from fisherScore import fisherScore
from newtry_ms import newtry_ms
from helper import _mapminmax_zero_one, redundancy_rate_subset

import gbfs_globals as GG


# ====================== MI helper functions (discretization-based) ======================

def _discretize_feature(x, n_bins=10):
    """
    Discretize a continuous feature into n_bins using uniform bins.
    Returns:
        idx: integer bin indices in [0, n_bins-1]
        n_states: number of states actually used (here == n_bins unless degenerate)
    """
    x = np.asarray(x, dtype=float).ravel()
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max == x_min:
        # constant feature -> single state
        return np.zeros_like(x, dtype=int), 1

    # uniform bins
    bins = np.linspace(x_min, x_max, num=n_bins + 1, endpoint=True)
    idx = np.digitize(x, bins, right=False) - 1  # in [0, n_bins]
    idx[idx < 0] = 0
    idx[idx >= n_bins] = n_bins - 1
    return idx.astype(int), n_bins


def _mutual_information_1d(x, y, n_bins=10):
    """
    Estimate MI(X; Y) where X is continuous (1D) and Y is discrete labels.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    assert x.shape[0] == y.shape[0]

    x_idx, n_x = _discretize_feature(x, n_bins=n_bins)
    # map y to 0..Ny-1
    y_unique, y_idx = np.unique(y, return_inverse=True)
    n_y = y_unique.shape[0]

    # joint counts over X x Y
    joint_counts = np.bincount(x_idx * n_y + y_idx,
                               minlength=n_x * n_y).reshape(n_x, n_y)
    n_samples = float(x.shape[0])
    joint_prob = joint_counts / n_samples

    px = joint_prob.sum(axis=1, keepdims=True)  # shape (n_x, 1)
    py = joint_prob.sum(axis=0, keepdims=True)  # shape (1, n_y)

    # avoid log(0) by masking zeros
    nz = joint_prob > 0
    # broadcast px, py
    px_b = np.broadcast_to(px, joint_prob.shape)
    py_b = np.broadcast_to(py, joint_prob.shape)

    mi = np.sum(
        joint_prob[nz] * (
            np.log(joint_prob[nz]) - np.log(px_b[nz]) - np.log(py_b[nz])
        )
    )
    return mi


def _mutual_information_pair(x1, x2, y, n_bins=10):
    """
    Estimate MI([X1, X2]; Y) by discretizing (X1, X2) jointly.
    """
    x1 = np.asarray(x1).ravel()
    x2 = np.asarray(x2).ravel()
    y = np.asarray(y).ravel()
    assert x1.shape[0] == x2.shape[0] == y.shape[0]

    x1_idx, n_x1 = _discretize_feature(x1, n_bins=n_bins)
    x2_idx, n_x2 = _discretize_feature(x2, n_bins=n_bins)
    # combine into joint X state index
    joint_x_idx = x1_idx * n_x2 + x2_idx
    n_x = n_x1 * n_x2

    # map y to 0..Ny-1
    y_unique, y_idx = np.unique(y, return_inverse=True)
    n_y = y_unique.shape[0]

    # joint counts over (X1,X2) x Y
    joint_counts = np.bincount(joint_x_idx * n_y + y_idx,
                               minlength=n_x * n_y).reshape(n_x, n_y)
    n_samples = float(x1.shape[0])
    joint_prob = joint_counts / n_samples

    px = joint_prob.sum(axis=1, keepdims=True)  # shape (n_x, 1)
    py = joint_prob.sum(axis=0, keepdims=True)  # shape (1, n_y)

    nz = joint_prob > 0
    px_b = np.broadcast_to(px, joint_prob.shape)
    py_b = np.broadcast_to(py, joint_prob.shape)

    mi = np.sum(
        joint_prob[nz] * (
            np.log(joint_prob[nz]) - np.log(px_b[nz]) - np.log(py_b[nz])
        )
    )
    return mi


# =======================================================================================


def Copy_of_js_ms(dataIdx, delt, omega, RUNS):
    """
    Parameters
    ----------
    dataIdx : int
        Dataset index.
    delt : float
        DELT parameter (used here as correlation threshold for redundancy graph).
    omega : float
        OMEGA parameter.
    RUNS : int
        Number of runs.

    Returns
    -------
    product : list of [ACC, FNUM, FSET, RED, T]
    T : np.ndarray, shape (RUNS+2, 1)
    datasetName : str
    assiNum : np.ndarray, shape (RUNS+2, 1)
    """

    GG.DELT = delt
    GG.OMEGA = omega

    condata, GG.label, datasetName = myinputdatasetXD(dataIdx)
    output_dir = f"{GG.visual_dir}/{datasetName}"
    os.makedirs(output_dir, exist_ok=True)
    conData = condata[:, 1:]

    ACC = np.zeros(RUNS, dtype=float)
    FNUM = np.zeros(RUNS, dtype=float)
    FSET = [[] for _ in range(RUNS + 2)]
    RED = np.zeros(RUNS, dtype=float)

    assiNum = np.zeros(RUNS, dtype=float)

    zData = _mapminmax_zero_one(conData)
    GG.data = zData
    GG.featNum = zData.shape[1]

    T = np.zeros(RUNS, dtype=float)
    GG.kNeigh = 5  # vẫn giữ để không phá chỗ khác, nhưng không dùng cho k-NN edges nữa

    row = zData.shape[0]
    GG.run_logs = {}

    for Rtimes in range(1, RUNS + 1):
        print(f"---- Run {Rtimes} ----")
        print(f"     kNeigh = {GG.kNeigh}")

        GG.current_run = Rtimes
        GG.run_logs[Rtimes] = {
            "pareto_fronts": [],
            "pop_metrics": []
        }

        # reset counter (will be updated inside newtry_ms)
        GG.assiNumInside = []
        run_dir = f"{output_dir}/run_{Rtimes}"
        shutil.rmtree(run_dir, ignore_errors=True)
        os.makedirs(run_dir)

        tic = perf_counter()

        # Random train/test split 70/30
        perm = np.random.permutation(row)
        n_train = int(round(row * 0.7))
        train_idx = perm[:n_train]

        trIdx_arr = np.zeros(row, dtype=bool)
        trIdx_arr[train_idx] = True
        teIdx_arr = ~trIdx_arr
        GG.trIdx = trIdx_arr

        GG.trData = zData[trIdx_arr, :]
        GG.trLabel = GG.label[trIdx_arr]
        GG.teData = zData[teIdx_arr, :]
        GG.teLabel = GG.label[teIdx_arr]

        # ====== Fisher score (node relevance) ======
        _, vWeight0 = fisherScore(GG.trData, GG.trLabel)
        GG.vWeight = 1.0 + _mapminmax_zero_one(vWeight0.reshape(-1, 1)).ravel()
        GG.vWeight1 = GG.vWeight  # bản sao để dynamic update dùng

        # ====== Build correlation-based redundancy graph (absolute corr, threshold by DELT) ======
        # pairwise correlation similarity in vector form
        adj_vec = 1.0 - pdist(GG.trData.T, metric="correlation")
        adj_vec = np.nan_to_num(adj_vec, nan=0.0)
        adj_mat = squareform(adj_vec)
        adj_mat = np.abs(adj_mat)
        np.fill_diagonal(adj_mat, 0.0)

        # threshold by GG.DELT (if DELT is None, keep everything)
        if GG.DELT is None:
            red_mat = adj_mat
        else:
            red_mat = np.where(adj_mat >= GG.DELT, adj_mat, 0.0)

        # store redundancy graph
        GG.Weight = red_mat
        GG.Zout = red_mat

        # for backward compatibility: kNeiZout = redundancy graph
        GG.kNeiZout = GG.Zout

        # ====== MI-based relevance & complementarity (computed once per run on train data) ======
        y_train = GG.trLabel.ravel()
        d = GG.featNum

        # MI(feature; Y)
        mi_feat = np.zeros(d, dtype=float)
        for j in range(d):
            mi_feat[j] = _mutual_information_1d(GG.trData[:, j], y_train)
        GG.miFeat = mi_feat

        # Complementary matrix: only on edges kept by redundancy threshold
        comp_mat = np.zeros((d, d), dtype=float)
        rows_idx, cols_idx = np.where(np.triu(red_mat, k=1) > 0)
        for i, j in zip(rows_idx, cols_idx):
            mi_pair = _mutual_information_pair(
                GG.trData[:, i],
                GG.trData[:, j],
                y_train
            )
            c_ij = mi_pair - max(mi_feat[i], mi_feat[j])
            if c_ij > 0.0:
                comp_mat[i, j] = c_ij
                comp_mat[j, i] = c_ij
        GG.compMat = comp_mat

        # ====== Vectorized adjacency for optimizer interface (now threshold-based, not k-NN) ======
        kNeiAdj = squareform(GG.kNeiZout, force='tovector', checks=False)

        # ====== Feature selection algorithm (MOEA on nodes, will use GG.Zout & GG.compMat inside) ======
        featIdx = newtry_ms(kNeiAdj, 20, 50, run_dir)
        featIdx = np.asarray(featIdx)
        selected_features = np.where(featIdx != 0)[0]
        selected_num = selected_features.size

        # ====== Evaluate by KNN ======
        if selected_num == 0:
            acc = 0.0
            red = 0.0
        else:
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(GG.trData[:, selected_features], GG.trLabel)
            predLabel = knn.predict(GG.teData[:, selected_features])
            acc = np.mean(predLabel == GG.teLabel)

            red = redundancy_rate_subset(selected_features, zData)

        # ====== Store results ======
        idx = Rtimes - 1
        ACC[idx] = acc
        FNUM[idx] = selected_num
        FSET[idx] = selected_features
        RED[idx] = red

        T[idx] = perf_counter() - tic

        assiNum[idx] = np.sum(GG.assiNumInside)

    # ====== Save logs for each run ======
    for run_id, logs in GG.run_logs.items():
        run_dir = f"{output_dir}/run_{run_id}"
        pop_rows = logs.get("pop_metrics", [])
        if pop_rows:
            df_pop = pd.DataFrame(pop_rows)
            df_pop.to_csv(os.path.join(run_dir, "pop_metrics.csv"), index=False)
        pareto_list = logs.get("pareto_fronts", [])
        for entry in pareto_list:
            gen = entry["gen"]
            df = entry["df"]
            csv_file_name = os.path.join(run_dir, f"gen_{gen:03d}.csv")
            df.to_csv(csv_file_name, index=False)

    # ====== Append mean and std (RUNS+1, RUNS+2) ======
    ACC = np.concatenate([ACC, [ACC.mean(), ACC.std()]])
    FNUM = np.concatenate([FNUM, [FNUM.mean(), FNUM.std()]])
    RED = np.concatenate([RED, [RED.mean(), RED.std()]])
    T = np.concatenate([T, [T.mean(), T.std()]])
    assiNum = np.concatenate([assiNum, [assiNum.mean(), assiNum.std()]])

    # [ACC, FNUM, FSET, RED, T]
    product = []
    for i in range(RUNS + 2):
        product.append([ACC[i], FNUM[i], FSET[i], RED[i], T[i]])

    return product, T.reshape(-1, 1), datasetName, assiNum.reshape(-1, 1)
