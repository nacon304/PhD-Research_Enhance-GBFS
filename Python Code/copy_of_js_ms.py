import os
import shutil
import numpy as np
from time import perf_counter
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KNeighborsClassifier

from myinputdatasetXD import myinputdatasetXD
from fisherScore import fisherScore
from newtry_ms import newtry_ms

import gbfs_globals as GG

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

def redundancy_rate_subset(selected_features, X):
    """
    Calculate Red(S) for a subset S of features:
        Red(S) = 1 / (|S|(|S|-1)) * sum_{i!=j} cos^2(fi, fj)

    selected_features : array-like of indices (shape (k,))
        Indices of selected features in S.
    X : np.ndarray, shape (n_samples, n_features)
        Data (preferably normalized, e.g., zData).
    """
    selected_features_copy = np.asarray(selected_features, dtype=int)
    k = selected_features_copy.size

    if k < 2:
        return 0.0

    F = X[:, selected_features_copy]  # each column is a feature

    # Normalize each column to unit vector so cos = dot product
    norms = np.linalg.norm(F, axis=0)
    norms[norms == 0] = 1e-12
    F_norm = F / norms

    # Cosine matrix k×k
    C = F_norm.T @ F_norm
    C2 = C ** 2

    # Sum of cos^2(fi, fj) with i != j
    sum_all = np.sum(C2)
    sum_diag = np.sum(np.diag(C2))
    sum_pairs = sum_all - sum_diag   # exclude i=j

    # Apply formula
    return float(sum_pairs / (k * (k - 1)))

def Copy_of_js_ms(dataIdx, delt, omega, RUNS):
    """
    Parameters
    ----------
    dataIdx : int
        Dataset index.
    delt : float
        DELT parameter.
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
    GG.kNeigh = 5

    row = zData.shape[0]

    for Rtimes in range(1, RUNS + 1):
        print(f"---- Run {Rtimes} ----")
        print(f"     kNeigh = {GG.kNeigh}")

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

        _, vWeight0 = fisherScore(GG.trData, GG.trLabel)
        GG.vWeight = 1.0 + _mapminmax_zero_one(vWeight0.reshape(-1, 1)).ravel()
        GG.vWeight1 = GG.vWeight

        # ====== Build correlation-based similarity graph 
        adj = 1.0 - pdist(GG.trData.T, metric="correlation")
        # replace NaN with 0
        adj = np.nan_to_num(adj, nan=0.0)
        adj = np.abs(adj)
        # full matrix
        GG.Weight = squareform(adj)
        GG.Zout = squareform(adj)

        # ====== Keep k nearest neighbors for each feature ======
        GG.kNeiMatrix = np.zeros((GG.featNum, GG.kNeigh), dtype=int)
        kNeiZoutMode = np.zeros_like(GG.Zout)
        for i in range(GG.Zout.shape[0]):
            # sort descending similarity
            idx_sorted = np.argsort(-GG.Zout[i, :])
            idx_topk = idx_sorted[:GG.kNeigh]
            GG.kNeiMatrix[i, :] = idx_topk
            kNeiZoutMode[i, idx_topk] = 1

        GG.kNeiZout = GG.Zout * (kNeiZoutMode != 0)
        kNeiAdj = squareform(GG.kNeiZout, force='tovector', checks=False)

        # ====== Feature selection algorithm ======
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
