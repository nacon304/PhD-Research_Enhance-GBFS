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
from helper import _mapminmax_zero_one, redundancy_rate_subset, plot_knn_graph_with_selected

from sklearn.model_selection import StratifiedShuffleSplit

import gbfs_globals as GG

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
    product : list of [ACC, FNUM, FSET, T]
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
    GG.run_logs = {}

    for Rtimes in range(1, RUNS + 1):
        print(f"---- Run {Rtimes} ----")
        print(f"     kNeigh = {GG.kNeigh}")

        # reset counter (will be updated inside newtry_ms)
        GG.current_run = Rtimes
        GG.run_logs[Rtimes] = {
            "pareto_fronts": [],
            "pop_metrics": []
        }
        GG.assiNumInside = []
        run_dir = f"{output_dir}/run_{Rtimes}"
        shutil.rmtree(run_dir, ignore_errors=True)
        os.makedirs(run_dir)

        tic = perf_counter()

        # ===== Stratified train/test split 70/30 =====
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.3,
            random_state=42
        )

        for train_index, test_index in sss.split(zData, GG.label):
            GG.trIdx = np.zeros(row, dtype=bool)
            GG.trIdx[train_index] = True

            trIdx_arr = GG.trIdx
            teIdx_arr = ~trIdx_arr

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
        featIdx, _, _, kNeigh_chosen = newtry_ms(kNeiAdj, 20, 50, run_dir)
        featIdx = np.asarray(featIdx)
        selected_features = np.where(featIdx != 0)[0]
        selected_num = selected_features.size
        GG.run_logs[Rtimes]["kNeigh_chosen"] = kNeigh_chosen
        GG.run_logs[Rtimes]["selected_features"] = selected_features
        GG.run_logs[Rtimes]["kNeiMatrix"] = GG.kNeiMatrix

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
        p_kNeigh_chosen = logs.get("kNeigh_chosen", None)
        p_selected_features = logs.get("selected_features", None)
        p_kNeiMatrix = logs.get("kNeiMatrix", None)
        plot_knn_graph_with_selected(
            p_kNeiMatrix,
            p_kNeigh_chosen,         
            p_selected_features,
            run_dir,
            run_id=run_id,
            filename_prefix="knn_graph"
        )

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
