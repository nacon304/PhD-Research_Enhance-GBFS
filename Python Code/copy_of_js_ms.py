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

from init_strategies import init_graph_local_threshold, init_graph_fisher_redundancy_degree, init_graph_mst_plus_local, init_graph_probabilistic

def build_initial_graph(Z, fisher_raw=None):
    """
    Trả về:
        A_init : adjacency n×n (float), dùng làm đồ thị sparse ban đầu
        neighbors : list[np.ndarray], neighbors[i] là list các đỉnh kề của i

    Dựa trên GG.init_mode và các hyper trong gbfs_globals.
    """
    mode = getattr(GG, "init_mode", "knn")  # mặc định là như code cũ
    Z = np.asarray(Z, dtype=float)

    if mode == "knn":
        # giữ đúng hành vi cũ: mỗi node nối với GG.kNeigh hàng xóm gần nhất (directed)
        n = Z.shape[0]
        k = GG.kNeigh
        A_init = np.zeros_like(Z, dtype=float)
        neighbors = []
        for i in range(n):
            row = Z[i].copy()
            row[i] = 0.0
            idx_sorted = np.argsort(-row)
            idx_topk = idx_sorted[:k]
            neighbors.append(idx_topk.astype(int))
            A_init[i, idx_topk] = Z[i, idx_topk]
        return A_init, neighbors

    elif mode == "local_thresh":
        A_init, neighbors = init_graph_local_threshold(
            Z,
            k_min=getattr(GG, "init_k_min", 1),
            quantile=getattr(GG, "init_quantile", 0.8),
            exclude_self=True,
            symmetrize=True,
        )
        return A_init, neighbors

    elif mode == "fisher_red_deg":
        if fisher_raw is None:
            raise ValueError("build_initial_graph: fisher_raw is required for fisher_red_deg mode")
        A_init, degs, neighbors = init_graph_fisher_redundancy_degree(
            Z,
            fisher_scores=fisher_raw,
            k_min=getattr(GG, "init_k_min", 1),
            k_max=getattr(GG, "init_k_max", min(GG.kNeigh * 3, Z.shape[0] - 1)),
            exclude_self=True,
            symmetrize=True,
        )
        GG.init_degrees = degs
        return A_init, neighbors

    elif mode == "mst_plus_local":
        A_init, neighbors = init_graph_mst_plus_local(
            Z,
            extra_k=getattr(GG, "init_extra_k", 2),
            exclude_self=True,
            symmetrize=True,
        )
        return A_init, neighbors

    elif mode == "probabilistic":
        if fisher_raw is None:
            raise ValueError("build_initial_graph: fisher_raw is required for probabilistic mode")
        A_init, neighbors = init_graph_probabilistic(
            Z,
            fisher_scores=fisher_raw,
            k_min=getattr(GG, "init_k_min", 1),
            k_max=getattr(GG, "init_k_max", min(GG.kNeigh * 3, Z.shape[0] - 1)),
            beta=getattr(GG, "init_beta", 2.0),
            exclude_self=True,
            symmetrize=True,
            random_state=getattr(GG, "init_seed", 42),
        )
        return A_init, neighbors

    else:
        raise ValueError(f"Unknown GG.init_mode = {mode}")

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
        adj = np.nan_to_num(adj, nan=0.0)
        adj = np.abs(adj)
        GG.Weight = squareform(adj)
        GG.Zout = squareform(adj)

        # ====== Keep k nearest neighbors for each feature ======
        A_init, neigh_list = build_initial_graph(GG.Zout, fisher_raw=vWeight0)
        GG.kNeiZout = A_init.copy()
        GG.neigh_list = neigh_list
        kNeiAdj = squareform(GG.kNeiZout, force="tovector", checks=False)
        
        # ====== Feature selection algorithm ======
        featIdx, _, _, kNeigh_chosen = newtry_ms(kNeiAdj, 20, 50, run_dir)
        featIdx = np.asarray(featIdx)
        selected_features = np.where(featIdx != 0)[0]
        selected_num = selected_features.size
        GG.run_logs[Rtimes]["kNeigh_chosen"] = kNeigh_chosen
        GG.run_logs[Rtimes]["selected_features"] = selected_features
        GG.run_logs[Rtimes]["neighbors"] = neigh_list

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
        p_neighbors = logs.get("neighbors", None)

        plot_knn_graph_with_selected(
            p_neighbors,
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
