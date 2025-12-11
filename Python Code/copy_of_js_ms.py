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
from helper import (
    _mapminmax_zero_one,
    redundancy_rate_subset,
    plot_knn_graph_with_selected,
    compute_single_feature_accuracy,
    compute_complementarity_matrix,
)

from sequential_strategies import (
    sfs_acc_only,
    sfs_acc_comp_red,
    sfs_prefilter_comp,
    buddy_sfs,
)

from sklearn.model_selection import StratifiedShuffleSplit

import gbfs_globals as GG

def eval_on_test(X_tr, y_tr, X_te, y_te, S):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_tr[:, S], y_tr)
    pred = knn.predict(X_te[:, S])
    return np.mean(pred == y_te)

def run_one_mode(kNeiAdj, run_dir, zData, seq_mode, pop=20, times=50):
    """
    Chạy newtry_ms với một chiến lược seq_mode (sau K-shell) và trả về:
        - selected_features
        - selected_num
        - acc_test
        - red (redundancy)
        - kNeigh_chosen
        - elapsed_time  (NEW)
    """
    GG.seq_mode = seq_mode

    tic_mode = perf_counter()

    if seq_mode != "none":
        run_dir = ''
    featIdx, _, _, kNeigh_chosen = newtry_ms(kNeiAdj, pop, times, run_dir)
    featIdx = np.asarray(featIdx)
    selected_features = np.where(featIdx != 0)[0]
    selected_num = selected_features.size

    if selected_num == 0:
        acc = 0.0
        red = 0.0
    else:
        acc = eval_on_test(GG.trData, GG.trLabel, GG.teData, GG.teLabel, selected_features)
        if seq_mode == "none":
            red = redundancy_rate_subset(selected_features, zData)
        else:
            red = 0.0

    elapsed = perf_counter() - tic_mode

    return selected_features, selected_num, acc, red, kNeigh_chosen, elapsed

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
    T = np.zeros(RUNS, dtype=float)

    ACC_S1 = np.zeros(RUNS, dtype=float)
    ACC_S2 = np.zeros(RUNS, dtype=float)
    ACC_S3 = np.zeros(RUNS, dtype=float)
    ACC_S4 = np.zeros(RUNS, dtype=float)
    FNUM_S1 = np.zeros(RUNS, dtype=float)
    FNUM_S2 = np.zeros(RUNS, dtype=float)
    FNUM_S3 = np.zeros(RUNS, dtype=float)
    FNUM_S4 = np.zeros(RUNS, dtype=float)
    TIME_S1 = np.zeros(RUNS, dtype=float)
    TIME_S2 = np.zeros(RUNS, dtype=float)
    TIME_S3 = np.zeros(RUNS, dtype=float)
    TIME_S4 = np.zeros(RUNS, dtype=float)

    assiNum = np.zeros(RUNS, dtype=float)

    zData = _mapminmax_zero_one(conData)
    GG.data = zData
    GG.featNum = zData.shape[1]

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

        # ====== (NEW) Compute relevance & complementarity on train set ======
        # 1) Accuracy for each feature
        GG.acc_single = compute_single_feature_accuracy(
            GG.trData, GG.trLabel,
            n_neighbors=5,
            cv=5,
            use_cv=True,
            random_state=42
        )
        # 2) Complementary matrix C_ij
        GG.C_matrix = compute_complementarity_matrix(
            GG.trData, GG.trLabel,
            acc_single=GG.acc_single,
            n_neighbors=5,
            cv=5,
            use_cv=True,
            candidate_pairs=GG.kNeiMatrix,  # only neighboring pairs
            random_state=42
        )

        init_time = perf_counter() - tic

        # # ====== Feature selection algorithm cho từng mode ======
        # # Mode 0: baseline (không sequential sau K-shell)
        # selected_features_base, selected_num_base, acc_base, red_base, kNeigh_chosen_base, elapsed_base = run_one_mode(
        #     kNeiAdj, run_dir, zData, seq_mode="none", pop=20, times=50
        # )
        # # ====== Store results (baseline) ======
        idx = Rtimes - 1
        # ACC[idx] = acc_base
        # FNUM[idx] = selected_num_base
        # FSET[idx] = selected_features_base
        # RED[idx] = red_base
        # T[idx] = elapsed_base + init_time
        # assiNum[idx] = np.sum(GG.assiNumInside)

        # Mode 1–4: các chiến lược sequential sau K-shell
        selected_S1, num_S1, acc_S1_run, _, _, elapsed_S1 = run_one_mode(
            kNeiAdj, run_dir, zData, seq_mode="rc_topk", pop=20, times=50
        )
        selected_S2, num_S2, acc_S2_run, _, _, elapsed_S2 = run_one_mode(
            kNeiAdj, run_dir, zData, seq_mode="rc_greedy", pop=20, times=50
        )
        # ====== Store results (4 sequential strategies) ======
        ACC_S1[idx] = acc_S1_run
        ACC_S2[idx] = acc_S2_run
        FNUM_S1[idx] = num_S1
        FNUM_S2[idx] = num_S2
        TIME_S1[idx] = elapsed_S1 + init_time
        TIME_S2[idx] = elapsed_S2 + init_time

    # ====== Save logs for each run ======
    # for run_id, logs in GG.run_logs.items():
    #     run_dir = f"{output_dir}/run_{run_id}"
    #     pop_rows = logs.get("pop_metrics", [])
    #     if pop_rows:
    #         df_pop = pd.DataFrame(pop_rows)
    #         df_pop.to_csv(os.path.join(run_dir, "pop_metrics.csv"), index=False)
    #     pareto_list = logs.get("pareto_fronts", [])
    #     for entry in pareto_list:
    #         gen = entry["gen"]
    #         df = entry["df"]
    #         csv_file_name = os.path.join(run_dir, f"gen_{gen:03d}.csv")
    #         df.to_csv(csv_file_name, index=False)
        # p_kNeigh_chosen = logs.get("kNeigh_chosen", None)
        # p_selected_features = logs.get("selected_features", None)
        # p_kNeiMatrix = logs.get("kNeiMatrix", None)
        # plot_knn_graph_with_selected(
        #     p_kNeiMatrix,
        #     p_kNeigh_chosen,
        #     p_selected_features,
        #     run_dir,
        #     run_id=run_id,
        #     filename_prefix="knn_graph"
        # )
    
    seq_rows = []
    for r in range(RUNS):
        seq_rows.append({
            "run": r + 1,
            "acc_base": ACC[r],
            "fnum_base": FNUM[r],
            "time_base": T[r],

            "acc_S1": ACC_S1[r],
            "fnum_S1": FNUM_S1[r],
            "time_S1": TIME_S1[r],

            "acc_S2": ACC_S2[r],
            "fnum_S2": FNUM_S2[r],
            "time_S2": TIME_S2[r],
        })
    seq_rows.append({
        "run": "mean",
        "acc_base": float(ACC.mean()),
        "fnum_base": float(FNUM.mean()),
        "time_base": float(T.mean()),

        "acc_S1": float(ACC_S1.mean()),
        "fnum_S1": float(FNUM_S1.mean()),
        "time_S1": float(TIME_S1.mean()),

        "acc_S2": float(ACC_S2.mean()),
        "fnum_S2": float(FNUM_S2.mean()),
        "time_S2": float(TIME_S2.mean()),
    })
    seq_rows.append({
        "run": "std",
        "acc_base": float(ACC.std()),
        "fnum_base": float(FNUM.std()),
        "time_base": float(T.std()),

        "acc_S1": float(ACC_S1.std()),
        "fnum_S1": float(FNUM_S1.std()),
        "time_S1": float(TIME_S1.std()),

        "acc_S2": float(ACC_S2.std()),
        "fnum_S2": float(FNUM_S2.std()),
        "time_S2": float(TIME_S2.std()),
    })
    df_seq = pd.DataFrame(seq_rows)
    return df_seq
    # seq_summary_path = os.path.join(output_dir, "sequential_summary.csv")
    # df_seq.to_csv(seq_summary_path, index=False)

    # # ====== Append mean and std (RUNS+1, RUNS+2) ======
    # ACC = np.concatenate([ACC, [ACC.mean(), ACC.std()]])
    # FNUM = np.concatenate([FNUM, [FNUM.mean(), FNUM.std()]])
    # RED = np.concatenate([RED, [RED.mean(), RED.std()]])
    # T = np.concatenate([T, [T.mean(), T.std()]])
    # assiNum = np.concatenate([assiNum, [assiNum.mean(), assiNum.std()]])

    # # [ACC, FNUM, FSET, RED, T]
    # product = []
    # for i in range(RUNS + 2):
    #     product.append([ACC[i], FNUM[i], FSET[i], RED[i], T[i]])

    # return product, T.reshape(-1, 1), datasetName, assiNum.reshape(-1, 1)
