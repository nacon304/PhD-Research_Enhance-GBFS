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

def Copy_of_js_ms_taugrid(
    dataIdx,
    delt,
    omega,
    RUNS,
    tau_grid,
):
    """
    Chạy RC-topk và RC-greedy cho nhiều giá trị rc_tau trên *cùng* dataset.

    Tránh re-split / re-build graph cho mỗi tau:
    - Với mỗi run:
        + Chia train/test
        + Tính Fisher, Zout, kNeiMatrix
        + Tính acc_single, C_matrix
      CHỈ 1 lần.
    - Sau đó lặp qua các rc_tau, set GG.rc_tau và gọi run_one_mode
    """

    tau_grid = list(tau_grid)
    n_tau = len(tau_grid)

    GG.DELT = delt
    GG.OMEGA = omega

    condata, GG.label, datasetName = myinputdatasetXD(dataIdx)
    output_dir = f"{GG.visual_dir}/{datasetName}"
    os.makedirs(output_dir, exist_ok=True)
    conData = condata[:, 1:]

    zData = _mapminmax_zero_one(conData)
    GG.data = zData
    GG.featNum = zData.shape[1]
    GG.kNeigh = 5

    row = zData.shape[0]
    GG.run_logs = {}

    # Mỗi tau & mỗi run sẽ có acc/fnum/time riêng
    ACC_topk   = np.zeros((n_tau, RUNS), dtype=float)
    FNUM_topk  = np.zeros((n_tau, RUNS), dtype=float)
    TIME_topk  = np.zeros((n_tau, RUNS), dtype=float)

    ACC_greedy  = np.zeros((n_tau, RUNS), dtype=float)
    FNUM_greedy = np.zeros((n_tau, RUNS), dtype=float)
    TIME_greedy = np.zeros((n_tau, RUNS), dtype=float)

    # nếu cần theo dõi số lượng evaluate bên trong toChangeWeight
    assiNum = np.zeros(RUNS, dtype=float)

    for Rtimes in range(1, RUNS + 1):
        print(f"---- Run {Rtimes} ----")
        print(f"     kNeigh = {GG.kNeigh}")

        GG.current_run = Rtimes
        GG.run_logs[Rtimes] = {
            "pareto_fronts": [],
            "pop_metrics": [],
        }
        GG.assiNumInside = []

        # em vẫn có thể giữ run_dir nếu muốn lưu gì đó cho baseline
        run_dir = f"{output_dir}/run_{Rtimes}"
        shutil.rmtree(run_dir, ignore_errors=True)
        os.makedirs(run_dir)

        tic_run = perf_counter()

        # ===== Stratified train/test split 70/30 =====
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.3,
            random_state=42,
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

        # ===== Fisher weight =====
        _, vWeight0 = fisherScore(GG.trData, GG.trLabel)
        GG.vWeight = 1.0 + _mapminmax_zero_one(vWeight0.reshape(-1, 1)).ravel()
        GG.vWeight1 = GG.vWeight

        # ===== Build correlation-based similarity graph =====
        adj = 1.0 - pdist(GG.trData.T, metric="correlation")
        adj = np.nan_to_num(adj, nan=0.0)
        adj = np.abs(adj)

        GG.Weight = squareform(adj)
        GG.Zout = squareform(adj)

        # ===== Keep k nearest neighbors =====
        GG.kNeiMatrix = np.zeros((GG.featNum, GG.kNeigh), dtype=int)
        kNeiZoutMode = np.zeros_like(GG.Zout)
        for i in range(GG.Zout.shape[0]):
            idx_sorted = np.argsort(-GG.Zout[i, :])
            idx_topk = idx_sorted[:GG.kNeigh]
            GG.kNeiMatrix[i, :] = idx_topk
            kNeiZoutMode[i, idx_topk] = 1

        GG.kNeiZout = GG.Zout * (kNeiZoutMode != 0)
        kNeiAdj = squareform(GG.kNeiZout, force="tovector", checks=False)

        # ===== Compute acc_single & complementarity on train set (1 lần / run) =====
        GG.acc_single = compute_single_feature_accuracy(
            GG.trData,
            GG.trLabel,
            n_neighbors=5,
            cv=5,
            use_cv=True,
            random_state=42,
        )

        GG.C_matrix = compute_complementarity_matrix(
            GG.trData,
            GG.trLabel,
            acc_single=GG.acc_single,
            n_neighbors=5,
            cv=5,
            use_cv=True,
            candidate_pairs=GG.kNeiMatrix,
            random_state=42,
        )

        init_time = perf_counter() - tic_run

        idx_run = Rtimes - 1
        assiNum[idx_run] = np.sum(GG.assiNumInside)

        # ===== Loop qua từng rc_tau, không re-split / re-build graph =====
        for t_idx, tau in enumerate(tau_grid):
            GG.rc_tau = float(tau)

            # --- rc_topk ---
            selected_S1, num_S1, acc_S1_run, _, _, elapsed_S1 = run_one_mode(
                kNeiAdj,
                run_dir,
                zData,
                seq_mode="rc_topk",
                pop=20,
                times=50,
            )
            ACC_topk[t_idx, idx_run] = acc_S1_run
            FNUM_topk[t_idx, idx_run] = num_S1
            TIME_topk[t_idx, idx_run] = elapsed_S1 + init_time

            # --- rc_greedy ---
            selected_S2, num_S2, acc_S2_run, _, _, elapsed_S2 = run_one_mode(
                kNeiAdj,
                run_dir,
                zData,
                seq_mode="rc_greedy",
                pop=20,
                times=50,
            )
            ACC_greedy[t_idx, idx_run] = acc_S2_run
            FNUM_greedy[t_idx, idx_run] = num_S2
            TIME_greedy[t_idx, idx_run] = elapsed_S2 + init_time

    # ===== Gộp kết quả thành DataFrame (mỗi dòng = 1 (tau, run)) =====
    rows = []
    for t_idx, tau in enumerate(tau_grid):
        for r in range(RUNS):
            rows.append(
                {
                    "tau": float(tau),
                    "run": r + 1,
                    "acc_rc_topk": ACC_topk[t_idx, r],
                    "fnum_rc_topk": FNUM_topk[t_idx, r],
                    "time_rc_topk": TIME_topk[t_idx, r],
                    "acc_rc_greedy": ACC_greedy[t_idx, r],
                    "fnum_rc_greedy": FNUM_greedy[t_idx, r],
                    "time_rc_greedy": TIME_greedy[t_idx, r],
                }
            )

        # thêm mean/std cho từng tau
        rows.append(
            {
                "tau": float(tau),
                "run": "mean",
                "acc_rc_topk": float(ACC_topk[t_idx].mean()),
                "fnum_rc_topk": float(FNUM_topk[t_idx].mean()),
                "time_rc_topk": float(TIME_topk[t_idx].mean()),
                "acc_rc_greedy": float(ACC_greedy[t_idx].mean()),
                "fnum_rc_greedy": float(FNUM_greedy[t_idx].mean()),
                "time_rc_greedy": float(TIME_greedy[t_idx].mean()),
            }
        )
        rows.append(
            {
                "tau": float(tau),
                "run": "std",
                "acc_rc_topk": float(ACC_topk[t_idx].std()),
                "fnum_rc_topk": float(FNUM_topk[t_idx].std()),
                "time_rc_topk": float(TIME_topk[t_idx].std()),
                "acc_rc_greedy": float(ACC_greedy[t_idx].std()),
                "fnum_rc_greedy": float(FNUM_greedy[t_idx].std()),
                "time_rc_greedy": float(TIME_greedy[t_idx].std()),
            }
        )

    df_seq = pd.DataFrame(rows)

    # nếu muốn vẫn lưu riêng cho dataset
    seq_summary_path = os.path.join(output_dir, "sequential_rc_tau_summary.csv")
    df_seq.to_csv(seq_summary_path, index=False)
    print(f"[save] {seq_summary_path}")

    return df_seq, datasetName, assiNum