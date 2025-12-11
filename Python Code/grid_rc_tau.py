import os
import numpy as np
import pandas as pd

import gbfs_globals as GG
from Testrun30 import Copy_of_js_ms

def grid_search_rc_tau(
    data_indices=(1, 2, 3, 4, 5),
    tau_grid=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
    delt=10.0,
    omega=0.8,
    RUNS=30,
    base_visual_dir="D:/PhD/The First Paper/Code Implement/GBFS-SND/Evaluation/Visualize Implementation 1 v2",
    out_csv="rc_tau_grid_results.csv",
):
    """
    Chạy grid search trên rc_tau cho các dataset đầu tiên.
    Lưu kết quả (mean trên RUNS) cho từng (dataset, tau) vào 1 file CSV.

    Kỳ vọng sequential_summary.csv trong mỗi folder dataset có cấu trúc:
      run, acc_base, fnum_base, time_base, acc_S1, fnum_S1, time_S1, acc_S2, fnum_S2, time_S2
      (S1 = rc_topk, S2 = rc_greedy trong Testrun30)
    """

    rows = []

    for tau in tau_grid:
        print(f"\n===== Grid rc_tau = {tau} =====")

        # set global param cho rc_topk / rc_greedy
        GG.rc_tau = float(tau)
        # nếu muốn giới hạn số feature thêm:
        GG.seq_max_add = 5
        GG.seq_max_buddy_per_core = 1

        # mỗi tau dùng 1 visual_dir riêng
        GG.visual_dir = os.path.join(base_visual_dir, f"tau_{tau}")
        os.makedirs(GG.visual_dir, exist_ok=True)

        for dataIdx in data_indices:
            print(f"  -> Dataset index {dataIdx}")
            product, T, datasetName, assiNum = Copy_of_js_ms(
                dataIdx, delt, omega, RUNS
            )

            # sequential_summary.csv nằm trong: visual_dir/datasetName/
            dataset_dir = os.path.join(GG.visual_dir, datasetName)
            seq_csv = os.path.join(dataset_dir, "sequential_summary.csv")

            if not os.path.exists(seq_csv):
                print(f"  [WARN] sequential_summary.csv not found for dataset {datasetName}, tau={tau}")
                continue

            df_seq = pd.read_csv(seq_csv)

            # lấy dòng mean
            if "mean" in df_seq["run"].values:
                row_mean = df_seq[df_seq["run"] == "mean"].iloc[0]
            else:
                # fallback: dùng trung bình từ phần RUNS dòng đầu
                row_mean = df_seq.iloc[0:RUNS].mean(numeric_only=True)

            rows.append({
                "dataset_idx": dataIdx,
                "dataset_name": datasetName,
                "rc_tau": tau,

                "acc_base": row_mean["acc_base"],
                "fnum_base": row_mean["fnum_base"],
                "time_base": row_mean["time_base"],

                "acc_rc_topk": row_mean["acc_S1"],
                "fnum_rc_topk": row_mean["fnum_S1"],
                "time_rc_topk": row_mean["time_S1"],

                "acc_rc_greedy": row_mean["acc_S2"],
                "fnum_rc_greedy": row_mean["fnum_S2"],
                "time_rc_greedy": row_mean["time_S2"],
            })

    df_all = pd.DataFrame(rows)
    out_path = os.path.join(base_visual_dir, out_csv)
    df_all.to_csv(out_path, index=False)
    print(f"\n[OK] Saved grid search results to: {out_path}")

    return df_all


if __name__ == "__main__":
    # ví dụ gọi nhanh cho 5 dataset đầu, RUNS=30
    df_res = grid_search_rc_tau(
        data_indices=(1, 2, 3, 4, 5),
        tau_grid=(0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5),
        delt=10.0,
        omega=0.8,
        RUNS=30,
        base_visual_dir="D:/PhD/The First Paper/Code Implement/GBFS-SND/Evaluation/Visualize Implementation 1 v2",
        out_csv="rc_tau_grid_results.csv",
    )
