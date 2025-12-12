import os
import numpy as np
import pandas as pd

import gbfs_globals as GG
from copy_of_js_ms import Copy_of_js_ms_taugrid

def grid_search_rc_tau(
    data_indices=(1, 2, 3, 4),
    tau_grid=(0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5),
    delt=10.0,
    omega=0.8,
    RUNS=10,
    base_visual_dir="D:/PhD/The First Paper/Code Implement/GBFS-SND/Evaluation/Visualize Implementation 1 v3",
    out_csv="rc_tau_grid_results.csv",
):
    all_rows = []

    for dataIdx in data_indices:
        print(f"\n========== DATASET {dataIdx} ==========")
        # visual_dir gốc cho dataset này (bên trong Copy_of_js_ms_taugrid sẽ tạo sub-folder datasetName)
        GG.visual_dir = base_visual_dir

        df_seq, datasetName, assiNum = Copy_of_js_ms_taugrid(
            dataIdx,
            delt,
            omega,
            RUNS,
            tau_grid,
        )

        # Lọc ra dòng mean để làm bảng tổng hợp (mỗi (dataset, tau) 1 dòng)
        df_mean = df_seq[df_seq["run"] == "mean"].copy()

        for _, row in df_mean.iterrows():
            all_rows.append(
                {
                    "dataset_idx": dataIdx,
                    "dataset_name": datasetName,
                    "rc_tau": row["tau"],
                    "acc_rc_topk": row["acc_rc_topk"],
                    "fnum_rc_topk": row["fnum_rc_topk"],
                    "time_rc_topk": row["time_rc_topk"],
                    "acc_rc_greedy": row["acc_rc_greedy"],
                    "fnum_rc_greedy": row["fnum_rc_greedy"],
                    "time_rc_greedy": row["time_rc_greedy"],
                }
            )

    df_all = pd.DataFrame(all_rows)
    out_path = os.path.join(base_visual_dir, out_csv)
    df_all.to_csv(out_path, index=False)
    print(f"\n[OK] Saved grid search results to: {out_path}")
    return df_all

if __name__ == "__main__":
    df_res = grid_search_rc_tau()