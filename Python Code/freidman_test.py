import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare

# Nếu cần Nemenyi test:
# pip install scikit-posthocs
import scikit_posthocs as sp


# ==== HÀM CHÍNH ====
def rank_and_significance(
    csv_path,
    metric_col="acc_rc_topk",   # metric để xếp hạng
    higher_is_better=True,      # accuracy => True, subset_size/time => False
    alpha=0.05
):
    # 1. Đọc file
    df = pd.read_csv(csv_path)

    # 2. Tạo ma trận: mỗi hàng = dataset, mỗi cột = rc_tau
    table = df.pivot(index="dataset_name", columns="rc_tau", values=metric_col)
    # Đảm bảo sắp xếp theo rc_tau tăng dần
    table = table.sort_index(axis=1)

    print("=== Bảng giá trị (dataset x tau) ===")
    print(table, "\n")

    # 3. Tính rank cho từng dataset
    #   - ascending=False nếu higher_is_better (accuracy)
    #   - ascending=True nếu lower_is_better (fnum, time, ...)
    ranks = table.rank(
        axis=1,
        ascending=not higher_is_better,
        method="average"
    )

    avg_rank = ranks.mean(axis=0).sort_values()

    print("=== Bảng rank cho từng dataset ===")
    print(ranks, "\n")

    print("=== Average rank (giống dòng AvgRank) ===")
    print(avg_rank, "\n")

    # 4. Friedman test trên các cột (mỗi cột 1 giá trị tau)
    stat, p = friedmanchisquare(*[table[c].values for c in table.columns])
    print(f"Friedman test: chi2 = {stat:.4f}, p-value = {p:.4f}")
    if p < alpha:
        print(f"==> Có khác biệt thống kê giữa các hệ số tau (alpha = {alpha}).\n")
    else:
        print(f"==> Không có khác biệt thống kê rõ rệt (alpha = {alpha}).\n")

    # 5. Nemenyi post-hoc test dựa trên rank
    # table.values có dạng (n_dataset, n_tau)
    nemenyi_p = sp.posthoc_nemenyi_friedman(table.values)
    nemenyi_p.index = table.columns
    nemenyi_p.columns = table.columns

    print("=== Ma trận p-value Nemenyi (so sánh từng cặp tau) ===")
    print(nemenyi_p, "\n")

    # 6. Chọn hệ số tau tốt nhất theo average rank
    best_tau = avg_rank.index[0]

    # Các tau không kém best_tau một cách có ý nghĩa thống kê
    # (p >= alpha khi so với best_tau)
    non_worse = [
        tau for tau in table.columns
        if (tau == best_tau) or (nemenyi_p.loc[best_tau, tau] >= alpha)
    ]

    print(f"Hệ số tau có avg rank nhỏ nhất (tốt nhất): {best_tau}")
    print(f"Các tau KHÔNG thua kém best_tau có ý nghĩa thống kê (alpha={alpha}):")
    print(non_worse)

    # Trả ra để bạn dùng tiếp nếu cần
    return {
        "table": table,
        "ranks": ranks,
        "avg_rank": avg_rank,
        "friedman": (stat, p),
        "nemenyi_p": nemenyi_p,
        "best_tau": best_tau,
        "non_worse": non_worse,
    }

# ==== VÍ DỤ GỌI HÀM ====
if __name__ == "__main__":
    result_topk = rank_and_significance(
        csv_path="D:/PhD/The First Paper/Code Implement/GBFS-SND/Evaluation/Visualize Implementation 1 v3/rc_tau_grid_results.csv",
        metric_col="acc_rc_topk",
        higher_is_better=True,
        alpha=0.05
    )

    # Nếu muốn làm tương tự cho acc_rc_greedy:
    result_greedy = rank_and_significance(
        csv_path="D:/PhD/The First Paper/Code Implement/GBFS-SND/Evaluation/Visualize Implementation 1 v3/rc_tau_grid_results.csv",
        metric_col="acc_rc_greedy",
        higher_is_better=True,
        alpha=0.05
    )

    # Nếu muốn xét số feature (fnum) thì đổi metric + higher_is_better:
    # result_fnum = rank_and_significance(
    #     csv_path="D:/PhD/The First Paper/Code Implement/GBFS-SND/Evaluation/Visualize Implementation 1 v3/rc_tau_grid_results.csv",
    #     metric_col="fnum_rc_topk",
    #     higher_is_better=False,   # ít feature hơn là tốt hơn
    #     alpha=0.05
    # )
