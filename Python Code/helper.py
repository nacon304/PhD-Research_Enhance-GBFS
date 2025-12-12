import numpy as np
import gbfs_globals as GG
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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

def save_pareto_front_csv(chromosome_f, i, V_f, run_dir):
    """
    Parameters
    ----------
    chromosome_f : np.ndarray
        Population after non-domination sort (with rank & objectives).
    i : int
        Generation number.
    V_f : int
        Number of decision variables (feature-length).
    run_dir : str
        Directory to save files.
    """
    if run_dir is None:
        return
    csv_file_name = f"{run_dir}/gen_{i:03d}.csv"

    ranks = chromosome_f[:, V_f + GG.M].astype(int)
    mask = (ranks == 1)
    pareto = chromosome_f[mask]

    objs = pareto[:, V_f: V_f + GG.M]

    df = pd.DataFrame(objs, columns=["obj1", "obj2"])
    run_id = getattr(GG, "current_run", None)
    if run_id is not None and run_id in GG.run_logs:
        GG.run_logs[run_id]["pareto_fronts"].append(
            {"gen": i, "df": df}
        )
    
    # df.to_csv(csv_file_name, index=False)

def log_population_metrics(chromosome_f, i, V_f, run_dir):
    """
    Parameters
    ----------
    chromosome_f : np.ndarray
        Population after replace_chromosome (already has objectives, rank, etc.)
    i : int
        Generation index (starting from 1)
    V_f : int
        Length of decision-variable part (feature-side)
    run_dir : str
        Directory of the current run (to save CSV files)
    """
    objs = chromosome_f[:, V_f:V_f + GG.M]
    obj1 = objs[:, 0]
    obj2 = objs[:, 1]

    mean_obj1 = float(np.mean(obj1))
    mean_obj2 = float(np.mean(obj2))
    std_obj1 = float(np.std(obj1))
    std_obj2 = float(np.std(obj2))

    rank_col_index = V_f + GG.M
    if rank_col_index < chromosome_f.shape[1]:
        ranks = chromosome_f[:, rank_col_index]
        prop_rank1 = float(np.mean(ranks == 1))
    else:
        prop_rank1 = np.nan

    metrics_path = os.path.join(run_dir, "pop_metrics.csv")

    row = {
        "run": getattr(GG, "current_run", None),
        "gen": i,
        "mean_obj1": mean_obj1,
        "mean_obj2": mean_obj2,
        "std_obj1": std_obj1,
        "std_obj2": std_obj2,
        "prop_rank1": prop_rank1,
    }

    run_id = row["run"]
    if run_id is not None and run_id in GG.run_logs:
        GG.run_logs[run_id]["pop_metrics"].append(row)
        return

    # if not os.path.exists(metrics_path):
    #     df = pd.DataFrame([row])
    #     df.to_csv(metrics_path, index=False)
    # else:
    #     df = pd.DataFrame([row])
    #     df.to_csv(metrics_path, mode="a", header=False, index=False)

def plot_knn_graph_with_selected(neighbors, edge_mask,
                                 selected_features,
                                 run_dir,
                                 run_id=None,
                                 filename_prefix="knn_graph"):
    """
    Vẽ đồ thị từ neigh_list (GG.neigh_list) và vector bit edge_mask (kNeigh_chosen).

    Representation (đồng nhất với decodeNet & V_f):
        - GG.neigh_list: list length = n_features,
              neigh_list[i] = array/list các neighbor j của i,
              duyệt theo đúng thứ tự để flatten.
        - edge_mask: mảng 0/1 độ dài V_f, với
              V_f = sum(len(nb) for nb in GG.neigh_list)
          Mỗi phần tử edge_mask[idx] quyết định có vẽ cạnh (i, j) tương ứng hay không,
          với thứ tự duyệt:
              idx chạy tăng dần theo:
                  for i in range(n_features):
                      for j in neigh_list[i]:
                          ...
    """
    # --- Lấy neigh_list từ global ---
    neigh_list = neighbors

    # Đảm bảo tất cả phần tử là np.array[int]
    neigh_list = [np.asarray(nb, dtype=int) for nb in neigh_list]

    n_features = len(neigh_list)
    edge_mask = np.asarray(edge_mask).ravel()
    selected_features = np.asarray(selected_features, dtype=int)

    expected_len = int(sum(len(nb) for nb in neigh_list))
    if edge_mask.size != expected_len:
        raise ValueError(
            f"edge_mask length {edge_mask.size} != "
            f"sum(len(nb) for nb in neigh_list) = {expected_len}"
        )

    # ---- Xây graph từ bit-mask ----
    G = nx.Graph()
    G.add_nodes_from(range(n_features))

    idx = 0
    for i in range(n_features):
        neigh = neigh_list[i]
        for j in neigh:
            if edge_mask[idx] != 0:
                j_int = int(j)
                if i != j_int:
                    G.add_edge(i, j_int)
            idx += 1

    # ---- Layout ----
    pos = nx.spring_layout(G, seed=42)

    # ---- Node style: tô đậm node được chọn ----
    node_colors = []
    node_sizes = []
    selected_set = set(selected_features.tolist())

    for node in G.nodes():
        if node in selected_set:
            node_colors.append("tab:red")   # feature được chọn
            node_sizes.append(700)
        else:
            node_colors.append("tab:blue")  # feature không được chọn
            node_sizes.append(300)

    # ---- Edge style: đậm hơn nếu nối giữa 2 selected features ----
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        if u in selected_set and v in selected_set:
            edge_colors.append("tab:red")
            edge_widths.append(2.5)
        elif u in selected_set or v in selected_set:
            edge_colors.append("tab:orange")
            edge_widths.append(1.5)
        else:
            edge_colors.append("lightgray")
            edge_widths.append(0.8)

    # ---- Vẽ ----
    plt.figure(figsize=(6, 6))

    if len(G.edges()) > 0:
        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.8
        )

    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
    )
    nx.draw_networkx_labels(G, pos, font_size=10)

    title = "Graph (mask-based edges)"
    if run_id is not None:
        title += f" - Run {run_id}"
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    # ---- Lưu hình ----
    if run_id is None:
        fname = f"{filename_prefix}.png"
    else:
        fname = f"{filename_prefix}_run{run_id}.png"

    out_path = os.path.join(run_dir, fname)
    plt.savefig(out_path, dpi=300)
    plt.close()