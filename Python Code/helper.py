import numpy as np
import gbfs_globals as GG
import pandas as pd
import os

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