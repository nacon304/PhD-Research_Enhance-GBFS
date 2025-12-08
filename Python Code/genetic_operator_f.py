import numpy as np

from CXoperate import CXoperate
from evaluate_objective_f import evaluate_objective_f


def genetic_operator_f(parent_chromosome, M, V, px, pm, templateAdj):
    """
    Genetic operators for NODE-based chromosome.

    Parameters
    ----------
    parent_chromosome : np.ndarray, shape (N, V+M+...)
        Current parent population; gene part is in the first V columns (node mask z).
    M : int
        Number of objectives.
    V : int
        Number of decision variables (length of node chromosome).
    px : float
        Crossover probability.
    pm : float
        Mutation probability.
    templateAdj : array-like
        Kept for API compatibility (not used in node-based version).

    Returns
    -------
    f : np.ndarray, shape (N_child, V+M)
        Offspring chromosomes: [genes (node mask), objectives].
    featIdx_f : np.ndarray, shape (N_child, n_features_full)
        Final feature masks F(z) corresponding to each offspring.
    """
    parent_chromosome = np.asarray(parent_chromosome, dtype=float)
    N, total_cols = parent_chromosome.shape

    if total_cols < V:
        raise ValueError(
            f"genetic_operator_f: parent_chromosome has {total_cols} columns, "
            f"but V = {V}."
        )

    # ---- Crossover & mutation trên phần GENES (node-based) ----
    # parent_chromosome[:, :V] là z-vector cho từng cá thể
    child_chromosome = CXoperate(parent_chromosome[:, :V], px, pm)

    child_chromosome = np.asarray(child_chromosome, dtype=float)
    N_child, V_child = child_chromosome.shape
    if V_child != V:
        raise ValueError(
            f"CXoperate must return array with {V} columns, got {V_child}."
        )

    # Đảm bảo gene là nhị phân 0/1 (phòng trường hợp CXoperate tạo giá trị khác)
    # (nếu bạn chắc chắn CXoperate luôn cho 0/1 thì có thể bỏ bước này)
    child_chromosome = (child_chromosome > 0.5).astype(int)

    # Tránh subset rỗng: nếu cá thể nào toàn 0 thì bật ngẫu nhiên 1 feature
    for i in range(N_child):
        if child_chromosome[i, :].sum() == 0:
            j = np.random.randint(0, V)
            child_chromosome[i, j] = 1

    # ---- Chuẩn bị output: [genes (V), objectives (M)] ----
    f = np.zeros((N_child, V + M), dtype=float)
    f[:, :V] = child_chromosome

    featIdx_list = []

    # ---- Evaluate từng offspring bằng Algorithm 1 (2-phase k-shell) ----
    for i in range(N_child):
        z = child_chromosome[i, :V].astype(int)

        obj_vals, feat_mask = evaluate_objective_f(M, z)
        obj_vals = np.asarray(obj_vals, dtype=float).ravel()

        if obj_vals.size != M:
            raise ValueError(
                f"evaluate_objective_f must return {M} objectives, "
                f"got {obj_vals.size}"
            )

        f[i, V : V + M] = obj_vals
        featIdx_list.append(np.asarray(feat_mask, dtype=bool).ravel())

    featIdx_f = np.vstack(featIdx_list)

    return f, featIdx_f
