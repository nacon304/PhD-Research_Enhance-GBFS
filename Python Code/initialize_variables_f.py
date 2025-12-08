import numpy as np

from evaluate_objective_f import evaluate_objective_f
import gbfs_globals as GG


def initialize_variables_f(N, M, V, templateAdj):
    """
    Initialize population for node-based MOEA.

    Parameters
    ----------
    N : int
        Population size.
    M : int
        Number of objectives.
    V : int
        Length of NODE chromosome (should equal GG.featNum).
    templateAdj : array-like
        Kept for API compatibility (not used in node-based version).

    Returns
    -------
    f : np.ndarray, shape (N, V+M)
        Population with:
        - columns 0..V-1: binary node chromosome z (candidate subset S(z))
        - columns V..V+M-1: objective values [f1, f2]
    featIdx_f : np.ndarray, shape (N, featNum)
        Final feature mask F(z) cho từng cá thể (sau 2 lần k-shell).
    """
    if GG.featNum is None:
        raise ValueError(
            "initialize_variables_f: GG.featNum chưa được set (cần số chiều feature)."
        )

    if V != GG.featNum:
        # Không raise cứng để linh hoạt, nhưng cảnh báo là hợp lý
        print(
            f"[WARN] initialize_variables_f: V={V} khác GG.featNum={GG.featNum}. "
            "Trong node-based encoding, V nên bằng số feature."
        )

    K = M + V

    # --- Random binary initialization for node-based chromosome ---
    # f[:, :V] : z ∈ {0,1}^V
    f = np.random.randint(0, 2, size=(N, V), dtype=int)

    featIdx_list = []

    # --- Evaluate each individual with Algorithm 1 (sẽ implement trong evaluate_objective_f) ---
    for i in range(N):
        z = f[i, :V].astype(int)

        # Tránh trường hợp subset rỗng -> assign worst, nhưng cũng nên ép ít nhất 1 feature
        if not z.any():
            j = np.random.randint(0, V)
            z[j] = 1
            f[i, :V] = z

        # obj_vals: [f1, f2], feat_mask: F(z) (bool mask size = featNum)
        obj_vals, feat_mask = evaluate_objective_f(M, z)
        obj_vals = np.asarray(obj_vals, dtype=float).ravel()

        if obj_vals.size != M:
            raise ValueError(
                f"initialize_variables_f: evaluate_objective_f cần {M} "
                f"objective, nhưng nhận được {obj_vals.size}"
            )

        # Lần đầu thêm cột objective cho toàn bộ population
        if f.shape[1] < K:
            extra = np.zeros((N, M), dtype=float)
            f = np.hstack([f.astype(float), extra])

        f[i, V:K] = obj_vals
        featIdx_list.append(np.asarray(feat_mask, dtype=bool).ravel())

    featIdx_f = np.vstack(featIdx_list)

    return f, featIdx_f
