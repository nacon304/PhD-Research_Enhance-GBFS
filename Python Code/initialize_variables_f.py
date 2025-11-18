import numpy as np

from decodeNet import decodeNet
from evaluate_objective_f import evaluate_objective_f

def initialize_variables_f(N, M, V, templateAdj):
    """
    Parameters
    ----------
    N : int
        Population size.
    M : int
        Number of objectives.
    V : int
        Length of feature-edge chromosome (must = featNum * kNeigh).

    Returns
    -------
    f : np.ndarray, shape (N, V+M)
        Population with:
        - columns 0..V-1: binary chromosome
        - columns V..V+M-1: objective values
    featIdx_f : np.ndarray, shape (N, n_features_full)
        Feature mask cho từng cá thể.
    """
    K = M + V

    f = np.random.randint(0, 2, size=(N, V), dtype=int)

    featIdx_list = []

    for i in range(N):
        indiv_net = decodeNet(f[i, :V], templateAdj)

        obj_vals, feat_mask = evaluate_objective_f(M, indiv_net)
        obj_vals = np.asarray(obj_vals, dtype=float).ravel()

        if obj_vals.size != M:
            raise ValueError(
                f"initialize_variables_f: evaluate_objective_f need {M} "
                f"objectives, but got {obj_vals.size}"
            )

        if f.shape[1] < K:
            # first time: add columns for the entire population
            extra = np.zeros((N, M), dtype=float)
            f = np.hstack([f.astype(float), extra])

        f[i, V:K] = obj_vals
        featIdx_list.append(np.asarray(feat_mask, dtype=bool).ravel())

    featIdx_f = np.vstack(featIdx_list)

    return f, featIdx_f
