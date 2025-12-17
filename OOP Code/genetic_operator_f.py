import numpy as np

from CXoperate import CXoperate
from decodeNet import decodeNet
from evaluate_objective_f import evaluate_objective_f

def genetic_operator_f(parent_chromosome, M, V, px, pm, templateAdj):
    """
    Parameters
    ----------
    parent_chromosome : np.ndarray, shape (N, V+M+...)
        Current parent population (at least V gene columns).
    M : int
        Number of objectives.
    V : int
        Number of decision variables (length of chromosome part).
    px : float
        Crossover probability.
    pm : float
        Mutation probability.
    templateAdj : array-like
        Adjacency template used in decodeNet.

    Returns
    -------
    f : np.ndarray, shape (N, V+M)
        Offspring chromosomes: [genes, objectives].
    featIdx_f : np.ndarray, shape (N, n_features_full)
        Feature masks corresponding to each offspring.
    """
    parent_chromosome = np.asarray(parent_chromosome, dtype=float)
    N, total_cols = parent_chromosome.shape

    if total_cols < V:
        raise ValueError(
            f"genetic_operator_f: parent_chromosome has {total_cols} columns, "
            f"but V = {V}."
        )

    # Only pass gene part (first V columns) to crossover/mutation
    child_chromosome = CXoperate(parent_chromosome[:, :V], px, pm)

    child_chromosome = np.asarray(child_chromosome, dtype=float)
    N_child, V_child = child_chromosome.shape
    if V_child != V:
        raise ValueError(
            f"CXoperate must return array with {V} columns, got {V_child}."
        )

    # Prepare output: [genes (V), objectives (M)]
    f = np.zeros((N_child, V + M), dtype=float)
    f[:, :V] = child_chromosome

    featIdx_list = []

    for i in range(N_child):
        indiv_net = decodeNet(child_chromosome[i, :V], templateAdj)

        obj_vals, feat_mask = evaluate_objective_f(M, indiv_net)
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
