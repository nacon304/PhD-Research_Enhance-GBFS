# genetic_operator_f.py
import numpy as np
from CXoperate import CXoperate

def genetic_operator_f(parent_chromosome, M, V, px, pm, templateAdj, evaluator=None):
    parent_chromosome = np.asarray(parent_chromosome, dtype=float)
    N, total_cols = parent_chromosome.shape
    if total_cols < V:
        raise ValueError(f"parent_chromosome has {total_cols} cols but V={V}")

    child = CXoperate(parent_chromosome[:, :V], px, pm)
    child = np.asarray(child, dtype=float)

    # ensure binary 0/1
    child01 = (child > 0.5).astype(int)

    if evaluator is None:
        from decodeNet import decodeNet
        from evaluate_objective_f import evaluate_objective_f

        f = np.zeros((child01.shape[0], V + M), dtype=float)
        f[:, :V] = child01

        featIdx_list = []
        for i in range(child01.shape[0]):
            indiv_net = decodeNet(child01[i, :], templateAdj)
            obj_vals, feat_mask = evaluate_objective_f(M, indiv_net)
            f[i, V:V+M] = np.asarray(obj_vals, dtype=float).ravel()
            featIdx_list.append(np.asarray(feat_mask, dtype=bool).ravel())

        featIdx_f = np.vstack(featIdx_list)
        return f, featIdx_f

    # ---- parallel path ----
    F, featIdx_f = evaluator.evaluate_batch(child01)
    f = np.hstack([child01.astype(float), F.astype(float)])
    return f, featIdx_f
