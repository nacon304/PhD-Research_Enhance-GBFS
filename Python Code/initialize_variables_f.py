# initialize_variables_f.py
import numpy as np

def initialize_variables_f(N, M, V, templateAdj, evaluator=None):
    K = M + V

    genes = np.random.randint(0, 2, size=(N, V), dtype=int)

    if evaluator is None:
        # fallback serial (giữ lại nếu muốn)
        from decodeNet import decodeNet
        from evaluate_objective_f import evaluate_objective_f

        f = genes.astype(float)
        extra = np.zeros((N, M), dtype=float)
        f = np.hstack([f, extra])

        featIdx_list = []
        for i in range(N):
            indiv_net = decodeNet(genes[i, :], templateAdj)
            obj_vals, feat_mask = evaluate_objective_f(M, indiv_net)
            f[i, V:K] = np.asarray(obj_vals, dtype=float).ravel()
            featIdx_list.append(np.asarray(feat_mask, dtype=bool).ravel())

        featIdx_f = np.vstack(featIdx_list)
        return f, featIdx_f

    # ---- parallel path ----
    F, featIdx_f = evaluator.evaluate_batch(genes)
    f = np.hstack([genes.astype(float), F.astype(float)])
    return f, featIdx_f
