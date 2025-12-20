import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

import gbfs_globals as GG
from kshell_2 import kshell_2
from sequential_strategies import apply_kshell_sequential

def evaluate_objective_f_internal(featIdx):
    Nfold = 5
    td = np.asarray(GG.trData, dtype=float)[:, featIdx]
    tl = np.asarray(GG.trLabel).ravel()

    skf = StratifiedKFold(n_splits=Nfold, shuffle=True, random_state=42)
    F1 = []

    for train_index, test_index in skf.split(td, tl):
        X_train = td[train_index, :]
        y_train = tl[train_index]
        X_test = td[test_index, :]
        y_test = tl[test_index]

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        plabel = knn.predict(X_test)

        acc = np.mean(plabel == y_test)
        F1.append(1-acc)

    f1 = float(np.mean(F1))
    f2 = float(len(featIdx) / GG.featNum)
    return np.array([f1, f2], dtype=float)

def evaluate_objective_f(M, individual_adj):
    corrMatrix = np.asarray(individual_adj, dtype=float)

    if not np.any(corrMatrix):
        if GG.data is None:
            raise ValueError("evaluate_objective_f: global 'data' is not set.")
        n_features = GG.data.shape[1]
        f = np.array([np.inf, np.inf], dtype=float)
        featIdx_full = np.ones(n_features, dtype=bool)
        return f, featIdx_full

    if GG.trData is None or GG.trLabel is None:
        raise ValueError("evaluate_objective_f: global 'trData' or 'trLabel' is not set.")

    featIdx_core = kshell_2(corrMatrix)
    featIdx_core = np.asarray(featIdx_core).astype(int).ravel()

    featIdx_ext = apply_kshell_sequential(featIdx_core)

    if not np.array_equal(featIdx_core, featIdx_ext):
        f_1 = evaluate_objective_f_internal(featIdx_core)
        f_2 = evaluate_objective_f_internal(featIdx_ext)

        def score(fv):
            return (1 - fv[0]) * 0.9 + 0.1 * fv[1]

        if score(f_1) < score(f_2):
            featIdx = featIdx_core
            f = f_1
        else:
            featIdx = featIdx_ext
            f = f_2
    else:
        featIdx = featIdx_core
        f = evaluate_objective_f_internal(featIdx)

    if GG.data is None:
        raise ValueError("evaluate_objective_f: global 'data' is not set.")
    n_features = GG.data.shape[1]
    featIdx_full = np.zeros(n_features, dtype=bool)
    featIdx_full[featIdx] = True

    if f.size != M:
        raise ValueError("Objective size != M in evaluate_objective_f")

    return f, featIdx_full