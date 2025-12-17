import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

import gbfs_globals as GG
from kshell_2 import kshell_2

def evaluate_objective_f(M, individual_adj):
    """
    Parameters
    ----------
    M : int
        Number of objectives (should be 2).
    individual_adj : np.ndarray
        Adjacency matrix representing the individual network.

    Returns
    -------
    f : np.ndarray, shape (M,)
        Objective values [f1, f2].
        f1 = mean negative accuracy over 5-fold CV (to be minimized),
        f2 = number of selected features.
    featIdx_full : np.ndarray, shape (n_features,)
        Boolean mask for selected features over full original feature space.
    """
    corrMatrix = np.asarray(individual_adj, dtype=float)

    if not np.any(corrMatrix):
        if GG.data is None:
            raise ValueError("evaluate_objective_f: global 'data' is not set.")
        n_features = GG.data.shape[1]
        f = np.array([np.inf, np.inf], dtype=float)
        featIdx_full = np.ones(n_features, dtype=bool)
    else:
        featIdx = kshell_2(corrMatrix)
        featIdx = np.asarray(featIdx).astype(int).ravel()

        if GG.trData is None or GG.trLabel is None:
            raise ValueError(
                "evaluate_objective_f: global 'trData' or 'trLabel' is not set."
            )

        # 5-fold cross-validation on training data
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
            F1.append(-acc)  # negative accuracy (to be minimized)

        f1 = float(np.mean(F1))
        f2 = float(len(featIdx))  # number of selected features
        f = np.array([f1, f2], dtype=float)

        # map to full feature mask (over full 'data' space)
        if GG.data is None:
            raise ValueError("evaluate_objective_f: global 'data' is not set.")
        n_features = GG.data.shape[1]
        featIdx_full = np.zeros(n_features, dtype=bool)
        featIdx_full[featIdx] = True

    if f.size != M:
        raise ValueError(
            "The number of objective values does not match your input M. "
            "Kindly check evaluate_objective_f."
        )

    return f, featIdx_full
