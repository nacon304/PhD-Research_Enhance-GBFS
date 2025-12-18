# sequential_strategies.py
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def eval_subset_acc(X, y, feat_set, n_neighbors=5, cv=5, use_cv=True, random_state=42):
    feat_set = np.asarray(list(feat_set), dtype=int)
    if feat_set.size == 0:
        return 0.0

    X_sub = X[:, feat_set]
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    if use_cv:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        scores = []
        for tr, te in skf.split(X_sub, y):
            knn.fit(X_sub[tr], y[tr])
            pred = knn.predict(X_sub[te])
            scores.append(accuracy_score(y[te], pred))
        return float(np.mean(scores))
    else:
        knn.fit(X_sub, y)
        pred = knn.predict(X_sub)
        return float(accuracy_score(y, pred))

def buddy_sfs(
    X, y, S0, C_matrix, R_matrix,
    max_buddy_per_core=1,
    n_neighbors=5, cv=5,
    lam_red=0.5,
    random_state=42
):
    S_core = list(S0)
    S = set(S_core)
    all_features = np.arange(X.shape[1], dtype=int)

    current_acc = eval_subset_acc(X, y, S, n_neighbors, cv, True, random_state) if len(S) > 0 else 0.0

    for s in S_core:
        candidates, scores = [], []
        for f in all_features:
            if f in S:
                continue
            comp_sf = C_matrix[s, f]
            red_sf  = R_matrix[s, f]
            score_sf = comp_sf - lam_red * red_sf
            candidates.append(f)
            scores.append(score_sf)

        candidates = np.array(candidates, dtype=int)
        scores = np.array(scores, dtype=float)
        order = np.argsort(-scores)

        added = 0
        for idx in order:
            if added >= max_buddy_per_core:
                break
            f = candidates[idx]
            acc_new = eval_subset_acc(X, y, list(S) + [f], n_neighbors, cv, True, random_state)
            if acc_new > current_acc:
                S.add(f)
                current_acc = acc_new
                added += 1

    return np.array(sorted(S)), current_acc
