# sequential_strategies.py
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import gbfs_globals as GG

def comp_to_set(f, S, C, mode="max"):
    S = list(S)
    if len(S) == 0:
        return 0.0
    vals = C[f, S]
    return float(np.max(vals)) if mode == "max" else float(np.mean(vals))

def red_to_set(f, S, R, mode="max"):
    S = list(S)
    if len(S) == 0:
        return 0.0
    vals = R[f, S]
    vals = np.abs(1.0 - vals)
    return float(np.max(vals)) if mode == "max" else float(np.mean(vals))

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

def rc_greedy(S0, C_matrix, R_matrix, max_add=10, tau=0.3, comp_mode="max", red_mode="max"):
    """
    Greedy add features by:
        score(f|S) = comp(f,S) - tau * red(f,S)
    No eval_subset_acc.
    """
    S = set(np.asarray(S0, dtype=int).ravel().tolist())
    n_feat = R_matrix.shape[0]
    all_features = np.arange(n_feat, dtype=int)

    for _ in range(max_add):
        best_f = None
        best_score = -1e9

        for f in all_features:
            if f in S:
                continue
            comp_f = comp_to_set(f, S, C_matrix, mode=comp_mode)
            red_f = red_to_set(f, S, R_matrix, mode=red_mode)
            score_f = comp_f - tau * red_f
            if score_f > best_score:
                best_score = score_f
                best_f = f

        if best_f is None or best_score <= 0:
            break
        S.add(best_f)

    return np.array(sorted(S), dtype=int)

def apply_kshell_sequential(core_idx: np.ndarray) -> np.ndarray:
    """
    Called inside evaluate_objective_f after K-shell.
    Reads GG.kshell_seq_mode.
    """
    mode = getattr(GG, "kshell_seq_mode", "none")
    core_idx = np.asarray(core_idx, dtype=int).ravel()

    if mode in [None, "none", "normal"]:
        return core_idx

    if mode == "rc_greedy":
        if GG.C_matrix is None or GG.Zout is None:
            raise ValueError("apply_kshell_sequential: need GG.C_matrix and GG.Zout for rc_greedy.")
        max_add = int(getattr(GG, "kshell_max_add", 5))
        tau = float(getattr(GG, "rc_tau", 0.3))
        return rc_greedy(core_idx, GG.C_matrix, GG.Zout, max_add=max_add, tau=tau)

    return core_idx

def apply_post_sequential(
    S0: np.ndarray,
    mode: str,
    X: np.ndarray,
    y: np.ndarray,
    C_matrix: np.ndarray = None,
    R_matrix: np.ndarray = None,
    buddy_kwargs: dict = None,
):
    """
    Post sequential after final selected set from newtry_ms.

    Parameters
    ----------
    S0 : array-like
        Selected features from newtry_ms (after evolution).
    mode : {"normal","none","buddy"}
    X, y : training data used to evaluate buddy additions (CV inside buddy_sfs).
    C_matrix, R_matrix : needed for buddy.
    buddy_kwargs : dict of args forwarded to buddy_sfs

    Returns
    -------
    S_post : np.ndarray (sorted int)
    """
    mode = (mode or "normal").lower()
    S0 = np.asarray(S0, dtype=int).ravel()

    if mode in ["normal", "none"]:
        return S0

    if mode == "buddy":
        if C_matrix is None or R_matrix is None:
            raise ValueError("apply_post_sequential(buddy): need C_matrix and R_matrix.")
        if buddy_kwargs is None:
            buddy_kwargs = {}

        SB, _ = buddy_sfs(
            X, y,
            S0=S0,
            C_matrix=C_matrix,
            R_matrix=R_matrix,
            **buddy_kwargs
        )
        return np.asarray(SB, dtype=int).ravel()

    raise ValueError(f"Unknown post mode: {mode}")