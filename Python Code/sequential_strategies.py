import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def eval_subset_acc(X, y, feat_set, n_neighbors=5, cv=5, use_cv=True, random_state=0):
    """
    Đánh giá accuracy của subset feat_set trên (X, y).
    feat_set: list/array index feature.
    """
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


def comp_to_set(f, S, C, mode="max"):
    """Complementary của feature f so với tập S, dùng C_matrix."""
    S = list(S)
    if len(S) == 0:
        return 0.0
    vals = C[f, S]
    if mode == "max":
        return float(np.max(vals))
    elif mode == "mean":
        return float(np.mean(vals))
    else:
        raise ValueError("Unknown mode in comp_to_set")


def red_to_set(f, S, R, mode="max"):
    """Redundancy của f so với tập S, dùng R = GG.Zout hoặc tương tự."""
    S = list(S)
    if len(S) == 0:
        return 0.0
    vals = R[f, S]
    if mode == "max":
        return float(np.max(vals))
    elif mode == "mean":
        return float(np.mean(vals))
    else:
        raise ValueError("Unknown mode in red_to_set")

def sfs_acc_only(X, y, S0, max_add=10, n_neighbors=5, cv=5):
    """
    Sequential Forward Selection bắt đầu từ S0, score = ΔAcc.
    """
    S = set(S0)
    current_acc = eval_subset_acc(X, y, S, n_neighbors, cv) if len(S) > 0 else 0.0

    all_features = np.arange(X.shape[1], dtype=int)

    for _ in range(max_add):
        best_f = None
        best_acc = current_acc

        for f in all_features:
            if f in S:
                continue
            acc_f = eval_subset_acc(X, y, list(S) + [f], n_neighbors, cv)
            if acc_f > best_acc:
                best_acc = acc_f
                best_f = f

        if best_f is None:
            break  # không còn feature nào cải thiện được nữa

        S.add(best_f)
        current_acc = best_acc

    return np.array(sorted(S)), current_acc

def sfs_acc_comp_red(
    X, y, S0, C_matrix, R_matrix,
    max_add=10, n_neighbors=5, cv=5,
    alpha=1.0, beta=0.5, gamma=0.5
):
    """
    Sequential search với score tổng hợp:
        Score = α ΔAcc + β Comp - γ Red
    """
    S = set(S0)
    current_acc = eval_subset_acc(X, y, S, n_neighbors, cv) if len(S) > 0 else 0.0

    all_features = np.arange(X.shape[1], dtype=int)

    for _ in range(max_add):
        best_f = None
        best_score = -1e9
        best_acc_new = current_acc

        for f in all_features:
            if f in S:
                continue

            # 1) ΔAcc
            acc_new = eval_subset_acc(X, y, list(S) + [f], n_neighbors, cv)
            delta_acc = acc_new - current_acc

            # 2) Complementary & Redundancy
            comp_f = comp_to_set(f, S, C_matrix, mode="max")
            red_f  = red_to_set(f, S, R_matrix, mode="max")

            score_f = alpha * delta_acc + beta * comp_f - gamma * red_f

            if score_f > best_score:
                best_score = score_f
                best_f = f
                best_acc_new = acc_new

        if best_f is None or best_score <= 0:
            break

        S.add(best_f)
        current_acc = best_acc_new

    return np.array(sorted(S)), current_acc

def sfs_prefilter_comp(
    X, y, S0, C_matrix, R_matrix,
    max_add=10, n_neighbors=5, cv=5,
    tau_comp=0.0, tau_red=1.0
):
    """
    Two-phase:
      1) Pre-filter candidate based on complementary & redundancy wrt S0.
      2) SFS ΔAcc-only trên tập candidate.
    """
    S = set(S0)
    all_features = np.arange(X.shape[1], dtype=int)

    # ---- Phase 1: pre-filter candidates ----
    candidates = []
    for f in all_features:
        if f in S:
            continue
        comp_f = comp_to_set(f, S, C_matrix, mode="max")
        red_f  = red_to_set(f, S, R_matrix, mode="max")
        if comp_f > tau_comp and red_f < tau_red:
            candidates.append(f)

    candidates = np.array(candidates, dtype=int)

    current_acc = eval_subset_acc(X, y, S, n_neighbors, cv) if len(S) > 0 else 0.0

    # ---- Phase 2: classic SFS trên candidates ----
    for _ in range(max_add):
        best_f = None
        best_acc = current_acc

        for f in candidates:
            if f in S:
                continue
            acc_f = eval_subset_acc(X, y, list(S) + [f], n_neighbors, cv)
            if acc_f > best_acc:
                best_acc = acc_f
                best_f = f

        if best_f is None:
            break

        S.add(best_f)
        current_acc = best_acc

    return np.array(sorted(S)), current_acc, candidates

def buddy_sfs(
    X, y, S0, C_matrix, R_matrix,
    max_buddy_per_core=1,
    n_neighbors=5, cv=5,
    lam_red=0.5
):
    """
    Mỗi core feature s ∈ S0 cố gắng thêm max_buddy_per_core buddy
    có complementary cao và redundancy thấp.
    """
    S_core = list(S0)
    S = set(S_core)
    all_features = np.arange(X.shape[1], dtype=int)

    current_acc = eval_subset_acc(X, y, S, n_neighbors, cv) if len(S) > 0 else 0.0

    for s in S_core:
        # candidate cho core s: f không thuộc S
        candidates = []
        scores = []
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

        # chọn theo thứ tự score_sf giảm dần
        order = np.argsort(-scores)
        added = 0

        for idx in order:
            if added >= max_buddy_per_core:
                break
            f = candidates[idx]
            # chỉ thêm nếu acc tăng (hoặc không giảm quá nhiều)
            acc_new = eval_subset_acc(X, y, list(S) + [f], n_neighbors, cv)
            if acc_new > current_acc:
                S.add(f)
                current_acc = acc_new
                added += 1

    return np.array(sorted(S)), current_acc
