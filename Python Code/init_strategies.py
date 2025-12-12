import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

def _neighbors_from_adj(A, tol=0.0):
    """
    Tạo list các đỉnh kề từ ma trận adjacency A.
    neighbors[i] = np.array các j sao cho A[i, j] > tol
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    neighbors = []
    for i in range(n):
        neigh_i = np.where(A[i] > tol)[0]
        neighbors.append(neigh_i.astype(int))
    return neighbors


def init_graph_local_threshold(
    Z,
    k_min=1,
    quantile=0.8,
    exclude_self=True,
    symmetrize=True,
):
    """
    Xây đồ thị dựa trên threshold riêng cho từng đỉnh:
      - Với mỗi feature i, chọn các cạnh có sim >= quantile của hàng i.
      - Nếu số cạnh < k_min thì bù thêm top k_min theo sim.

    Parameters
    ----------
    Z : np.ndarray, shape (n_features, n_features)
        Ma trận similarity (ví dụ GG.Zout).
    k_min : int
        Số neighbor tối thiểu của mỗi đỉnh.
    quantile : float in (0,1)
        Ngưỡng quantile cho mỗi hàng, ví dụ 0.8 = top 20% similarity.
    exclude_self : bool
        Nếu True thì bỏ cạnh (i, i).
    symmetrize : bool
        Nếu True thì A = max(A, A^T) để đồ thị vô hướng.

    Returns
    -------
    A : np.ndarray, shape (n_features, n_features)
        Ma trận adjacency (0 = không có cạnh).
    neighbors : list[np.ndarray]
        neighbors[i] = list các đỉnh kề với i (chỉ số feature).
    """
    Z = np.asarray(Z, dtype=float)
    n = Z.shape[0]
    A = np.zeros_like(Z, dtype=float)

    for i in range(n):
        row = Z[i].copy()
        if exclude_self:
            row[i] = 0.0

        positive = row[row > 0]
        if positive.size == 0:
            continue

        tau_i = np.quantile(positive, quantile)
        neigh = np.where(row >= tau_i)[0]
        print(tau_i)
        print(neigh)

        if exclude_self:
            neigh = neigh[neigh != i]

        # đảm bảo ít nhất k_min neighbor
        if neigh.size < k_min:
            order = np.argsort(-row)  # giảm dần
            if exclude_self:
                order = order[order != i]
            neigh = order[:k_min]

        for j in neigh:
            if i == j:
                continue
            A[i, j] = Z[i, j]

    if symmetrize:
        A = np.maximum(A, A.T)

    neighbors = _neighbors_from_adj(A)
    return A, neighbors


def init_graph_fisher_redundancy_degree(
    Z,
    fisher_scores,
    k_min=1,
    k_max=20,
    exclude_self=True,
    symmetrize=True,
):
    """
    Mỗi đỉnh i có số neighbor deg_i phụ thuộc:
      - Fisher score f_i (cao -> nhiều cạnh)
      - Redundancy r_i (cao -> ít cạnh)

    deg_i = k_min + (k_max - k_min) * [0.5*f_norm_i + 0.5*(1 - r_norm_i)]

    Parameters
    ----------
    Z : np.ndarray, (n, n)
        Similarity matrix.
    fisher_scores : np.ndarray, (n,)
        Fisher score cho từng feature (trước khi scale).
    k_min, k_max : int
        Số cạnh min/max cho mỗi đỉnh.
    exclude_self, symmetrize : bool

    Returns
    -------
    A : np.ndarray, (n, n)
        Adjacency matrix.
    degs : np.ndarray, (n,)
        Số neighbor đã chọn cho mỗi đỉnh (trước khi symmetrize).
    neighbors : list[np.ndarray]
        neighbors[i] theo A (sau khi symmetrize nếu symmetrize=True).
    """
    Z = np.asarray(Z, dtype=float)
    fisher_scores = np.asarray(fisher_scores, dtype=float)
    n = Z.shape[0]
    assert fisher_scores.shape[0] == n

    # redundancy r_i = mean similarity với các đỉnh khác
    R = np.abs(Z.copy())
    if exclude_self:
        np.fill_diagonal(R, 0.0)
    r_i = R.mean(axis=1)

    # normalize fisher và redundancy về [0,1]
    f_min, f_max = fisher_scores.min(), fisher_scores.max()
    if f_max - f_min < 1e-12:
        f_norm = np.full_like(fisher_scores, 0.5, dtype=float)
    else:
        f_norm = (fisher_scores - f_min) / (f_max - f_min)

    r_min, r_max = r_i.min(), r_i.max()
    if r_max - r_min < 1e-12:
        r_norm = np.full_like(r_i, 0.5, dtype=float)
    else:
        r_norm = (r_i - r_min) / (r_max - r_min)

    score = 0.5 * f_norm + 0.5 * (1.0 - r_norm)

    k_max = min(k_max, n - 1)
    degs_float = k_min + (k_max - k_min) * score
    degs = np.clip(np.round(degs_float).astype(int), k_min, k_max)

    A = np.zeros_like(Z, dtype=float)
    for i in range(n):
        row = Z[i].copy()
        if exclude_self:
            row[i] = 0.0

        order = np.argsort(-row)  # giảm dần
        if exclude_self:
            order = order[order != i]

        ki = int(degs[i])
        if ki <= 0 or order.size == 0:
            continue

        ki = min(ki, order.size)
        neigh = order[:ki]
        for j in neigh:
            A[i, j] = Z[i, j]

    if symmetrize:
        A = np.maximum(A, A.T)

    neighbors = _neighbors_from_adj(A)
    return A, degs, neighbors


def init_graph_mst_plus_local(
    Z,
    extra_k=2,
    exclude_self=True,
    symmetrize=True,
):
    """
    Xây đồ thị với 2 tầng:
      1) MST backbone (trên distance = 1 - similarity).
      2) Mỗi đỉnh được thêm extra_k cạnh local mạnh nhất
         (không trùng cạnh MST).

    Parameters
    ----------
    Z : np.ndarray, (n, n)
        Ma trận similarity.
    extra_k : int
        Số neighbor local thêm vào mỗi đỉnh ngoài MST.
    exclude_self, symmetrize : bool

    Returns
    -------
    A : np.ndarray, (n, n)
        Adjacency matrix sau khi thêm MST + local edges.
    neighbors : list[np.ndarray]
        neighbors[i] theo A (sau khi symmetrize nếu symmetrize=True).
    """
    Z = np.asarray(Z, dtype=float)
    n = Z.shape[0]

    # 1) MST trên distance = 1 - similarity
    D = 1.0 - Z
    D[D < 0.0] = 0.0

    mst_sparse = minimum_spanning_tree(D)
    mst = mst_sparse.toarray()
    # Symmetric MST (vô hướng)
    mst = np.maximum(mst, mst.T)

    A = np.zeros_like(Z, dtype=float)

    # Thêm cạnh MST với trọng số similarity
    i_idx, j_idx = np.where(mst > 0)
    for i, j in zip(i_idx, j_idx):
        if i == j:
            continue
        A[i, j] = Z[i, j]
        A[j, i] = Z[j, i]

    # 2) Thêm extra_k local neighbors mạnh nhất cho mỗi đỉnh
    for i in range(n):
        row = Z[i].copy()
        if exclude_self:
            row[i] = 0.0

        # Loại bỏ các node đã được nối bởi MST
        already = np.where(A[i] > 0)[0]
        if already.size > 0:
            row[already] = 0.0

        if not np.any(row > 0):
            continue

        order = np.argsort(-row)
        if exclude_self:
            order = order[order != i]

        ki = min(extra_k, order.size)
        neigh_extra = order[:ki]

        for j in neigh_extra:
            if i == j:
                continue
            A[i, j] = Z[i, j]
            A[j, i] = Z[j, i]

    if symmetrize:
        A = np.maximum(A, A.T)

    neighbors = _neighbors_from_adj(A)
    return A, neighbors


def init_graph_probabilistic(
    Z,
    fisher_scores,
    k_min=1,
    k_max=10,
    beta=2.0,
    exclude_self=True,
    symmetrize=True,
    random_state=None,
):
    """
    Với mỗi đỉnh i:
      - Độ lớn deg_i = k_min + (k_max - k_min) * f_norm_i  (fisher lớn -> nhiều cạnh).
      - Chọn deg_i neighbor bằng cách sample không thay thế với xác suất:
            p_ij ∝ (sim_ij ** beta) * g(f_i, f_j)
        với g(f_i, f_j) ~ 0.5*(f_norm_i + f_norm_j).
      - Nếu tất cả weight bằng 0 thì fallback sang lấy top-k theo similarity.

    Parameters
    ----------
    Z : np.ndarray, (n, n)
        Similarity matrix.
    fisher_scores : np.ndarray, (n,)
        Fisher scores.
    k_min, k_max : int
        Min/max degree mỗi đỉnh.
    beta : float
        Độ "sắc" của phân phối theo similarity (beta lớn -> ưu tiên cạnh rất mạnh).
    exclude_self, symmetrize : bool
    random_state : int or None
        Seed cho reproducibility.

    Returns
    -------
    A : np.ndarray, (n, n)
        Adjacency matrix.
    neighbors : list[np.ndarray]
        neighbors[i] theo A (sau khi symmetrize nếu symmetrize=True).
    """
    Z = np.asarray(Z, dtype=float)
    fisher_scores = np.asarray(fisher_scores, dtype=float)
    n = Z.shape[0]
    assert fisher_scores.shape[0] == n

    rng = np.random.default_rng(random_state)

    # normalize Fisher
    f_min, f_max = fisher_scores.min(), fisher_scores.max()
    if f_max - f_min < 1e-12:
        f_norm = np.full_like(fisher_scores, 0.5, dtype=float)
    else:
        f_norm = (fisher_scores - f_min) / (f_max - f_min)

    k_max = min(k_max, n - 1)
    A = np.zeros_like(Z, dtype=float)

    indices_all = np.arange(n)

    for i in range(n):
        row = Z[i].copy()
        if exclude_self:
            row[i] = 0.0

        # degree mục tiêu cho node i
        k_i_float = k_min + (k_max - k_min) * f_norm[i]
        k_i = int(round(k_i_float))
        k_i = max(k_min, min(k_i, n - 1))

        if k_i <= 0:
            continue

        vals = np.maximum(row, 0.0)

        # Nếu tất cả weight = 0, fallback sang top-k
        if not np.any(vals > 0):
            order = np.argsort(-row)
            if exclude_self:
                order = order[order != i]
            neigh = order[:k_i]
        else:
            # weight w_j = (sim_ij^beta) * g(f_i, f_j)
            g_ij = 0.5 * (f_norm[i] + f_norm)  # vector over j
            w = (vals ** beta) * g_ij

            if exclude_self:
                mask_valid = indices_all != i
                cand_idx = indices_all[mask_valid]
                w = w[mask_valid]
            else:
                cand_idx = indices_all

            if not np.any(w > 0):
                # fallback top-k theo similarity
                row2 = row.copy()
                if exclude_self:
                    row2[i] = 0.0
                order = np.argsort(-row2)
                if exclude_self:
                    order = order[order != i]
                neigh = order[:k_i]
            else:
                probs = w / w.sum()
                k_eff = min(k_i, cand_idx.size)
                neigh = rng.choice(cand_idx, size=k_eff, replace=False, p=probs)

        for j in neigh:
            if i == j:
                continue
            A[i, j] = Z[i, j]

    if symmetrize:
        A = np.maximum(A, A.T)

    neighbors = _neighbors_from_adj(A)
    return A, neighbors
