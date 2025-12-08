import numpy as np


def kshell_2(Gadj0, nodes=None, node_weights=None):
    """
    Weighted k-shell decomposition + selection.

    Parameters
    ----------
    Gadj0 : array-like, shape (d, d)
        Weighted adjacency matrix (redundancy or complementary).
    nodes : array-like of int, optional
        List of node indices (trong [0..d-1]) tham gia k-shell.
        Nếu None -> dùng toàn bộ nodes [0..d-1].
    node_weights : array-like, shape (d,), optional
        Trọng số node w_i (relevance / combined relevance+complementarity).
        Nếu None -> tất cả w_i = 1.

    Returns
    -------
    selected : np.ndarray, 1D dtype=int
        Tập node được chọn:
        - tất cả node trong shell cao nhất,
        - cộng thêm 1 node có weight lớn nhất từ mỗi shell còn lại.
    """
    W = np.asarray(Gadj0, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("kshell_2: Gadj0 phải là ma trận vuông (d x d).")

    d = W.shape[0]

    if nodes is None:
        active = list(range(d))
    else:
        nodes = np.asarray(nodes, dtype=int).ravel()
        active = list(nodes)

    if node_weights is None:
        w_full = np.ones(d, dtype=float)
    else:
        w_full = np.asarray(node_weights, dtype=float).ravel()
        if w_full.size != d:
            raise ValueError(
                f"kshell_2: kích thước node_weights={w_full.size} "
                f"không khớp số node d={d}."
            )

    buckets = []
    eps = 1e-12

    # Lặp cho đến khi không còn node trong active
    while len(active) > 0:
        # weighted-degree cho từng node trong active
        degs = np.zeros(len(active), dtype=float)
        for idx, node in enumerate(active):
            # tổng trọng số cạnh đến các node active khác
            degs[idx] = w_full[node] * np.sum(W[node, active])

        # nếu mọi degree đều 0 -> tất cả active nằm trong 1 shell cuối
        if np.all(degs <= 0):
            buckets.append(np.array(active, dtype=int))
            break

        minD = degs.min()
        # shell hiện tại = tất cả node có deg gần minD
        shell_nodes = [
            active[i] for i in range(len(active)) if degs[i] <= minD + eps
        ]

        if len(shell_nodes) == 0:
            # đề phòng số học lạ, cho tất cả vào 1 shell
            buckets.append(np.array(active, dtype=int))
            break

        buckets.append(np.array(shell_nodes, dtype=int))

        # loại các node này khỏi active
        active = [node for node in active if node not in shell_nodes]

    if len(buckets) == 0:
        return np.array([], dtype=int)

    # Chọn:
    # - tất cả node trong bucket cuối (shell cao nhất)
    # - 1 node có weight lớn nhất từ mỗi bucket trước đó
    if len(buckets) == 1:
        selected = np.sort(buckets[0])
    else:
        last_shell = buckets[-1]
        selected_list = list(last_shell)

        for shell in buckets[:-1]:
            if shell.size == 0:
                continue
            best_node = shell[np.argmax(w_full[shell])]
            if best_node not in selected_list:
                selected_list.append(best_node)

        selected = np.array(sorted(selected_list), dtype=int)

    return selected
