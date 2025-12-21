import numpy as np

def non_domination_sort_mod(x, M, V):
    """
    Parameters
    ----------
    x : np.ndarray, shape (N, >= V+M)
        Population matrix:
        - columns 0..V-1: decision variables
        - columns V..V+M-1: objective values (to be minimized)
        - rank / crowding distance: will be appended after V+M-1
    M : int
        Number of objectives.
    V : int
        Number of decision variables.

    Returns
    -------
    f : np.ndarray, shape (N, V+M+2)
        - 0..V-1: decision variables
        - V..V+M-1: objective values
        - V+M: rank
        - V+M+1: crowding distance
    """
    x = np.asarray(x, dtype=float)
    N, m = x.shape

    if m < V + M:
        raise ValueError(
            f"non_domination_sort_mod: x has {m} columns, "
            f"but need at least V+M = {V+M}"
        )

    num_cols = V + M + 2
    if m < num_cols:
        pad = np.zeros((N, num_cols - m))
        x_work = np.hstack([x, pad])
    else:
        x_work = x[:, :num_cols].copy()

    rank_col = V + M    
    dist_col = V + M + 1   

    # ===== Non-dominated sorting =====
    # individual[i]['n'] : number of individuals dominating i
    # individual[i]['p'] : list of individuals dominated by i
    individual = [{"n": 0, "p": []} for _ in range(N)]

    # F is a list of fronts, each front is a list of indices of individuals in that front
    F = [[]]

    # initialize first front
    for i in range(N):
        for j in range(N):
            dom_less = 0
            dom_equal = 0
            dom_more = 0
            for k in range(M):
                fi = x_work[i, V + k]
                fj = x_work[j, V + k]
                if fi < fj:
                    dom_less += 1
                elif fi == fj:
                    dom_equal += 1
                else:
                    dom_more += 1
            # j dominates i
            if dom_less == 0 and dom_equal != M:
                individual[i]["n"] += 1
            # i dominates j
            elif dom_more == 0 and dom_equal != M:
                individual[i]["p"].append(j)

        if individual[i]["n"] == 0:
            x_work[i, rank_col] = 1  # rank = 1
            F[0].append(i)           # first front

    # find subsequent fronts
    front = 0
    while len(F[front]) > 0:
        Q = []
        for p_idx in F[front]:
            for q in individual[p_idx]["p"]:
                individual[q]["n"] -= 1
                if individual[q]["n"] == 0:
                    x_work[q, rank_col] = front + 2
                    Q.append(q)
        front += 1
        F.append(Q)

    # ===== Sort by rank =====
    ranks = x_work[:, rank_col]
    # mergesort to maintain stable order
    index_of_fronts = np.argsort(ranks, kind="mergesort")
    sorted_based_on_front = x_work[index_of_fronts, :]

    # ===== Crowding distance =====
    z = np.zeros_like(sorted_based_on_front)
    current_index = 0

    for front_idx in range(len(F) - 1):
        n_front = len(F[front_idx])
        if n_front == 0:
            continue

        previous_index = current_index
        # get the segment corresponding to this front in sorted_based_on_front
        y = sorted_based_on_front[current_index: current_index + n_front, :].copy()
        current_index += n_front

        y[:, dist_col] = 0.0

        for obj_i in range(M):
            obj_col = V + obj_i
            obj_values = y[:, obj_col]
            # sort by this objective
            index_obj = np.argsort(obj_values, kind="mergesort")
            sorted_y = y[index_obj, :]

            f_max = sorted_y[-1, obj_col]
            f_min = sorted_y[0, obj_col]

            # mark boundary = Inf
            y[index_obj[0], dist_col] = np.inf
            y[index_obj[-1], dist_col] = np.inf

            if f_max - f_min == 0:
                y[:, dist_col] = np.inf
                continue

            # points in the middle add normalized distance
            for j_pos in range(1, len(index_obj) - 1):
                idx_mid = index_obj[j_pos]
                next_obj = sorted_y[j_pos + 1, obj_col]
                prev_obj = sorted_y[j_pos - 1, obj_col]
                contrib = (next_obj - prev_obj) / (f_max - f_min)

                if np.isfinite(y[idx_mid, dist_col]):
                    y[idx_mid, dist_col] += contrib

        z[previous_index: current_index, :] = y

    f = z
    return f
