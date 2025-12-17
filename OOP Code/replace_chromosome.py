import numpy as np

def replace_chromosome(intermediate_chromosome, M, V, pop, featIdx):
    """
    Parameters
    ----------
    intermediate_chromosome : np.ndarray, shape (N, V+M+2)
        Population sau khi non_domination_sort_mod:
        - 0..V-1: decision variables
        - V..V+M-1: objective values
        - V+M: rank
        - V+M+1: crowding distance
    M : int
        Number of objectives.
    V : int
        Number of decision variables.
    pop : int
        Target population size.
    featIdx : np.ndarray or None, shape (N, F), optional
        Ma trận feature index tương ứng từng cá thể (cùng thứ tự hàng với
        intermediate_chromosome). Nếu None, hàm chỉ chọn cho chromosome.

    Returns
    -------
    f : np.ndarray, shape (pop, V+M+2)
        New population (chromosome) after selection.
    f_feat : np.ndarray, shape (pop, F)   (chỉ trả về nếu featIdx != None)
        Feature index được chọn tương ứng.
    """
    intermediate_chromosome = np.asarray(intermediate_chromosome, dtype=float)
    N, m = intermediate_chromosome.shape

    featIdx = np.asarray(featIdx)
    if featIdx.shape[0] != N:
        raise ValueError(
            "replace_chromosome: featIdx must have same number of rows as intermediate_chromosome"
        )

    rank_col = V + M       # rank
    cd_col   = V + M + 1   # crowding distance

    ranks = intermediate_chromosome[:, rank_col]
    idx_sort = np.argsort(ranks, kind="mergesort")
    sorted_chromosome = intermediate_chromosome[idx_sort, :]

    sorted_featIdx = featIdx[idx_sort, :]
    F = sorted_featIdx.shape[1]
    f_feat = np.zeros((pop, F), dtype=sorted_featIdx.dtype)

    max_rank = int(np.max(sorted_chromosome[:, rank_col]))

    f = np.zeros((pop, m), dtype=float)

    previous_index = 0  

    # ---- Add each front based on rank ----
    for r in range(1, max_rank + 1):
        mask_r    = (sorted_chromosome[:, rank_col] == r)
        indices_r = np.where(mask_r)[0]

        if indices_r.size == 0:
            continue

        current_index = int(indices_r[-1] + 1)  # +1 vì dùng kiểu [prev:current)

        if current_index > pop:
            remaining = pop - previous_index
            if remaining <= 0:
                if featIdx is None:
                    return f
                else:
                    return f, f_feat

            temp_pop = sorted_chromosome[previous_index:current_index, :]
            temp_cd  = temp_pop[:, cd_col]

            sort_cd_idx = np.argsort(-temp_cd, kind="mergesort")
            chosen_idx  = sort_cd_idx[:remaining]

            f[previous_index: previous_index + remaining, :] = temp_pop[chosen_idx, :]

            if featIdx is not None:
                temp_feat = sorted_featIdx[previous_index:current_index, :]
                f_feat[previous_index: previous_index + remaining, :] = temp_feat[chosen_idx, :]

            if featIdx is None:
                return f
            else:
                return f, f_feat

        elif current_index < pop:
            f[previous_index:current_index, :] = sorted_chromosome[previous_index:current_index, :]
            if featIdx is not None:
                f_feat[previous_index:current_index, :] = sorted_featIdx[previous_index:current_index, :]

        else:  # current_index == pop
            f[previous_index:current_index, :] = sorted_chromosome[previous_index:current_index, :]
            if featIdx is not None:
                f_feat[previous_index:current_index, :] = sorted_featIdx[previous_index:current_index, :]

            if featIdx is None:
                return f
            else:
                return f, f_feat

        previous_index = current_index

    if featIdx is None:
        return f
    else:
        return f, f_feat
