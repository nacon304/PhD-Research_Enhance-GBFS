import numpy as np

def replace_chromosome(intermediate_chromosome, M, V, pop):
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

    Returns
    -------
    f : np.ndarray, shape (pop, V+M+2)
        New population after choosing based on rank + crowding distance.
    """
    intermediate_chromosome = np.asarray(intermediate_chromosome, dtype=float)
    N, m = intermediate_chromosome.shape

    rank_col = V + M   
    cd_col = V + M + 1    

    ranks = intermediate_chromosome[:, rank_col]
    idx_sort = np.argsort(ranks, kind="mergesort")
    sorted_chromosome = intermediate_chromosome[idx_sort, :]

    max_rank = int(np.max(intermediate_chromosome[:, rank_col]))

    f = np.zeros((pop, m), dtype=float)

    previous_index = 0  # 0..pop

    # ---- Add each front based on rank ----
    for r in range(1, max_rank + 1):
        # indices in sorted_chromosome with rank = r
        mask_r = sorted_chromosome[:, rank_col] == r
        indices_r = np.where(mask_r)[0]

        if indices_r.size == 0:
            continue

        current_index = int(indices_r[-1] + 1)  # +1 to be equivalent to 1-based

        if current_index > pop:
            remaining = pop - previous_index
            if remaining <= 0:
                return f

            temp_pop = sorted_chromosome[previous_index:current_index, :]

            # sort based on crowding distance in descending order
            temp_cd = temp_pop[:, cd_col]
            sort_cd_idx = np.argsort(-temp_cd, kind="mergesort")

            # take 'remaining' individuals with the largest crowding distance
            chosen_idx = sort_cd_idx[:remaining]

            f[previous_index: previous_index + remaining, :] = temp_pop[chosen_idx, :]
            return f

        elif current_index < pop:
            # copy entire front rank r
            f[previous_index:current_index, :] = sorted_chromosome[previous_index:current_index, :]

        else:  
            f[previous_index:current_index, :] = sorted_chromosome[previous_index:current_index, :]
            return f

        previous_index = current_index

    return f
