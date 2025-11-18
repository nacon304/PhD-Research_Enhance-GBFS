import numpy as np

def tournament_selection(chromosome, pool_size, tour_size):
    """
    Parameters
    ----------
    chromosome : np.ndarray, shape (pop, variables)
        Current population. Penultimate column = rank, last column = crowding distance.
    pool_size : int
        Size of the mating pool to select.
    tour_size : int
        Tournament size (number of individuals in each "tournament").

    Returns
    -------
    f : np.ndarray, shape (pool_size, variables)
        Selected individuals (mating pool).
    """
    chromosome = np.asarray(chromosome, dtype=float)
    pop, variables = chromosome.shape

    # penultimate column: rank, last column: crowding distance
    rank_col = variables - 2
    dist_col = variables - 1

    f = np.zeros((pool_size, variables), dtype=float)

    for i in range(pool_size):
        # ---- select 'tour_size' unique candidates ----
        candidates = []
        while len(candidates) < tour_size:
            c = int(round(np.random.rand() * pop))
            if c == 0:
                c = 1
            if c < 1:
                c = 1
            if c > pop:
                c = pop
            idx0 = c - 1
            if idx0 not in candidates:
                candidates.append(idx0)

        candidates = np.array(candidates, dtype=int)

        # ---- get rank and distance of candidates ----
        c_obj_rank = chromosome[candidates, rank_col]
        c_obj_distance = chromosome[candidates, dist_col]

        # ---- find candidate with smallest rank ----
        min_rank = np.min(c_obj_rank)
        min_candidate_indices = np.where(c_obj_rank == min_rank)[0]

        if min_candidate_indices.size != 1:
            # multiple individuals have the same rank -> choose the one with the largest distance
            distances_sub = c_obj_distance[min_candidate_indices]
            max_dist = np.max(distances_sub)
            max_candidate_indices = np.where(distances_sub == max_dist)[0]

            if max_candidate_indices.size == 0:
                r_idx = np.random.randint(0, min_candidate_indices.size)
                chosen_rel = r_idx
            else:
                chosen_rel = max_candidate_indices[0]

            chosen_idx = candidates[min_candidate_indices[chosen_rel]]
        else:
            chosen_idx = candidates[min_candidate_indices[0]]

        f[i, :] = chromosome[chosen_idx, :]

    return f
