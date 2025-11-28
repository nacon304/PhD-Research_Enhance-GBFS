import os
import numpy as np
import pandas as pd

from initialize_variables_f import initialize_variables_f
from non_domination_sort_mod import non_domination_sort_mod
from tournament_selection import tournament_selection
from genetic_operator_f import genetic_operator_f
from disEva import disEva
from replace_chromosome import replace_chromosome
from toChangeWeight import toChangeWeight
from helper import save_pareto_front_csv, log_population_metrics

import gbfs_globals as GG

def copy_of_en_nsga_2_mating_strategy(pop, gen, templateAdj, V_f, run_dir=None):
    """
    Parameters
    ----------
    pop : int
        Population size.
    gen : int
        Number of generations.
    templateAdj : array-like
        Adjacency template.
    V_f : int
        Length of decision-variable part for feature-side (V_f).
    run_dir : str, optional
        Directory to save run-specific files.

    Returns
    -------
    chromosome_output : np.ndarray
        [featIdx, objectives], shape (pop, featNum + M)
    """
    templateAdj = np.asarray(templateAdj, dtype=float).ravel()

    print(f"--GAP={GG.GAPGEN}")

    if pop is None or gen is None:
        raise ValueError(
            "NSGA-II: Please enter the population size and number of generations as input arguments."
        )

    if not isinstance(pop, (int, float)) or not isinstance(gen, (int, float)):
        raise TypeError("Both input arguments pop and gen should be numeric (integer).")

    pop = int(round(pop))
    gen = int(round(gen))

    if pop < 5:
        raise ValueError("Minimum population for running this function is 20")
    if gen < 5:
        raise ValueError("Minimum number of generations is 5")

    # ----- Initialize population (feature-side) -----
    chromosome_f, featIdx = initialize_variables_f(pop, GG.M, V_f, templateAdj)

    # Sort initial population by non-domination
    chromosome_f = non_domination_sort_mod(chromosome_f, GG.M, V_f)
    save_pareto_front_csv(chromosome_f, 0, V_f, run_dir)
    log_population_metrics(chromosome_f, 0, V_f, run_dir)

    # ----- Evolution process -----
    shareFlag = False              # flag for weight sharing
    arChive = np.empty((0, 0))     # archive of feature subsets
    kGenWeiArchive = np.array([])  # archive of "weight" evaluations
    kGenAccArchive = np.array([])  # archive of accuracies
    archivePop = pop

    for i in range(1, gen + 1):
        # --- Parent selection ---
        pool = int(round(pop / 2.0))
        tour = 2
        parent_chromosome_f = tournament_selection(chromosome_f, pool, tour)

        # --- Genetic operators ---
        px = 0.9
        pm = 0.01
        offspring_chromosome_f, featIdx_f = genetic_operator_f(
            parent_chromosome_f, GG.M, V_f, px, pm, templateAdj
        )

        # --- Build intermediate population (chromosome_f) ---
        main_pop_f = chromosome_f.shape[0]
        offspring_pop_f = offspring_chromosome_f.shape[0]
        cols_f = chromosome_f.shape[1]  # total columns of chromosome_f

        intermediate_chromosome_f = np.zeros(
            (main_pop_f + offspring_pop_f, cols_f), dtype=float
        )

        # parents
        intermediate_chromosome_f[:main_pop_f, :] = chromosome_f
        intermediate_chromosome_f[
            main_pop_f : main_pop_f + offspring_pop_f, : (GG.M + V_f)
        ] = offspring_chromosome_f[:, : (GG.M + V_f)]

        merge_f = intermediate_chromosome_f.copy()

        # --- Non-dominated sorting on intermediate population ---
        intermediate_chromosome_f = non_domination_sort_mod(
            intermediate_chromosome_f, GG.M, V_f
        )

        # --- Build intermediate featIdx population ---
        featIdx = np.asarray(featIdx, dtype=float)
        featIdx_f = np.asarray(featIdx_f, dtype=float)

        temp_featidx = np.vstack([featIdx, featIdx_f])

        temp_NDsort = merge_f[:, V_f : V_f + GG.M]

        intermediate_featIdx = np.hstack([temp_featidx, temp_NDsort])

        intermediate_featIdx = non_domination_sort_mod(
            intermediate_featIdx, GG.M, GG.featNum
        )

        # print(f"---- Generation {i} ----")
        # print(intermediate_chromosome_f)

        # --- Parent population in "feature space" (featIdx + objectives) ---
        parentPop = np.hstack(
            [featIdx, chromosome_f[:, V_f : V_f + GG.M]]
        )  # shape (pop, featNum + M)

        # mask_rank1 = parentPop[:, -2] == 1
        ranks = intermediate_chromosome_f[:, -2].astype(int)
        best_rank  = ranks.min()
        worst_rank = ranks.max()
        mask_rank1   = ranks == best_rank
        mask_rankEnd = ranks == worst_rank

        t_pos = np.abs(intermediate_chromosome_f[mask_rank1, V_f]).ravel()
        idx_pos_all = np.where(mask_rank1)[0]
        if t_pos.size > archivePop:
            sorted_idx = np.argsort(-t_pos)
            top_idx_small = sorted_idx[:archivePop]
            t_pos = t_pos[top_idx_small]
            idx_pos = idx_pos_all[top_idx_small]
        else:
            idx_pos = idx_pos_all
        
        t_neg = np.abs(intermediate_chromosome_f[mask_rankEnd, V_f]).ravel()
        idx_neg_all = np.where(mask_rankEnd)[0]
        if t_neg.size > archivePop:
            sorted_idx = np.argsort(-t_neg)
            top_idx_small = sorted_idx[:archivePop]
            t_neg = t_neg[top_idx_small]
            idx_neg = idx_neg_all[top_idx_small]
        else:
            idx_neg = idx_neg_all
        alpha_neg = 0.2
        t_neg = alpha_neg * t_neg
        
        if best_rank == worst_rank:
            t = t_pos
            selected_indices = idx_pos
        else:
            t = np.concatenate([+t_pos, -t_neg])
            selected_indices = np.concatenate([idx_pos, idx_neg])

        if kGenAccArchive.size == 0:
            kGenAccArchive = t.copy()
        else:
            kGenAccArchive = np.concatenate([kGenAccArchive, t])

        # tmpArc = parentPop[selected_indices, :GG.featNum]
        tmpArc = intermediate_chromosome_f[selected_indices, :GG.featNum]
        # print("Selected indices:", selected_indices)
        # print("Temporary archive:", tmpArc)

        if arChive.size == 0:
            arChive = tmpArc.copy()
        else:
            arChive = np.vstack([arChive, tmpArc])

        tempkGenArchive = disEva(tmpArc != 0)

        tempkGenArchive = np.asarray(tempkGenArchive).ravel()
        if kGenWeiArchive.size == 0:
            kGenWeiArchive = tempkGenArchive.copy()
        else:
            kGenWeiArchive = np.concatenate([kGenWeiArchive, tempkGenArchive])

        featIdx_new = replace_chromosome(intermediate_featIdx, GG.M, GG.featNum, pop)

        featIdx = (np.asarray(featIdx_new)[:, :GG.featNum] != 0)

        chromosome_f = replace_chromosome(intermediate_chromosome_f, GG.M, V_f, pop)
        save_pareto_front_csv(chromosome_f, i, V_f, run_dir)
        log_population_metrics(chromosome_f, i, V_f, run_dir)

        leapGen = 1
        gapGen = GG.GAPGEN

        if i >= leapGen:
            shareFlag = True

        if shareFlag and (gapGen is not None) and (i % gapGen == 0):
            toChangeWeight(kGenWeiArchive, kGenAccArchive, arChive)

            length2 = kGenWeiArchive.size
            if GG.assiNumInside is None:
                GG.assiNumInside = [length2]
            else:
                GG.assiNumInside.append(length2)

            # reset archive
            kGenWeiArchive = np.array([])
            kGenAccArchive = np.array([])
            arChive = np.empty((0, GG.featNum))

    chromosome_f2s = np.hstack([featIdx.astype(float), chromosome_f[:, V_f : V_f + GG.M]])

    chromosome_output = chromosome_f2s
    return chromosome_output
