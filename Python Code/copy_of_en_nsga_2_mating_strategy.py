import numpy as np

from initialize_variables_f import initialize_variables_f
from non_domination_sort_mod import non_domination_sort_mod
from tournament_selection import tournament_selection
from genetic_operator_f import genetic_operator_f
from disEva import disEva
from replace_chromosome import replace_chromosome
from toChangeWeight import toChangeWeight

import gbfs_globals as GG

def copy_of_en_nsga_2_mating_strategy(pop, gen, templateAdj, V_f):
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

    M = 2  # number of objectives

    # ----- Initialize population (feature-side) -----
    chromosome_f, featIdx = initialize_variables_f(pop, M, V_f, templateAdj)
    # print("Initial chromosome_f:")
    # print(chromosome_f)
    # print("Initial featIdx:")
    # print(featIdx)

    # Sort initial population by non-domination (sync chromosome_f & featIdx)
    chromosome_f, featIdx = non_domination_sort_mod(chromosome_f, M, V_f, featIdx)
    # print("After initial non-dominated sorting:")
    # print(chromosome_f)
    # print(featIdx)

    # ----- Evolution process -----
    shareFlag = False
    arChive = np.empty((0, GG.featNum))  # archive of feature subsets
    kGenWeiArchive = np.array([])        # archive of "weight" evaluations
    kGenAccArchive = np.array([])        # archive of accuracies
    archivePop = pop

    for i in range(1, gen + 1):
        # print(f"\n=== Generation {i} ===")

        # --- Parent selection ---
        pool = int(round(pop / 2.0))
        tour = 2
        parent_chromosome_f = tournament_selection(chromosome_f, pool, tour)

        # --- Genetic operators ---
        px = 0.9
        pm = 0.01
        offspring_chromosome_f, featIdx_f = genetic_operator_f(
            parent_chromosome_f, M, V_f, px, pm, templateAdj
        )
        # print("Offspring chromosome_f:")
        # print(offspring_chromosome_f)
        # print("Offspring featIdx_f:")
        # print(featIdx_f)

        # --- Build intermediate population (chromosome_f) ---
        main_pop_f = chromosome_f.shape[0]
        offspring_pop_f = offspring_chromosome_f.shape[0]
        cols_f = chromosome_f.shape[1]  # total columns of chromosome_f

        intermediate_chromosome_f = np.zeros(
            (main_pop_f + offspring_pop_f, cols_f), dtype=float
        )

        # parents
        intermediate_chromosome_f[:main_pop_f, :] = chromosome_f
        # offspring: copy decision vars + objective values; rank & crowding will be updated
        intermediate_chromosome_f[
            main_pop_f : main_pop_f + offspring_pop_f, : (M + V_f)
        ] = offspring_chromosome_f[:, : (M + V_f)]

        # print("Intermediate chromosome_f BEFORE non-dominated sorting:")
        # print(intermediate_chromosome_f)

        # --- Build combined featIdx (parents + offspring) ---
        featIdx_par = np.asarray(featIdx, dtype=float)
        featIdx_off = np.asarray(featIdx_f, dtype=float)
        temp_featidx = np.vstack([featIdx_par, featIdx_off])

        # --- Single non-dominated sorting for both chromosome & featIdx ---
        intermediate_chromosome_f, temp_featidx_sorted = non_domination_sort_mod(
            intermediate_chromosome_f, M, V_f, temp_featidx
        )
        # print("Intermediate chromosome_f AFTER non-dominated sorting:")
        # print(intermediate_chromosome_f)
        # print("Intermediate featIdx AFTER non-dominated sorting:")
        # print(temp_featidx_sorted)

        # --- Parent population in "feature space" (featIdx + objectives + rank + dist) ---
        parentPop = np.hstack(
            [
                featIdx_par.astype(float),
                chromosome_f[:, V_f : V_f + M + 2],  # objectives + rank + crowding
            ]
        )  # shape (pop, featNum + M + 2)
        # print("ParentPop (features + objs + rank + dist):")
        # print(parentPop)

        mask_rank1 = parentPop[:, -2] == 1  # rank == 1
        t = np.abs(parentPop[mask_rank1, GG.featNum]).ravel()

        if t.size > archivePop:
            sorted_idx = np.argsort(-t)
            top_idx_small = sorted_idx[:archivePop]
            t = t[top_idx_small]

            rank1_indices = np.where(mask_rank1)[0]
            selected_indices = rank1_indices[top_idx_small]
        else:
            selected_indices = np.where(mask_rank1)[0]

        if kGenAccArchive.size == 0:
            kGenAccArchive = t.copy()
        else:
            kGenAccArchive = np.concatenate([kGenAccArchive, t])

        tmpArc = parentPop[selected_indices, :GG.featNum]

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

        # --- Environmental selection (NSGA-II) ---
        chromosome_f, featIdx = replace_chromosome(
            intermediate_chromosome_f, M, V_f, pop, featIdx=temp_featidx_sorted
        )
        featIdx = (featIdx != 0)
        # print("Selected chromosome_f for next generation:")
        # print(chromosome_f)
        # print(featIdx)

        # --- Adaptive weight sharing ---
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

    # ----- Output: [featIdx, objectives] -----
    chromosome_f2s = np.hstack(
        [featIdx.astype(float), chromosome_f[:, V_f : V_f + M]]
    )

    chromosome_output = chromosome_f2s
    return chromosome_output
