import numpy as np

from initialize_variables_f import initialize_variables_f
from non_domination_sort_mod import non_domination_sort_mod
from tournament_selection import tournament_selection
from genetic_operator_f import genetic_operator_f
from disEva import disEva
from replace_chromosome import replace_chromosome
from toChangeWeight import toChangeWeight

import gbfs_globals as GG
from helper import log_population_metrics, save_pareto_front_csv

from gbfs_parallel_eval import GBFSParallelEvaluator


def copy_of_en_nsga_2_mating_strategy(
    pop,
    gen,
    templateAdj,
    V_f,
    run_dir,
    n_jobs=8,
    parallel_backend="thread",   # "thread" | "process"
):
    """
    Returns
    -------
    chromosome_output : np.ndarray
        [featIdx, objectives], shape (pop, featNum + M)
    """
    templateAdj = np.asarray(templateAdj, dtype=float).ravel()

    print(f"--GAP={GG.GAPGEN}")

    if pop is None or gen is None:
        raise ValueError("NSGA-II: Please enter pop and gen.")

    if not isinstance(pop, (int, float)) or not isinstance(gen, (int, float)):
        raise TypeError("pop and gen should be numeric.")

    pop = int(round(pop))
    gen = int(round(gen))

    # (optional) sửa message cho đúng
    if pop < 20:
        raise ValueError("Minimum population for running this function is 20")
    if gen < 5:
        raise ValueError("Minimum number of generations is 5")

    M = int(GG.M)  # number of objectives

    # ============================================================
    # Create ONE evaluator (ONE pool) reused for:
    #  - init population evaluation
    #  - offspring evaluation each generation
    # ============================================================
    with GBFSParallelEvaluator(M=M, templateAdj=templateAdj, n_jobs=n_jobs, backend=parallel_backend) as evaluator:

        # ----- Initialize population (parallel eval) -----
        chromosome_f, featIdx = initialize_variables_f(
            pop, M, V_f, templateAdj, evaluator=evaluator
        )

        # Sort initial population by non-domination (sync chromosome_f & featIdx)
        chromosome_f, featIdx = non_domination_sort_mod(chromosome_f, M, V_f, featIdx)
        save_pareto_front_csv(chromosome_f, 0, V_f, run_dir)
        log_population_metrics(chromosome_f, 0, V_f, run_dir)

        # ----- Evolution process -----
        shareFlag = False
        arChive = np.empty((0, GG.featNum))  # archive of feature subsets
        kGenWeiArchive = np.array([])        # archive of "weight" evaluations
        kGenAccArchive = np.array([])        # archive of accuracies
        archivePop = pop

        for i in range(1, gen + 1):

            # --- Parent selection ---
            pool = int(round(pop / 2.0))
            tour = 2
            parent_chromosome_f = tournament_selection(chromosome_f, pool, tour)

            # --- Genetic operators (parallel eval) ---
            px = 0.9
            pm = 0.01
            offspring_chromosome_f, featIdx_f = genetic_operator_f(
                parent_chromosome_f, M, V_f, px, pm, templateAdj, evaluator=evaluator
            )

            # --- Build intermediate population (chromosome_f) ---
            main_pop_f = chromosome_f.shape[0]
            offspring_pop_f = offspring_chromosome_f.shape[0]
            cols_f = chromosome_f.shape[1]

            intermediate_chromosome_f = np.zeros((main_pop_f + offspring_pop_f, cols_f), dtype=float)
            intermediate_chromosome_f[:main_pop_f, :] = chromosome_f
            intermediate_chromosome_f[main_pop_f: main_pop_f + offspring_pop_f, : (M + V_f)] = \
                offspring_chromosome_f[:, : (M + V_f)]

            # --- Build combined featIdx (parents + offspring) ---
            featIdx_par = np.asarray(featIdx, dtype=float)
            featIdx_off = np.asarray(featIdx_f, dtype=float)
            temp_featidx = np.vstack([featIdx_par, featIdx_off])

            # --- Single non-dominated sorting for both chromosome & featIdx ---
            intermediate_chromosome_f, temp_featidx_sorted = non_domination_sort_mod(
                intermediate_chromosome_f, M, V_f, temp_featidx
            )

            # --- Parent population in "feature space" (featIdx + objectives + rank + dist) ---
            parentPop = np.hstack(
                [
                    featIdx_par.astype(float),
                    chromosome_f[:, V_f : V_f + M + 2],  # objectives + rank + crowding
                ]
            )

            mask_rank1 = parentPop[:, -2] == 1
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
            arChive = tmpArc.copy() if arChive.size == 0 else np.vstack([arChive, tmpArc])

            tempkGenArchive = np.asarray(disEva(tmpArc != 0)).ravel()
            kGenWeiArchive = tempkGenArchive.copy() if kGenWeiArchive.size == 0 else \
                np.concatenate([kGenWeiArchive, tempkGenArchive])

            # --- Environmental selection (NSGA-II) ---
            chromosome_f, featIdx = replace_chromosome(
                intermediate_chromosome_f, M, V_f, pop, featIdx=temp_featidx_sorted
            )
            featIdx = (featIdx != 0)

            save_pareto_front_csv(chromosome_f, i, V_f, run_dir)
            log_population_metrics(chromosome_f, i, V_f, run_dir)

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

                kGenWeiArchive = np.array([])
                kGenAccArchive = np.array([])
                arChive = np.empty((0, GG.featNum))

        # ----- Output: [featIdx, objectives] -----
        chromosome_f2s = np.hstack([featIdx.astype(float), chromosome_f[:, : V_f + M]])
        return chromosome_f2s
