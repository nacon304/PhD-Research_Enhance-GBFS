import numpy as np

from copy_of_en_nsga_2_mating_strategy import copy_of_en_nsga_2_mating_strategy
import gbfs_globals as GG

def newtry_ms(inputAdj, pop=20, times=40):
    """
    Parameters
    ----------
    inputAdj : array-like
    pop : int
        Population size
    times : int
        Number of generations

    Returns
    -------
    featidx : np.ndarray, shape (featNum,)
        Binary mask indicating selected features (0/1).
    """
    # V_f = featNum * kNeigh
    if GG.featNum is None or GG.kNeigh is None:
        raise ValueError("newtry_ms: GG.featNum hoặc GG.kNeigh chưa được set.")

    V_f = GG.featNum * GG.kNeigh

    templateAdj = np.asarray(inputAdj, dtype=float)

    chromes = copy_of_en_nsga_2_mating_strategy(pop, times, templateAdj, V_f)

    feat_part = chromes[:, :GG.featNum]
    obj_part = chromes[:, GG.featNum:GG.featNum + 2]

    # alpha = 0.9; fits = alpha*abs(obj1) + (1-alpha)*(1 - obj2 / n_feat)
    alpha = 0.9
    obj1 = np.abs(obj_part[:, 0])  # f(1)
    obj2 = obj_part[:, 1]          # f(2) = số feature
    n_feat = GG.featNum

    fits = alpha * obj1 + (1 - alpha) * (1 - obj2 / n_feat)

    # choose index with fitness max
    idx = np.argmax(fits)

    featidx = feat_part[idx, :]

    return featidx
