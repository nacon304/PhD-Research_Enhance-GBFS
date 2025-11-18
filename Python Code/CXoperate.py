import numpy as np

from uniformX_new import uniformX_new

def CXoperate(parent_chromosome, px, pm):
    """
    Parameters
    ----------
    parent_chromosome : np.ndarray, shape (N, V)
        Gene part of parent population.
    px : float
        Crossover probability.
    pm : float
        Mutation probability.

    Returns
    -------
    NewChrom : np.ndarray, shape (N, V)
        New chromosomes after crossover/mutation.
    """
    parent_chromosome = np.asarray(parent_chromosome, dtype=float)

    NewChrom = uniformX_new(parent_chromosome, px, pm)

    return np.asarray(NewChrom, dtype=float)
