import numpy as np
from mut import mut

def uniformX_new(OldChrom, px, pm):
    """
    Parameters
    ----------
    OldChrom : np.ndarray, shape (M, N)
        Parent population (genes only).
    px : float
        Crossover probability for each gene position.
    pm : float
        Mutation probability (passed to mut).

    Returns
    -------
    NewChrom : np.ndarray, shape (2*M, N)
        Offspring population (child_1 and child_2 interleaved by rows).
    """
    OldChrom = np.asarray(OldChrom, dtype=float)
    M, N = OldChrom.shape

    parent_1 = np.arange(M)
    parent_2 = np.random.permutation(M)

    # ensure parent_2(i) != parent_1(i)
    eqIdx = parent_1 == parent_2
    while np.any(eqIdx):
        # reassign only those positions that are equal
        parent_2[eqIdx] = np.random.randint(0, M, size=eqIdx.sum())
        eqIdx = parent_1 == parent_2

    # -> Bernoulli(px) mask of True/False
    template_x = np.random.rand(M, N) < px

    child_1 = OldChrom[parent_2, :] * template_x + OldChrom[parent_1, :] * (~template_x)
    child_2 = OldChrom[parent_1, :] * template_x + OldChrom[parent_2, :] * (~template_x)

    # mutation
    child_1 = mut(child_1, pm)
    child_2 = mut(child_2, pm)

    NewChrom = np.zeros((2 * M, N), dtype=float)
    NewChrom[0::2, :] = child_1
    NewChrom[1::2, :] = child_2

    return NewChrom
