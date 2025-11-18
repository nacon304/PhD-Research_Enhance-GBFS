import numpy as np
from crtbase import crtbase

def mut(OldChrom, Pm=None, BaseV=None):
    """
    Parameters
    ----------
    OldChrom : np.ndarray, shape (Nind, Lind)
        Current population.
    Pm : float, optional
        Mutation probability. Default 0.7/Lind if None or nan.
    BaseV : array-like, optional
        Base of each locus. If None, binary (base 2) is assumed.

    Returns
    -------
    NewChrom : np.ndarray, shape (Nind, Lind)
        Mutated population.
    """
    OldChrom = np.asarray(OldChrom, dtype=int)
    Nind, Lind = OldChrom.shape

    # handle Pm
    if Pm is None or (isinstance(Pm, float) and np.isnan(Pm)):
        Pm = 0.7 / Lind

    # handle BaseV
    if BaseV is None or (isinstance(BaseV, float) and np.isnan(BaseV)):
        BaseV = crtbase(Lind)
    else:
        BaseV = np.asarray(BaseV, dtype=int).ravel()
        if BaseV.size == 0:
            BaseV = crtbase(Lind)

    # check compatibility
    if BaseV.size != Lind:
        raise ValueError("mut: OldChrom and BaseV are incompatible.")

    BaseM = np.tile(BaseV, (Nind, 1))

    # mutation mask
    mut_mask = np.random.rand(Nind, Lind) < Pm

    # random offsets in [1..BaseM-1]
    rand_vals = np.random.rand(Nind, Lind)
    offsets = np.ceil(rand_vals * (BaseM - 1)).astype(int)

    # apply mask
    delta = mut_mask * offsets

    NewChrom = np.mod(OldChrom + delta, BaseM)
    return NewChrom
