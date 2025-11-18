import numpy as np

def crtbase(Lind, Base=None):
    """
    Parameters
    ----------
    Lind : int or array-like
        Length(s) of alleles. Sum(Lind) is total chromosome length.
    Base : int or array-like, optional
        Base(s) of loci. Default: 2 for all.

    Returns
    -------
    BaseVec : np.ndarray, shape (sum(Lind),)
        Base for each locus in chromosome.
    """
    Lind_arr = np.atleast_1d(Lind).astype(int).ravel()

    if Base is None:
        Base_arr = np.full(Lind_arr.size, 2, dtype=int)
    else:
        Base_arr = np.atleast_1d(Base).astype(int).ravel()
        if Base_arr.size == 1 and Lind_arr.size > 1:
            Base_arr = np.full(Lind_arr.size, Base_arr[0], dtype=int)
        elif Base_arr.size != Lind_arr.size:
            raise ValueError("crtbase: Lind and Base dimensions must agree.")

    BaseVec_list = []
    for L_i, B_i in zip(Lind_arr, Base_arr):
        BaseVec_list.append(np.full(L_i, B_i, dtype=int))

    if BaseVec_list:
        BaseVec = np.concatenate(BaseVec_list)
    else:
        BaseVec = np.array([], dtype=int)

    return BaseVec
