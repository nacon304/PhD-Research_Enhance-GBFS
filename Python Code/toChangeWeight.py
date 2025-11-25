import numpy as np
from pprint import pprint

import gbfs_globals as GG

def toChangeWeight(kGenWeiArchive, kGenAccArchive, arChive):
    """
    Parameters
    ----------
    kGenWeiArchive : array-like, shape (K,) or (1, K)
        Weight evaluations of k-best individuals (weights).
    kGenAccArchive : array-like
    arChive : array-like, shape (K, featNum)
        Archive of feature subsets (0/1) for K individuals.

    Side effect
    -----------
    Update global vWeight:
        vWeight = vWeight + (kGenWeiArchive @ arChive)
    """

    # kGenWeiArchive = np.asarray(kGenWeiArchive, dtype=float).ravel()
    # kGenAccArchive = np.asarray(kGenAccArchive, dtype=float).ravel()
    # arChive = np.asarray(arChive, dtype=float)

    # if arChive.ndim != 2:
    #     raise ValueError("toChangeWeight: arChive must be 2D (K x featNum).")

    # K = arChive.shape[0]
    # if kGenWeiArchive.size != K or kGenAccArchive.size != K:
    #     raise ValueError("Size mismatch.")

    # # 1) relative accuracy to mean
    # acc_centered = kGenAccArchive - kGenAccArchive.mean()

    # # 2) combine original weight + relative accuracy
    # #    alpha can be adjusted in [0,1]
    # alpha = 0.5
    # coeff = alpha * kGenWeiArchive + (1 - alpha) * acc_centered

    # w0 = coeff @ arChive  # (featNum,)

    # if GG.vWeight is None:
    #     GG.vWeight = w0.copy()
    # else:
    #     GG.vWeight = np.asarray(GG.vWeight, dtype=float).ravel()
    #     GG.vWeight += w0

    kGenWeiArchive = np.asarray(kGenWeiArchive, dtype=float).ravel()  # (K,)
    arChive = arChive * kGenAccArchive[:, None]
    arChive = np.asarray(arChive, dtype=float)  # (K, featNum)
    # print("toChangeWeight:")
    # pprint(kGenWeiArchive)
    # pprint(kGenAccArchive)
    # pprint(arChive)

    # print("Shapes:", kGenWeiArchive.shape, arChive.shape, kGenAccArchive.shape)

    if arChive.ndim != 2:
        raise ValueError("toChangeWeight: arChive must be 2D (K x featNum).")

    K = arChive.shape[0]
    if kGenWeiArchive.size != K:
        raise ValueError(
            f"toChangeWeight: kGenWeiArchive length {kGenWeiArchive.size} "
            f"!= number of rows in arChive {K}."
        )

    w0 = kGenWeiArchive @ arChive  # shape (featNum,)

    if GG.vWeight is None:
        GG.vWeight = w0.copy()
    else:
        GG.vWeight = np.asarray(GG.vWeight, dtype=float).ravel()
        if GG.vWeight.size != w0.size:
            raise ValueError(
                f"toChangeWeight: vWeight length {GG.vWeight.size} != w0 length {w0.size}."
            )
        GG.vWeight = GG.vWeight + w0
    GG.vWeight = np.maximum(GG.vWeight, 0.0)
