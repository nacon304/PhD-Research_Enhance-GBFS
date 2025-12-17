import numpy as np

import gbfs_globals as GG

def toChangeWeight(kGenWeiArchive, kGenAccArchive, arChive):
    """
    Parameters
    ----------
    kGenWeiArchive : array-like, shape (K,) or (1, K)
        Weight evaluations of k-best individuals (trọng số).
    kGenAccArchive : array-like
        (Không dùng trong bản MATLAB hiện tại, nhưng giữ tham số cho đủ.)
    arChive : array-like, shape (K, featNum)
        Archive các tập con đặc trưng (0/1) cho K cá thể.

    Side effect
    -----------
    Cập nhật global vWeight:
        vWeight = vWeight + (kGenWeiArchive @ arChive)
    """

    kGenWeiArchive = np.asarray(kGenWeiArchive, dtype=float).ravel()  # (K,)
    arChive = np.asarray(arChive, dtype=float)  # (K, featNum)

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
