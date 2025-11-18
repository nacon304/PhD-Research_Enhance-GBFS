import numpy as np
from scipy.spatial.distance import pdist, squareform

import gbfs_globals as GG

def mapminmax(x, ymin=0.0, ymax=1.0):
    """
    Simple equivalent of MATLAB's mapminmax for 1D array.
    Scales x linearly to [ymin, ymax].
    """
    x = np.asarray(x, dtype=float)
    xmin = np.min(x)
    xmax = np.max(x)
    if xmax == xmin:
        return np.full_like(x, (ymin + ymax) / 2.0, dtype=float)
    return (x - xmin) * (ymax - ymin) / (xmax - xmin) + ymin


def disEva(arChive):
    """
    Parameters
    ----------
    arChive : np.ndarray, shape (M, featNum)
        Each row is a feature subset (0/1 or bool) for one archive individual.

    Returns
    -------
    tempkGenArchive : np.ndarray, shape (1, M)
        Evaluation values for each archive subset (higher is better or worse
        base on logic).
    """
    arChive = np.asarray(arChive)
    if arChive.ndim != 2:
        raise ValueError("disEva: arChive must be 2D (M x featNum).")

    M, _ = arChive.shape
    tempkGenArchive = np.zeros((1, M), dtype=float)

    if GG.data is None or GG.label is None or GG.trIdx is None:
        raise ValueError("disEva: global 'data', 'label' or 'trIdx' not set.")

    trlabel = np.asarray(GG.label)[GG.trIdx]

    for j in range(M):  # each row is a feature subset
        featIdx_row = arChive[j, :]
        feat_mask = (featIdx_row != 0)

        # select data with that feature subset on training set
        sub_data = np.asarray(GG.data)[GG.trIdx][:, feat_mask]

        # if no feature selected => distance = 0
        if sub_data.shape[1] == 0:
            tempkGenArchive[0, j] = 0.0
            continue

        D = pdist(sub_data)  # condensed distances
        DM = squareform(D)   # full matrix NxN

        n = DM.shape[0]
        farHit = np.zeros(n, dtype=float)
        nearMiss = np.zeros(n, dtype=float)

        for i in range(n):
            sameClassIdx = (trlabel == trlabel[i])
            otherClassIdx = ~sameClassIdx

            if np.any(sameClassIdx):
                farHit[i] = np.max(DM[i, sameClassIdx])
            else:
                farHit[i] = 0.0

            if np.any(otherClassIdx):
                nearMiss[i] = np.min(DM[i, otherClassIdx])
            else:
                nearMiss[i] = 0.0

        # normalize farHit & nearMiss using mapminmax
        tmpCom = np.concatenate([farHit, nearMiss])
        tmpComNom = mapminmax(tmpCom, 0.0, 1.0)

        len_n = n
        farHitNom = tmpComNom[:len_n]
        nearMissNom = tmpComNom[len_n:]

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = nearMissNom / farHitNom
        tempkGenArchive[0, j] = np.nanmean(ratio)

    return tempkGenArchive
