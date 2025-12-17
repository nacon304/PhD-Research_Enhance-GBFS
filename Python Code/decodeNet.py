import numpy as np

import gbfs_globals as GG

def decodeNet(f, templateAdj):
    """
    Python version of decodeNet.m

    Parameters
    ----------
    f : array-like, shape (featNum*kNeigh,)
        Binary chromosome for edges (flattened).

    Returns
    -------
    indivNet : np.ndarray, shape (featNum, featNum)
        Adjacency matrix for the individual network (MODE .* K).
    """
    K = np.asarray(GG.kNeiZout, dtype=float)
    MODE = np.zeros_like(K)

    f = np.asarray(f, dtype=float).ravel()
    if GG.featNum is None or GG.kNeigh is None:
        raise ValueError("decodeNet: featNum or kNeigh is not set (global).")

    if f.size != GG.featNum * GG.kNeigh:
        raise ValueError(
            f"decodeNet: f size {f.size} != featNum*kNeigh = {GG.featNum*GG.kNeigh}"
        )

    # reshape chromosome into featNum x kNeigh
    rShapeF = f.reshape((GG.featNum, GG.kNeigh))

    # For each feature, activate edges according to genetic mask
    for i in range(GG.featNum):
        chosen_pos = np.where(rShapeF[i, :] != 0)[0]
        if chosen_pos.size > 0:
            neigh_indices = GG.kNeiMatrix[i, chosen_pos].astype(int)
            MODE[i, neigh_indices] = 1.0

    indivNet = MODE * K
    return indivNet
