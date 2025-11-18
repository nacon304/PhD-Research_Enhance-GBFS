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

    STD = np.asarray(GG.kNeiMatrix, dtype=int) * rShapeF

    # For each feature, activate edges according to genetic mask
    for i in range(GG.featNum):
        # indices where STD(i,:) != 0, these are neighbor indices
        idx = np.nonzero(STD[i, :])[0]
        if idx.size > 0:
            # neighbor indices (0-based; kNeiMatrix đã được xây 0-based ở Python)
            neigh_indices = STD[i, idx].astype(int)
            MODE[i, neigh_indices] = 1.0

    indivNet = MODE * K
    return indivNet
