import numpy as np

import gbfs_globals as GG

def my_degree(Gadj, featSeq):
    """
    Parameters
    ----------
    Gadj : array-like, shape (n_features, n_features)
        Adjacency matrix (weighted).
    featSeq : list[int] or np.ndarray
        Indices (0-based) of nodes still in the graph.

    Returns
    -------
    DEG : np.ndarray, shape (len(featSeq),)
        Degree measure for nodes in featSeq.
    """
    # global vWeight

    Gadj = np.asarray(Gadj, dtype=float)
    featSeq = np.asarray(featSeq, dtype=int)

    if GG.vWeight is None:
        raise ValueError("my_degree: global vWeight is not set.")

    lastWeg = np.asarray(GG.vWeight, dtype=float).ravel()

    orideg = np.sum(Gadj != 0, axis=1)

    oriweg = np.sum(Gadj, axis=1)

    deg_val = orideg[featSeq] * oriweg[featSeq] * lastWeg[featSeq]
    DEG = np.round(np.sqrt(deg_val))
    # print(DEG)

    return DEG
