import numpy as np

import gbfs_globals as GG

def decodeNet(f, templateAdj):
    """
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

    total_genes = GG.V_f
    if f.size != total_genes:
        raise ValueError(
            f"decodeNet: f size {f.size} != total genes from neigh_list = {total_genes}"
        )
    
    offset = 0
    for i in range(GG.featNum):
        neigh = np.asarray(GG.neigh_list[i], dtype=int)
        deg_i = neigh.size
        if deg_i == 0:
            continue

        genes_i = f[offset : offset + deg_i]
        offset += deg_i

        on_local = np.nonzero(genes_i)[0]
        if on_local.size == 0:
            continue

        j_idx = neigh[on_local]
        MODE[i, j_idx] = 1.0

    indivNet = MODE * K
    return indivNet
