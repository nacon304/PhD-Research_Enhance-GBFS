import numpy as np

import gbfs_globals as GG

def select2(bucket):
    """
    Parameters
    ----------
    bucket : list of array-like
        bucket[i] is a list or array of feature indices in shell i.

    Returns
    -------
    p : np.ndarray, shape (<= M-1,)
        choose 1 feature "representative" for each bucket (except the last one),
        based on weight vWeight (choose feature with the highest weight).
    """
    if GG.vWeight is None:
        raise ValueError("select: global vWeight is not set.")

    M = len(bucket)
    p_list = []

    for i in range(M - 1):
        feats = np.asarray(bucket[i], dtype=int).ravel()
        if feats.size == 0:
            continue

        featweg = np.asarray(GG.vWeight, dtype=float)[feats]
        # sort by weight descending
        idx_desc = np.argsort(-featweg)
        # feature with the highest weight
        best_feat = feats[idx_desc[0]]
        p_list.append(int(best_feat))

    if len(p_list) == 0:
        return np.array([], dtype=int)
    return np.asarray(p_list, dtype=int)
