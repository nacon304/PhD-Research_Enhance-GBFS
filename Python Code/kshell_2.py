import numpy as np
from my_degree import my_degree
from select2 import select2

def kshell_2(Gadj0):
    """
    Parameters
    ----------
    Gadj0 : array-like, shape (n_features, n_features)
        Adjacency matrix.

    Returns
    -------
    p : np.ndarray, 1D, dtype=int
        Sorted indices of selected features (0-based).
    """
    Gadj = np.array(Gadj0, dtype=float).copy()
    _, M = Gadj.shape

    featSeq = list(range(M))

    bucket = []
    temp = []
    it = 1
    bucketidx = 0

    # loop until all nodes are removed
    while len(featSeq) > 0:
        # degree measure on current subgraph
        # print(f"Iteration {it}, remaining nodes: {len(featSeq)}")
        D = my_degree(Gadj, featSeq)
        D = np.asarray(D, dtype=float).ravel()

        if np.isnan(D).any():
            raise ValueError("Degree vector contains NaN.")

        minD = np.min(D)

        while True:
            # indices (positions) in featSeq that have minimal degree
            feat_pos = np.where(D == minD)[0]
            # print(feat_pos)

            if feat_pos.size == 0:
                # store current shell
                bucket.append(np.array(temp, dtype=int))
                temp = []
                bucketidx += 1
                it += 1
                break
            else:
                a = [featSeq[pos] for pos in feat_pos]
                # print(f"Nodes with degree {minD}: {a}")

                # add to current bucket
                temp.extend(a)

                # remove these nodes from featSeq
                for pos in sorted(feat_pos, reverse=True):
                    del featSeq[pos]

                # remove their edges from adjacency matrix
                Gadj[feat_pos, :] = 0
                Gadj[:, feat_pos] = 0

                # recompute degree for remaining nodes
                if len(featSeq) == 0:
                    break
                D = my_degree(Gadj, featSeq)
                D = np.asarray(D, dtype=float).ravel()

        if len(featSeq) == 0:
            if len(temp) > 0:
                bucket.append(np.array(temp, dtype=int))
                bucketidx += 1

    Nbkt = len(bucket)
    # print(f"Number of shells (buckets) = {Nbkt}")
    # print(Gadj0)
    if Nbkt > 1:
        feat = select2(bucket)
        p = np.concatenate(
            [np.asarray(feat, dtype=int).ravel(), bucket[Nbkt - 1].ravel()]
        )
    else:
        p = bucket[Nbkt - 1].ravel()

    p = np.sort(p)
    return p
