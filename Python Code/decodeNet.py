import numpy as np

import gbfs_globals as GG


def decodeNet(f, templateAdj=None):
    """
    Node-based version of decodeNet.

    Parameters
    ----------
    f : array-like, shape (featNum,)
        Binary chromosome for NODES (z-vector): z_i = 1 nếu feature i nằm trong S(z).
    templateAdj : array-like, optional
        Kept for API compatibility (not used here).

    Returns
    -------
    indivNet : np.ndarray, shape (featNum, featNum)
        Induced redundancy adjacency matrix for S(z):
        - lấy GG.Zout (ma trận redundancy sau threshold),
        - chỉ giữ các hàng/cột tương ứng với S(z) (các node có z_i != 0),
        - phần còn lại = 0.
    """
    if GG.featNum is None:
        raise ValueError("decodeNet: GG.featNum chưa được set (global).")

    if GG.Zout is None:
        raise ValueError(
            "decodeNet: GG.Zout (redundancy graph) chưa được khởi tạo. "
            "Hãy đảm bảo Copy_of_js_ms đã build ma trận tương quan threshold."
        )

    f = np.asarray(f, dtype=float).ravel()
    if f.size != GG.featNum:
        raise ValueError(
            f"decodeNet (node-based): kích thước chromosome f={f.size} "
            f"không khớp GG.featNum={GG.featNum}."
        )

    # S(z) = {i | f_i != 0}
    S_idx = np.where(f != 0)[0]

    K = np.asarray(GG.Zout, dtype=float)
    MODE = np.zeros_like(K)

    if S_idx.size > 0:
        # chỉ giữ subgraph induced trên S(z)
        MODE[np.ix_(S_idx, S_idx)] = 1.0

    indivNet = MODE * K
    return indivNet
