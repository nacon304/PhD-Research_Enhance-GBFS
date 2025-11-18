import numpy as np

def fsFisher(X, Y):
    """
    Fisher Score using the N-class formulation.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data matrix, each row is an instance.
    Y : array-like, shape (n_samples,)
        Labels in format 1, 2, 3, ...

    Returns
    -------
    out : dict
        {
            "W":    array of shape (n_features,), Fisher scores,
            "fList": array of feature indices (1..numF),
            "prf":  1
        }
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y).ravel()

    classes = np.unique(Y)
    numC = len(classes)
    _, numF = X.shape

    W = np.zeros(numF, dtype=float)

    # statistics for classes
    cIDX = []
    n_i = np.zeros(numC, dtype=int)
    for j, c in enumerate(classes):
        idx = np.where(Y == c)[0]
        cIDX.append(idx)
        n_i[j] = idx.size

    # calculate score for each feature
    for i in range(numF):
        temp1 = 0.0
        temp2 = 0.0
        f_i = X[:, i]
        u_i = np.mean(f_i)

        for j in range(numC):
            f_cj = f_i[cIDX[j]]
            u_cj = np.mean(f_cj)
            var_cj = np.var(f_cj, ddof=0)

            temp1 += n_i[j] * (u_cj - u_i) ** 2
            temp2 += n_i[j] * var_cj

        if temp1 == 0:
            W[i] = 0.0
        else:
            if temp2 == 0:
                W[i] = 100.0
            else:
                W[i] = temp1 / temp2

    fList = np.arange(1, numF + 1)
    out = {
        "W": W,
        "fList": fList,
        "prf": 1,
    }
    return out
