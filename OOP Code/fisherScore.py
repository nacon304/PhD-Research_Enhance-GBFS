from fsFisher import fsFisher

def fisherScore(X, Y):
    """
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    Y : array-like, shape (n_samples,)

    Returns
    -------
    feats : array-like
        Feature indices (1..numF).
    w : array-like
        Fisher scores for each feature.
    """
    out = fsFisher(X, Y)
    feats = out["fList"]
    w = out["W"]
    return feats, w
