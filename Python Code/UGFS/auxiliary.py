"""
auxiliary.py

Python equivalents of R auxiliary.R from Rdimtools:
- aux.typecheck
- check_ndim
- aux.preprocess
- aux.preprocess.hidden
- aux.featureindicator
"""

from __future__ import annotations

from typing import Literal, Dict, Any
import numpy as np


# ---------------------------------------------------------------------
# aux.typecheck
# ---------------------------------------------------------------------
def aux_typecheck(data, verbose: bool = False) -> bool:
    """
    Check that input is numeric and has no Inf / NaN.
    Rough equivalent of R's aux.typecheck.

    Parameters
    ----------
    data : array-like
    verbose : bool

    Returns
    -------
    ok : bool
    """
    arr = np.asarray(data)

    if arr.ndim == 1 and verbose:
        # Trong R họ chỉ cảnh báo, không sửa data
        print("WARNING : input data should be matrix, not a vector.")

    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError("ERROR : input data should be numeric arrays.")

    if not np.isfinite(arr).all():
        raise ValueError("ERROR : input data should contain no Inf or NA values.")

    return True


# ---------------------------------------------------------------------
# check_ndim
# ---------------------------------------------------------------------
def check_ndim(ndim, p: int) -> bool:
    """
    Check that ndim is a single numeric in [1, p].
    """
    if (
        np.ndim(ndim) != 0
        or not isinstance(ndim, (int, np.integer, float, np.floating))
        or np.isinf(ndim)
        or np.isnan(ndim)
    ):
        return False
    ndim = int(ndim)
    if ndim < 1 or ndim > p:
        return False
    return True


# ---------------------------------------------------------------------
# aux.preprocess
#   type in {"center","scale","cscale","decorrelate","whiten"}
# ---------------------------------------------------------------------
def aux_preprocess(
    data,
    type: Literal["center", "scale", "cscale", "decorrelate", "whiten"] = "center",
) -> Dict[str, Any]:
    """
    Preprocess data matrix (n x p).

    Returns
    -------
    result : dict
        {
          "pX": (n, p) preprocessed data,
          "info": {
              "type": str,
              "mean": (p,),
              "multiplier": (p, p) or scalar 1
          }
        }
    """
    # R code chuyển sang d-by-n rồi làm; ở đây làm trực tiếp n-by-p
    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError("aux_preprocess: input should be 2D (n x p).")

    n, p = X.shape
    type = str(type)

    # ----------------- "scale" : only scale to var=1, no centering
    if type == "scale":
        std = np.std(X, axis=0, ddof=1)
        # tránh chia cho 0
        std_safe = np.where(std == 0, 1.0, std)
        multiplier_vec = 1.0 / std_safe  # length p
        pX = X * multiplier_vec  # broadcasting: (n,p) * (p,)

        info = {
            "type": "scale",
            "mean": np.zeros(p),
            "multiplier": np.diag(multiplier_vec),
        }
        return {"pX": pX, "info": info}

    # ----------------- "cscale" : center + scale
    if type == "cscale":
        mean = X.mean(axis=0)
        X_centered = X - mean
        std = np.std(X_centered, axis=0, ddof=1)
        std_safe = np.where(std == 0, 1.0, std)
        multiplier_vec = 1.0 / std_safe
        pX = X_centered * multiplier_vec

        info = {
            "type": "cscale",
            "mean": mean,
            "multiplier": np.diag(multiplier_vec),
        }
        return {"pX": pX, "info": info}

    # ----------------- "center" / "decorrelate" / "whiten"
    # R dùng aux_preprocess(matinput, 1/2/3); ở đây tự cài lại.
    mean = X.mean(axis=0)
    X_centered = X - mean

    if type == "center":
        info = {
            "type": "center",
            "mean": mean,
            "multiplier": 1,  # giống docs: 1 cho "center"
        }
        return {"pX": X_centered, "info": info}

    # Covariance matrix (p x p) with centered data
    # S = (1/(n-1)) * X_centered^T X_centered
    S = np.cov(X_centered, rowvar=False, ddof=1)  # (p, p)

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(S)
    # sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    eps = 1e-12

    if type == "decorrelate":
        # Rotate to eigenbasis: covariance becomes diagonal (eigvals)
        multiplier = eigvecs  # (p, p)
        pX = X_centered @ multiplier
        info = {
            "type": "decorrelate",
            "mean": mean,
            "multiplier": multiplier,
        }
        return {"pX": pX, "info": info}

    if type == "whiten":
        # Whiten: decorrelate + scale so diag(cov) = 1
        inv_sqrt = np.diag(1.0 / np.sqrt(eigvals + eps))  # (p, p)
        multiplier = eigvecs @ inv_sqrt  # (p, p)
        pX = X_centered @ multiplier
        info = {
            "type": "whiten",
            "mean": mean,
            "multiplier": multiplier,
        }
        return {"pX": pX, "info": info}

    raise ValueError(f"aux_preprocess: unknown type '{type}'.")


# ---------------------------------------------------------------------
# aux.preprocess.hidden
# ---------------------------------------------------------------------
def aux_preprocess_hidden(
    data,
    type: Literal["null", "center", "scale", "cscale", "decorrelate", "whiten"] = "null",
    algtype: Literal["linear", "nonlinear"] = "linear",
) -> Dict[str, Any]:
    """
    Minimal preprocessing wrapper.

    Parameters
    ----------
    data : (n, p) array-like
    type : {"null","center","scale","cscale","decorrelate","whiten"}
    algtype : {"linear","nonlinear"}

    Returns
    -------
    result : dict
        {
          "pX": (n, p) preprocessed data,
          "info": {
             "type": ...,
             "mean": ...,
             "multiplier": ...,
             "algtype": "linear"/"nonlinear"
          }
        }
    """
    X = np.asarray(data, dtype=float)
    n, p = X.shape
    pptype = str(type)
    ppalgtype = str(algtype)

    if pptype == "null":
        info = {
            "type": "null",
            "mean": np.zeros(p),
            "multiplier": 1,
            "algtype": ppalgtype,
        }
        return {"pX": X, "info": info}
    else:
        result = aux_preprocess(X, type=pptype)
        result["info"]["algtype"] = ppalgtype
        return result


# ---------------------------------------------------------------------
# aux.featureindicator
#   generate (p x ndim) indicator matrix for projection
#   NOTE: in Python we assume idxvec is 0-based indices.
# ---------------------------------------------------------------------
def aux_featureindicator(p: int, ndim: int, idxvec) -> np.ndarray:
    """
    Generate (p x ndim) indicator matrix for projection.

    Parameters
    ----------
    p : int
        Total number of features.
    ndim : int
        Number of selected features (columns in indicator).
    idxvec : array-like of length ndim
        Indices of selected features (0-based).

    Returns
    -------
    output : ndarray, shape (p, ndim)
        Indicator matrix: column j has 1 at row idxvec[j].
    """
    idxvec = np.asarray(idxvec, dtype=int).ravel()

    if idxvec.size != ndim:
        raise ValueError("* aux_featureindicator : selection had some problem.")

    output = np.zeros((p, ndim), dtype=float)

    for i in range(ndim):
        selected = int(idxvec[i])
        if selected < 0 or selected >= p:
            raise ValueError(
                f"* aux_featureindicator : index {selected} out of range 0..{p-1}."
            )
        output[selected, i] = 1.0

    return output
