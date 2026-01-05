import os
import numpy as np
from scipy.io import loadmat
from pathlib import Path

def myinputdatasetXD(i):
    """
    Parameters
    ----------
    i : int
        Index of the dataset to load.

    Returns
    -------
    dataset : np.ndarray
        Data matrix (samples x features+1), first column is label.
    whether : np.ndarray
        Label vector (samples, ).
    datasetName : str
        Name of the dataset (first letter upper, rest lower).
    """

    base_dir = str(Path(__file__).resolve().parents[1] / "Data")

    # Map index -> (filename, datasetName)
    mapping = {
        1:  ("glass.mat",      "Glass"),
        2:  ("Urban.mat",      "Urban"),
        3:  ("Musk1.mat",      "Musk1"),
        4:  ("USPS.mat",       "USPS"),
        5:  ("madelon.mat",    "Madelon"),
        6:  ("ISOLET.mat",     "ISOLET"),
        7:  ("GINA_01.mat",    "GINA"),
        8:  ("Bioresponse.mat","Bioresponse"),
        9:  ("Colon.mat",      "Colon"),
        10: ("PIE10P.mat",     "PIE10P"),
        11: ("BASEHOCK.mat",   "BASEHOCK"),
        12: ("GISETTE.mat",    "GISETTE"),
        13: ("TOX171.mat",     "TOX_171"),
        14: ("ARCENE.mat",     "ARCENE"),
    }

    if i not in mapping:
        raise ValueError(f"No dataset mapped for index {i}.")

    filename, datasetName = mapping[i]
    full_path = os.path.join(base_dir, filename)

    # Load .mat file
    mat = loadmat(full_path)
    if "Label_data" not in mat:
        raise KeyError(f"'Label_data' not found in {full_path}")

    dataset = np.array(mat["Label_data"])

    whether = dataset[:, 0]

    if isinstance(whether, np.ndarray) and whether.ndim == 2 and whether.shape[1] != 1:
        dataset = dataset.T
        whether = dataset[:, 0]
        print("Dataset and labels were transposed to match expected shape.")

    # Basic stats
    Ins, featnum = dataset.shape
    classNum = np.unique(whether).shape[0]

    print(
        f"------{datasetName}    #features = {featnum - 1}   "
        f"#samples = {Ins}    #classes = {classNum}"
    )

    if dataset.shape[0] != whether.shape[0]:
        raise ValueError("Error. Dataset and label dimensions are inconsistent.")

    # Convert label -1 to 0 (same as whether(whether==-1)=0)
    whether = np.array(whether, dtype=float)
    whether[whether == -1] = 0

    # Normalize datasetName: first letter upper, rest lower
    if len(datasetName) > 1:
        datasetName = datasetName[0].upper() + datasetName[1:].lower()
    else:
        datasetName = datasetName.upper()

    return dataset, whether, datasetName
