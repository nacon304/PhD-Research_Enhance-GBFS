import os
import numpy as np
from scipy.io import loadmat

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

    base_dir = "D:/PhD/The First Paper/Code Implement/GBFS-SND/Data"

    # Map index -> (filename, datasetName)
    mapping = {
        1:  ("glass.mat",      "Glass"),
        2:  ("wine.mat",       "Wine"),
        3:  ("hepatitis.mat",  "Hepatitis"),
        4:  ("WDBC.mat",       "Wdbc"),
        5:  ("ionosphere1.mat","ionosphere"),
        6:  ("sonar.mat",      "Sonar"),
        7:  ("Hill-Valley3.mat","Hill-Valley3"),
        8:  ("Urban.mat",      "Urban"),
        9:  ("Musk1.mat",      "Musk1"),
        10: ("LSVT.mat",       "LSVT"),
        11: ("madelon.mat",    "Madelon"),
        14: ("ORL.mat",        "ORL"),
        13: ("Yale.mat",       "Yale"),
        12: ("colon.mat",      "Colon"),
        15: ("SRBCT.mat",      "SRBCT"),
        16: ("lung.mat",       "lung"),
        17: ("lymphoma.mat",   "lymphoma"),
        18: ("TOX_171.mat",    "TOX_171"),
        19: ("leukemia.mat",   "leukemia"),
        20: ("ALLAML.mat",     "ALLAML"),
        21: ("Prostate.mat",   "Prostate"),
        22: ("Carcinom.mat",   "Carcinom"),
        23: ("DLBCL.mat",      "DLBCL"),
        24: ("COLL20.mat",     "COLL20"),
        25: ("BASEHOCK.mat",   "BASEHOCK"),
        26: ("PCMAC.mat",      "PCMAC"),
        27: ("GLIOMA.mat",     "GLIOMA"),
        28: ("9Tumor.mat",     "9Tumor"),
        71: ("Hill_Valley1.mat","Hill-Valley1"),
        72: ("Hill_Valley2.mat","Hill-Valley2"),
        73: ("Hill_Valley3.mat","Hill-Valley3"),
        74: ("Hill_Valley4.mat","Hill-Valley4"),
        75: ("Urban_train.mat","Urban_train"),
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
