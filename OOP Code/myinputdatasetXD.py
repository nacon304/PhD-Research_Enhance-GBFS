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

    base_dir = "D:/PhD/The First Paper/Code Implement/Data"

    # Map index -> (filename, datasetName)
    mapping = {
        1:  ("glass.mat",      "Glass"),
        2:  ("Urban.mat",      "Urban"),
        3:  ("Musk1.mat",      "Musk1"),
        4:  ("madelon.mat",    "Madelon"),
        5:  ("Bioresponse.mat","Bioresponse"),
        6:  ("Colon.mat",      "Colon"),
        7:  ("PIE10P.mat",     "PIE10P"),
        8:  ("BASEHOCK.mat",   "BASEHOCK"),
        9:  ("GISETTE.mat",    "GISETTE"),
        10: ("TOX171.mat",     "TOX_171"),
        11: ("ARCENE.mat",     "ARCENE"),
        12: ("SMKCAN187.mat",  "SMK_CAN_187"),
        13: ("DEXTER.mat",     "DEXTER"),
        
        14: ("GAMETES_Epi2_1000atts.mat","GAMETES"),
        15: ("USPS.mat",                 "USPS"),
        16: ("ISOLET.mat",               "ISOLET"),
        17: ("GINA_01.mat",              "GINA"),
        18: ("ORL_32x32.mat",            "ORL"),
        19: ("COIL100_32x32_01.mat",     "COIL100"),
        # 11: ("wine.mat",       "Wine"),
        # 12: ("hepatitis.mat",  "Hepatitis"),
        # 13: ("Yale.mat",       "Yale"),
        # 14: ("ORL.mat",        "ORL"),
        # 15: ("SRBCT.mat",      "SRBCT"),
        # 16: ("lung.mat",       "lung"),
        # 17: ("lymphoma.mat",   "lymphoma"),
        # 18: ("Hill-Valley3.mat","Hill-Valley3"),
        # 19: ("leukemia.mat",   "leukemia"),
        # 20: ("ALLAML.mat",     "ALLAML"),
        # 21: ("Prostate.mat",   "Prostate"),
        # 22: ("Carcinom.mat",   "Carcinom"),
        # 23: ("DLBCL.mat",      "DLBCL"),
        # 24: ("COLL20.mat",     "COLL20"),
        # 25: ("ionosphere1.mat","ionosphere"),
        # 26: ("PCMAC.mat",      "PCMAC"),
        # 27: ("GLIOMA.mat",     "GLIOMA"),
        # 28: ("9Tumor.mat",     "9Tumor"),
        # 29: ("WDBC.mat",       "Wdbc"),
        # 71: ("Hill_Valley1.mat","Hill-Valley1"),
        # 72: ("Hill_Valley2.mat","Hill-Valley2"),
        # 73: ("Hill_Valley3.mat","Hill-Valley3"),
        # 74: ("Hill_Valley4.mat","Hill-Valley4"),
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
