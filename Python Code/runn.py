import os
from copy_of_js_ms import Copy_of_js_ms
import gbfs_globals as GG

def runn(data_idx):
    """
    Parameters
    ----------
    data_idx : int or iterable

    Returns
    -------
    PP : list
    name : str
        Dataset name (dataName).
    """
    GG.GAPGEN = 5

    base_dir = os.path.join(os.getcwd(), f"{GG.GAPGEN}_GapGens")
    mat_dir = os.path.join(base_dir, "data")
    pf_dir = os.path.join(base_dir, "pf")

    os.makedirs(mat_dir, exist_ok=True)
    os.makedirs(pf_dir, exist_ok=True)

    DELT = 0.2
    RUNS = GG.RUNS

    # Regularize data_idx
    if isinstance(data_idx, (int, float)):
        data_idx_list = [int(data_idx)]
    else:
        data_idx_list = list(data_idx)

    count = len(data_idx_list)

    PallRec = [None] * count
    AN = [[0] for _ in range(count * 2)]         # assiNum
    P = [[0.0, 0.0, 0.0] for _ in range(count * 2)]  # statistic [acc, featNum, time]
    time_stats = [[0.0, 0.0] for _ in range(count)]  # time stats mean/std

    idx = 1
    print(f"Processing dataset number {idx:2d}....")

    list_pos = idx - 1
    current_data_id = data_idx_list[list_pos]

    p, T, data_name, assi_num = Copy_of_js_ms(current_data_id, DELT, 0, RUNS)

    PP = p
    name = data_name

    PallRec[list_pos] = p

    AN[(idx - 1) * 2 : idx * 2] = [
        [assi_num[-2][0]],
        [assi_num[-1][0]],
    ]

    last_two_p = p[-2:]
    P[(idx - 1) * 2 : idx * 2] = [
        [last_two_p[0][0], last_two_p[0][1], last_two_p[0][4]],
        [last_two_p[1][0], last_two_p[1][1], last_two_p[1][4]],
    ]

    time_stats[list_pos] = [T[-2][0], T[-1][0]]

    return PP, name
