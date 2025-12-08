import numpy as np

from copy_of_en_nsga_2_mating_strategy import copy_of_en_nsga_2_mating_strategy
import gbfs_globals as GG

def newtry_ms(inputAdj, pop=20, times=40, run_dir=None):
    """
    Node-based NSGA-II wrapper.

    Parameters
    ----------
    inputAdj : array-like
        Vectorized adjacency template (giữ nguyên tham số để tương thích),
        nhưng tối ưu hoá sẽ encode theo NODE (V_f = featNum).
    pop : int
        Population size
    times : int
        Number of generations
    run_dir : str, optional
        Directory to save run-specific files.

    Returns
    -------
    featidx : np.ndarray, shape (featNum,)
        Binary mask indicating selected features (0/1).
    """
    # ---- YÊU CẦU: đã set GG.featNum trước khi gọi (copy_of_js_ms đảm bảo) ----
    if GG.featNum is None:
        raise ValueError("newtry_ms: GG.featNum chưa được set (cần số chiều feature).")

    # ---- NODE-BASED representation ----
    V_f = int(GG.featNum)   # chiều của vector quyết định = số đỉnh (feature)

    templateAdj = np.asarray(inputAdj, dtype=float)

    # Tiếp tục tái sử dụng pipeline NSGA-II (initialize/genetic eval sẽ cần đọc GG.Zout/GG.compMat)
    chromes = copy_of_en_nsga_2_mating_strategy(pop, times, templateAdj, V_f, run_dir)

    # chromes = [featIdx(mask theo NODE), objectives]
    feat_part = chromes[:, :GG.featNum]
    obj_part  = chromes[:, GG.featNum:GG.featNum + 2]

    # scalar hoá tạm thời để chọn 1 cá thể tốt nhất (giữ logic cũ)
    alpha = 0.9
    obj1 = obj_part[:, 0]          # f1 (1 - acc) hoặc acc tuỳ implement downstream
    obj2 = obj_part[:, 1]          # f2 (|F|/d)
    fits = alpha * (1 - obj1) + (1 - alpha) * (1 - obj2)

    idx = np.argmax(fits)
    featidx = feat_part[idx, :]

    return featidx
