import numpy as np

from copy_of_en_nsga_2_mating_strategy import copy_of_en_nsga_2_mating_strategy
import gbfs_globals as GG


def _get_pareto_indices(objs, minimize=True):
    """
    objs : np.ndarray, shape (N, M)  (N nghiệm, M objectives)
    minimize : bool
        Nếu True: giả sử tất cả objectives đều dạng "càng nhỏ càng tốt".

    Trả về:
        idx_front : np.ndarray chỉ số các nghiệm không bị dominated (Pareto front).
    """
    objs = np.asarray(objs, dtype=float)
    N = objs.shape[0]

    if not minimize:
        # Nếu bạn có objective dạng maximize thì có thể đổi dấu trước khi gọi,
        # hoặc xử lý riêng ở đây.
        pass

    is_nd = np.ones(N, dtype=bool)  # non-dominated flags

    for i in range(N):
        if not is_nd[i]:
            continue
        # nghiệm i
        oi = objs[i]

        # nghiệm j bị dominated bởi i nếu:
        #   - oi <= oj với mọi chiều
        #   - oi <  oj với ít nhất 1 chiều
        # → ở đây mình tìm các nghiệm (j) bị i dominate
        dominated = np.all(objs >= oi, axis=1) & np.any(objs > oi, axis=1)
        # nhưng KHÔNG loại chính i
        dominated[i] = False
        is_nd[dominated] = False

    idx_front = np.where(is_nd)[0]
    return idx_front


def newtry_ms(inputAdj, pop=20, times=40, run_dir=None):
    """
    Parameters
    ----------
    inputAdj : array-like
    pop : int
        Population size
    times : int
        Number of generations

    Returns
    -------
    featidx_best : np.ndarray, shape (featNum,)
        Binary mask nghiệm tốt nhất theo scalar fitness (như code cũ).
    pareto_masks : np.ndarray, shape (n_solutions, featNum)
        Mặt nạ 0/1 cho các nghiệm trên Pareto front theo (obj1, obj2).
    pareto_objs  : np.ndarray, shape (n_solutions, 2)
        Giá trị (obj1, obj2) tương ứng với mỗi nghiệm Pareto.
    """
    if GG.featNum is None or GG.kNeigh is None:
        raise ValueError("newtry_ms: GG.featNum hoặc GG.kNeigh chưa được set.")

    V_f = GG.featNum * GG.kNeigh
    templateAdj = np.asarray(inputAdj, dtype=float)

    chromes = copy_of_en_nsga_2_mating_strategy(pop, times, templateAdj, V_f, run_dir)

    # Giả sử format: [feat_part | obj1 | obj2 | ...]
    feat_part = chromes[:, :GG.featNum]
    kNeigh_part = chromes[:, GG.featNum : GG.featNum + V_f]
    obj_part = chromes[:, GG.featNum + V_f : GG.featNum + V_f + 2]  # (N, 2)

    # -----------------------------
    # 1) Chọn nghiệm "best" như cũ
    # -----------------------------
    alpha = 0.9
    obj1 = np.abs(obj_part[:, 0])  # f(1)  (tuỳ định nghĩa của bạn)
    obj2 = obj_part[:, 1]          # f(2) = số feature (hoặc cái gì đó)
    n_feat = GG.featNum

    fits = alpha * obj1 + (1 - alpha) * (1 - obj2 / n_feat)
    idx_best = np.argmax(fits)
    featidx_best = feat_part[idx_best, :]
    kNeigh_best = kNeigh_part[idx_best, :]
    # print("Best solution:")
    # print("Feature mask:", featidx_best)
    # print("kNeigh mask:", kNeigh_best)

    # -----------------------------
    # 2) Lấy Pareto front dựa trên (obj1, obj2)
    objs_for_pareto = obj_part.copy()

    idx_front = _get_pareto_indices(objs_for_pareto, minimize=True)
    pareto_masks = feat_part[idx_front, :]
    pareto_objs  = obj_part[idx_front, :]

    return featidx_best, pareto_masks, pareto_objs, kNeigh_best
