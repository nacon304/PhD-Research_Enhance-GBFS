# TRAIN:
#   - Score-based methods: ratios 0.1 -> 0.9, evaluate TRAIN by CV
#   - GBFS baseline/enhanced: take solver Pareto masks -> evaluate TRAIN by CV
#
# TEST:
#   - GBFS baseline/enhanced: ONLY selected-best subset (as your pipeline)
#   - Score-based methods: (1) ratios 0.1 -> 0.9, (2) k_ref = |S_enhanced|
#
# Outputs:
#   OUT_ROOT/compare_front_train.csv
#   OUT_ROOT/compare_test_points.csv
#
# Key fixes vs v4:
#   (1) CSV writer unions all keys across rows (no "score_time" crash)
#   (2) Hard isolation between OOP_ROOT and BASELINE_ROOT via purge+rebind
#       to prevent module-cache collisions (decodeNet / initialize_variables_f / etc.)
#   (3) Sanitizes complex-valued scores from traditional/graph methods
#       to avoid "Casting complex values..." warning
# ---------------------------------------------------------

from __future__ import annotations

import os
import sys
import json
import csv
import time
import warnings
import importlib
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings(
    "error",
    message=r".*Casting complex values to real discards the imaginary part.*",
)
# =========================================================
# 0) USER CONFIG
# =========================================================

BASELINE_ROOT = r"D:\PhD\The First Paper\Code Implement\GBFS-SND\Python Code"
OOP_ROOT      = r"D:\PhD\The First Paper\Code Implement\GBFS-SND\OOP Code"
OUT_ROOT      = r"D:\PhD\The First Paper\Code Implement\GBFS-SND\Compare\Results"

DATASET_INDICES = [14, 15, 16, 17, 18, 19]  # GAMETES, USPS, ISOLET, GINA, ORL, COIL100
RUNS = 3

TEST_SIZE = 0.30
SPLIT_SEED = 42 

KNN_EVAL_K = 5

CV_FOLDS = 5
CV_SEED = 42

BASE_DELT = 10.0
BASE_OMEGA = 0.8
BASE_KNEIGH = 5
BASE_POP = 20
BASE_GEN = 50
# BASE_POP = 1
# BASE_GEN = 1

RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

SILENCE_WARNINGS = False


# =========================================================
# 1) BASIC UTILITIES
# =========================================================

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _write_json(path: str, obj: Any) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _write_text(path: str, s: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)

def _write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    """
    Robust CSV writer:
      - unions all keys across all rows so DictWriter never crashes
      - keeps first-seen column order
    """
    _ensure_dir(os.path.dirname(path))
    if not rows:
        return

    if fieldnames is None:
        fieldnames = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def _mapminmax_zero_one(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    # if anything becomes complex due to upstream numeric issues, force to real safely
    if np.iscomplexobj(X):
        X = np.real(X)
    X = np.asarray(X, dtype=float)

    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    denom = X_max - X_min
    denom[denom == 0] = 1.0
    return (X - X_min) / denom

def _fixed_split_indices(X: np.ndarray, y: np.ndarray, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=int).ravel()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    for tr_idx, te_idx in sss.split(X, y):
        return tr_idx.astype(int), te_idx.astype(int)
    raise RuntimeError("No split produced.")

def _redundancy_rate_subset_fallback(S: np.ndarray, zData: np.ndarray) -> float:
    S = np.asarray(S, dtype=int).ravel()
    if S.size <= 1:
        return 0.0
    X = np.asarray(zData[:, S], dtype=float)
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    C = np.abs(C)
    iu = np.triu_indices_from(C, k=1)
    vals = C[iu]
    return float(np.mean(vals)) if vals.size else 0.0

def _cv_acc_knn(X: np.ndarray, y: np.ndarray, S: np.ndarray, cv: int, seed: int) -> float:
    S = np.asarray(S, dtype=int).ravel()
    if S.size == 0:
        return 0.0

    Xs = X[:, S]
    y = np.asarray(y, dtype=int).ravel()

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    accs = []
    for tr, va in skf.split(Xs, y):
        clf = KNeighborsClassifier(n_neighbors=KNN_EVAL_K)
        clf.fit(Xs[tr], y[tr])
        pred = clf.predict(Xs[va])
        accs.append(np.mean(pred == y[va]))
    return float(np.mean(accs)) if accs else 0.0

def _knn_acc_test(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray, S: np.ndarray) -> float:
    S = np.asarray(S, dtype=int).ravel()
    if S.size == 0:
        return 0.0
    clf = KNeighborsClassifier(n_neighbors=KNN_EVAL_K)
    clf.fit(Xtr[:, S], ytr)
    pred = clf.predict(Xte[:, S])
    return float(np.mean(pred == yte))

def _red(S: np.ndarray, zData_full: np.ndarray) -> float:
    # always use fallback here to avoid helper collision across roots
    return float(_redundancy_rate_subset_fallback(S, zData_full))

def _sanitize_scores(s: np.ndarray, mode: str = "abs") -> np.ndarray:
    """
    Convert any score array to a finite real-valued 1D float array.

    mode:
      - "abs": use magnitude for complex (recommended for stable ranking)
      - "real": drop imaginary part
    """
    s = np.asarray(s).ravel()

    if np.iscomplexobj(s):
        s = np.abs(s) if mode == "abs" else np.real(s)

    s = s.astype(float, copy=False)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    return s


# =========================================================
# 2) COLLISION CONTROL (CRITICAL)
# =========================================================

if SILENCE_WARNINGS:
    warnings.filterwarnings("ignore")

_PROJECT_NAMES = [
    # high collision (appears in both roots)
    "gbfs_globals",
    "myinputdatasetXD",
    "fisherScore",
    "helper",
    "newtry_ms",
    "copy_of_en_nsga_2_mating_strategy",
    "initialize_variables_f",
    "decodeNet",
    "encodeNet",
    "evaluate_objective_f",
    "evaluate_objective_f2",
    "graph_fs",
    "traditional_fs",
    # OOP-only but can linger
    "runner_oop",
    "oop_core",
    "sequential_strategies",
]

def _prioritize_path(root: str) -> None:
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)

def _is_under(file_path: str, root: str) -> bool:
    try:
        fp = os.path.abspath(file_path)
        rr = os.path.abspath(root)
        return fp.startswith(rr + os.sep)
    except Exception:
        return False

def _purge_modules_from_root(root: str) -> None:
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f and _is_under(f, root):
            del sys.modules[name]

def _purge_module_names(names: List[str]) -> None:
    for n in names:
        if n in sys.modules:
            del sys.modules[n]

def _enter_runtime(primary_root: str, secondary_root: Optional[str] = None) -> None:
    """
    Make runtime imports resolve ONLY to `primary_root`, and aggressively purge cached
    modules from both roots + known project module names to avoid cross-root collisions.
    """
    if secondary_root:
        _purge_modules_from_root(secondary_root)
    _purge_modules_from_root(primary_root)
    _purge_module_names(_PROJECT_NAMES)
    _prioritize_path(primary_root)

def _import(name: str):
    return importlib.import_module(name)


# =========================================================
# 3) ROW BUILDERS
# =========================================================

def _front_train_rows(
    dataset_idx: int,
    dataset_name: str,
    method: str,
    run_id: int,
    feature_sets: List[np.ndarray],
    X_train_for_cv: np.ndarray,
    y_train: np.ndarray,
    zData_full_for_red: np.ndarray,
    extra_cols: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    n_features = X_train_for_cv.shape[1]
    best_by_fnum: Dict[int, Dict[str, Any]] = {}

    for sid, S in enumerate(feature_sets):
        S = np.asarray(S, dtype=int).ravel()
        if S.size == 0:
            continue

        acc_cv = _cv_acc_knn(X_train_for_cv, y_train, S, cv=CV_FOLDS, seed=CV_SEED)
        er_cv = float(1.0 - acc_cv)
        fnum = int(S.size)
        fr = float(fnum / n_features)
        redv = _red(S, zData_full_for_red)

        row = {
            "dataset_idx": int(dataset_idx),
            "dataset": str(dataset_name),
            "method": str(method),
            "run": int(run_id),
            "point_id": int(sid),
            "fnum": int(fnum),
            "fRatio": float(fr),
            "acc_train_cv": float(acc_cv),
            "eRate_train_cv": float(er_cv),
            "red": float(redv),
            "fset": " ".join(map(str, S.tolist())),
        }
        if extra_cols:
            row.update(extra_cols)

        prev = best_by_fnum.get(fnum)
        if prev is None or er_cv < float(prev["eRate_train_cv"]):
            best_by_fnum[fnum] = row

    rows = list(best_by_fnum.values())
    rows.sort(key=lambda d: float(d["fRatio"]))
    return rows

def _test_point_row(
    dataset_idx: int,
    dataset_name: str,
    method: str,
    run_id: int,
    test_mode: str,
    S: np.ndarray,
    Xtr: np.ndarray, ytr: np.ndarray,
    Xte: np.ndarray, yte: np.ndarray,
    zData_full_for_red: np.ndarray,
    time_total: float,
    ratio: Optional[float] = None,
    k_ref_used: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    S = np.asarray(S, dtype=int).ravel()
    acc = _knn_acc_test(Xtr, ytr, Xte, yte, S) if S.size else 0.0

    row = {
        "dataset_idx": int(dataset_idx),
        "dataset": str(dataset_name),
        "method": str(method),
        "run": int(run_id),
        "test_mode": str(test_mode),
        "ratio": float(ratio) if ratio is not None else "",
        "k_ref_used": int(k_ref_used) if k_ref_used is not None else "",
        "acc_test": float(acc),
        "eRate_test": float(1.0 - acc),
        "fnum": int(S.size),
        "fRatio": float(S.size / Xtr.shape[1]) if Xtr.shape[1] else "",
        "red": float(_red(S, zData_full_for_red)) if S.size else 0.0,
        "time_total": float(time_total),
        "fset": " ".join(map(str, S.tolist())),
    }
    if extra:
        row.update(extra)
    return row


# =========================================================
# 4) GBFS BASELINE
# =========================================================

def run_gbfs_baseline_one_run(
    dataset_idx: int,
    dataset_name: str,
    X_raw: np.ndarray,
    y: np.ndarray,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    out_dir: str,
    run_id: int,
    delt: float,
    omega: float,
    kNeigh: int,
    pop: int,
    gen: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], int]:

    method = "GBFS_baseline"
    run_dir = os.path.join(out_dir, f"run_{run_id:02d}")
    _ensure_dir(run_dir)

    # ---- hard bind baseline runtime to avoid OOP collisions
    _enter_runtime(BASELINE_ROOT, secondary_root=OOP_ROOT)

    GG = _import("gbfs_globals")
    fisherScore = _import("fisherScore").fisherScore
    baseline_newtry_ms = _import("newtry_ms").newtry_ms

    tic = time.perf_counter()

    X_raw = np.asarray(X_raw)
    if np.iscomplexobj(X_raw):
        X_raw = np.real(X_raw)
    X_raw = np.asarray(X_raw, dtype=float)

    y = np.asarray(y, dtype=int).ravel()

    zData_full = _mapminmax_zero_one(X_raw)
    tr_mask = np.zeros(zData_full.shape[0], dtype=bool)
    tr_mask[tr_idx] = True

    GG.DELT = float(delt)
    GG.OMEGA = float(omega)
    GG.data = zData_full
    GG.label = y
    GG.featNum = zData_full.shape[1]
    GG.kNeigh = int(kNeigh)

    GG.trIdx = tr_mask
    GG.trData = zData_full[tr_mask, :]
    GG.trLabel = y[tr_mask]
    GG.teData = zData_full[~tr_mask, :]
    GG.teLabel = y[~tr_mask]
    GG.assiNumInside = []

    _, vWeight0 = fisherScore(GG.trData, GG.trLabel)
    GG.vWeight = 1.0 + _mapminmax_zero_one(vWeight0.reshape(-1, 1)).ravel()
    GG.vWeight1 = GG.vWeight

    adj = 1.0 - pdist(GG.trData.T, metric="correlation")
    adj = np.nan_to_num(adj, nan=0.0)
    adj = np.abs(adj)
    GG.Weight = squareform(adj)
    GG.Zout = squareform(adj)

    GG.kNeiMatrix = np.zeros((GG.featNum, GG.kNeigh), dtype=int)
    kNeiZoutMode = np.zeros_like(GG.Zout)
    for i in range(GG.Zout.shape[0]):
        idx_sorted = np.argsort(-GG.Zout[i, :])
        idx_topk = idx_sorted[:GG.kNeigh]
        GG.kNeiMatrix[i, :] = idx_topk
        kNeiZoutMode[i, idx_topk] = 1

    GG.kNeiZout = GG.Zout * (kNeiZoutMode != 0)
    kNeiAdj = squareform(GG.kNeiZout, force="tovector", checks=False)

    out = baseline_newtry_ms(kNeiAdj, pop, gen)

    pareto_masks = None
    pareto_objs = None
    if isinstance(out, tuple):
        featidx_best = np.asarray(out[0])
        if len(out) > 1:
            pareto_masks = out[1]
        if len(out) > 2:
            pareto_objs = out[2]
    else:
        featidx_best = np.asarray(out)

    S_best = np.where(featidx_best != 0)[0].astype(int)

    t_total = float(time.perf_counter() - tic)
    assiNum = float(np.sum(np.asarray(getattr(GG, "assiNumInside", []), dtype=float))) if hasattr(GG, "assiNumInside") else 0.0

    fsets: List[np.ndarray] = []
    if pareto_masks is not None:
        pm = np.asarray(pareto_masks)
        if pm.ndim == 1 and pm.size > 0:
            pm = pm.reshape(1, -1)
        if pm.ndim == 2:
            for m in pm:
                S = np.where(np.asarray(m) != 0)[0].astype(int)
                if S.size > 0:
                    fsets.append(S)
    if not fsets:
        fsets = [S_best]

    train_front_rows = _front_train_rows(
        dataset_idx=dataset_idx,
        dataset_name=dataset_name,
        method=method,
        run_id=run_id,
        feature_sets=fsets,
        X_train_for_cv=GG.trData,
        y_train=GG.trLabel,
        zData_full_for_red=zData_full,
    )

    test_row = _test_point_row(
        dataset_idx=dataset_idx,
        dataset_name=dataset_name,
        method=method,
        run_id=run_id,
        test_mode="selected",
        S=S_best,
        Xtr=GG.trData, ytr=GG.trLabel,
        Xte=GG.teData, yte=GG.teLabel,
        zData_full_for_red=zData_full,
        time_total=t_total,
        extra={"assiNum": float(assiNum)},
    )

    _write_text(os.path.join(run_dir, "best_selected.txt"), " ".join(map(str, S_best.tolist())))
    _write_csv(os.path.join(run_dir, "front_train_cv.csv"), train_front_rows)
    _write_json(os.path.join(run_dir, "run_meta.json"), {
        "dataset_idx": int(dataset_idx),
        "dataset": str(dataset_name),
        "method": method,
        "run": int(run_id),
        "params": {"delt": delt, "omega": omega, "kNeigh": kNeigh, "pop": pop, "gen": gen},
        "assiNum": float(assiNum),
        "time_total": float(t_total),
        "best_fnum": int(S_best.size),
    })
    if pareto_objs is not None:
        np.save(os.path.join(run_dir, "pareto_objs.npy"), np.asarray(pareto_objs))
    if pareto_masks is not None:
        np.save(os.path.join(run_dir, "pareto_masks.npy"), np.asarray(pareto_masks))

    return train_front_rows, test_row, int(S_best.size)


# =========================================================
# 5) GBFS ENHANCED (OOP)
# =========================================================

def run_gbfs_enhanced_oop(
    dataset_idx: int,
    dataset_dir: str,
    runs: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[int], str]:

    enhanced_root = os.path.join(dataset_dir, "gbfs_enhanced")
    _ensure_dir(enhanced_root)

    # ---- hard bind OOP runtime to avoid baseline collisions
    _enter_runtime(OOP_ROOT, secondary_root=BASELINE_ROOT)

    # Import OOP stack fresh
    oop_core = _import("oop_core")
    runner_oop_mod = _import("runner_oop")
    myinput = _import("myinputdatasetXD").myinputdatasetXD

    ExperimentConfig = oop_core.ExperimentConfig
    InitParams = oop_core.InitParams
    KShellParams = oop_core.KShellParams
    BuddyParams = oop_core.BuddyParams
    EvalParams = oop_core.EvalParams
    LogParams = oop_core.LogParams
    GBFSRunner = runner_oop_mod.GBFSRunner

    # Patch runner_oop.newtry_ms to save pareto masks/objs to disk
    _ORIG_OOP_NEWTRY = runner_oop_mod.newtry_ms

    def _patched_oop_newtry(*args, **kwargs):
        res = _ORIG_OOP_NEWTRY(*args, **kwargs)
        run_dir = kwargs.get("run_dir", None)
        if run_dir is None and len(args) >= 4:
            run_dir = args[3]
        if run_dir and isinstance(res, tuple) and len(res) >= 3:
            try:
                np.save(os.path.join(run_dir, "pareto_masks.npy"), np.asarray(res[1]))
                np.save(os.path.join(run_dir, "pareto_objs.npy"),  np.asarray(res[2]))
            except Exception:
                pass
        return res

    runner_oop_mod.newtry_ms = _patched_oop_newtry

    cfg = ExperimentConfig(
        runs=runs,
        pop=BASE_POP,
        gen=BASE_GEN,

        init_modes=["knn", "probabilistic"],
        kshell_seq_modes=["normal", "rc_greedy"],
        post_seq_modes=["normal", "buddy"],

        log_init_mode="knn",
        log_kshell_seq_mode="rc_greedy",
        log_post_seq_mode="buddy",

        init_params=InitParams(
            k_neigh=BASE_KNEIGH,
            k_min=1,
            quantile=0.8,
            extra_k=2,
            beta=2.0,
            seed=42,
        ),

        kshell_params=KShellParams(
            max_add=5,
            rc_tau=0.3,
        ),

        buddy_params=BuddyParams(
            max_per_core=1,
            lam_red=0.5,
            cv=5,
            knn_k=BASE_KNEIGH,
            seed=42,
        ),

        eval_params=EvalParams(
            test_size=TEST_SIZE,
            split_seed=SPLIT_SEED,
            knn_eval_k=KNN_EVAL_K,
        ),

        log_params=LogParams(
            enabled=True,
            log_only_selected_combo=False,
            export_init_graph=True,
            export_solver_meta=True,
            export_solver_logs=True,
            export_post_logs=True,
            export_plots=False,
            keep_run_logs_in_memory=False,
        )
    )

    runner = GBFSRunner(cfg, visual_root=enhanced_root)
    product, dataset_name = runner.run_dataset(dataset_idx)

    _write_json(
        os.path.join(enhanced_root, f"{dataset_idx:02d}", "enhanced_cfg_snapshot.json"),
        asdict(cfg),
    )

    # rebuild split + zData for CV/red (use OOP dataset loader)
    dataset, labels, _ = myinput(int(dataset_idx))
    X_raw = np.asarray(dataset[:, 1:], dtype=float)
    y = np.asarray(labels, dtype=int).ravel()
    zData_full = _mapminmax_zero_one(X_raw)

    tr_idx, te_idx = _fixed_split_indices(X_raw, y, TEST_SIZE, SPLIT_SEED)
    Xtr = zData_full[tr_idx, :]
    ytr = y[tr_idx]
    Xte = zData_full[te_idx, :]
    yte = y[te_idx]

    init_mode = str(cfg.log_init_mode).lower()
    ks_mode   = str(cfg.log_kshell_seq_mode).lower()
    post_mode = str(cfg.log_post_seq_mode).lower()

    front_rows_all: List[Dict[str, Any]] = []
    test_rows_all: List[Dict[str, Any]] = []
    k_refs: List[int] = []

    for r in range(1, runs + 1):
        combo_dir = os.path.join(enhanced_root, f"{dataset_idx:02d}", f"run_{r:02d}", init_mode, f"ks_{ks_mode}")
        post_dir  = os.path.join(combo_dir, f"post_{post_mode}")

        # TRAIN front from pareto_masks.npy (patched)
        pm_path = os.path.join(combo_dir, "pareto_masks.npy")
        fsets: List[np.ndarray] = []
        if os.path.exists(pm_path):
            pm = np.load(pm_path, allow_pickle=True)
            pm = np.asarray(pm)
            if pm.ndim == 1 and pm.size > 0:
                pm = pm.reshape(1, -1)
            if pm.ndim == 2:
                for m in pm:
                    S = np.where(np.asarray(m) != 0)[0].astype(int)
                    if S.size > 0:
                        fsets.append(S)

        if not fsets:
            s0_path = os.path.join(combo_dir, "S0_selected.txt")
            if os.path.exists(s0_path):
                txt = open(s0_path, "r", encoding="utf-8").read().strip()
                if txt:
                    S0 = np.array([int(x) for x in txt.split()], dtype=int)
                    if S0.size > 0:
                        fsets = [S0]

        front_rows = _front_train_rows(
            dataset_idx=dataset_idx,
            dataset_name=str(dataset_name),
            method="GBFS_enhanced",
            run_id=r,
            feature_sets=fsets if fsets else [np.array([], int)],
            X_train_for_cv=Xtr,
            y_train=ytr,
            zData_full_for_red=zData_full,
        )
        front_rows_all.extend(front_rows)

        # TEST summary from post_metrics.json + selected_features.txt
        pm_json = os.path.join(post_dir, "post_metrics.json")
        sel_txt = os.path.join(post_dir, "selected_features.txt")

        if os.path.exists(pm_json):
            meta = json.load(open(pm_json, "r", encoding="utf-8"))
            acc_test = float(meta.get("acc_test", 0.0))
            redv = float(meta.get("red", 0.0))
            fnum = int(meta.get("fnum", 0))
            time_total = float(meta.get("time_total", meta.get("time_solver", 0.0) + meta.get("time_post_extra", 0.0)))

            if os.path.exists(sel_txt):
                txt = open(sel_txt, "r", encoding="utf-8").read().strip()
                S = np.array([int(x) for x in txt.split()], dtype=int) if txt else np.array([], int)
            else:
                S = np.array([], int)

            k_refs.append(int(S.size if S.size else fnum))

            test_rows_all.append({
                "dataset_idx": int(dataset_idx),
                "dataset": str(dataset_name),
                "method": "GBFS_enhanced",
                "run": int(r),
                "test_mode": "selected",
                "ratio": "",
                "k_ref_used": "",
                "acc_test": float(acc_test),
                "eRate_test": float(1.0 - acc_test),
                "fnum": int(S.size if S.size else fnum),
                "fRatio": float((S.size if S.size else fnum) / zData_full.shape[1]),
                "red": float(redv),
                "time_total": float(time_total),
                "fset": " ".join(map(str, S.tolist())),
                "note": "from post_metrics.json",
            })
        else:
            k_refs.append(0)
            test_rows_all.append({
                "dataset_idx": int(dataset_idx),
                "dataset": str(dataset_name),
                "method": "GBFS_enhanced",
                "run": int(r),
                "test_mode": "selected",
                "ratio": "",
                "k_ref_used": "",
                "acc_test": 0.0,
                "eRate_test": 1.0,
                "fnum": 0,
                "fRatio": 0.0,
                "red": 0.0,
                "time_total": 0.0,
                "fset": "",
                "note": "missing post_metrics.json",
            })

    return front_rows_all, test_rows_all, k_refs, str(dataset_name)


# =========================================================
# 6) SCORE-BASED METHODS
# =========================================================

def run_score_based_methods_one_run(
    dataset_idx: int,
    dataset_name: str,
    X_raw: np.ndarray,
    y: np.ndarray,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    dataset_dir: str,
    run_id: int,
    k_ref: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

    # bind baseline runtime (traditional_fs / graph_fs live there)
    _enter_runtime(BASELINE_ROOT, secondary_root=OOP_ROOT)

    HAS_TRAD = os.path.exists(os.path.join(BASELINE_ROOT, "traditional_fs.py"))
    HAS_GRAPH = os.path.exists(os.path.join(BASELINE_ROOT, "graph_fs.py"))

    run_all_filters = None
    run_embedded_methods = None
    run_graph_fs_methods = None

    if HAS_TRAD:
        try:
            trad_mod = _import("traditional_fs")
            run_all_filters = trad_mod.run_all_filters
            run_embedded_methods = trad_mod.run_embedded_methods
        except Exception:
            HAS_TRAD = False

    if HAS_GRAPH:
        try:
            graph_mod = _import("graph_fs")
            run_graph_fs_methods = graph_mod.run_graph_fs_methods
        except Exception:
            HAS_GRAPH = False

    if not (HAS_TRAD or HAS_GRAPH):
        return [], []

    root = os.path.join(dataset_dir, "traditional_graph")
    _ensure_dir(root)

    X_raw = np.asarray(X_raw)
    if np.iscomplexobj(X_raw):
        X_raw = np.real(X_raw)
    X_raw = np.asarray(X_raw, dtype=float)

    y = np.asarray(y, dtype=int).ravel()

    Xtr_raw = X_raw[tr_idx, :]
    Xte_raw = X_raw[te_idx, :]
    ytr = y[tr_idx]
    yte = y[te_idx]

    # evaluation style for these: StandardScaler
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr_raw)
    Xte = scaler.transform(Xte_raw)

    # redundancy on minmax full
    zData_full = _mapminmax_zero_one(X_raw)

    n_features = Xtr.shape[1]
    k_ref = max(1, min(int(k_ref) if k_ref else 1, n_features))

    # raw scores (can be complex)
    raw_scores: Dict[str, np.ndarray] = {}

    t0 = time.perf_counter()
    if HAS_TRAD and run_all_filters and run_embedded_methods:
        f_scores, _ = run_all_filters(Xtr, ytr, k=n_features)
        e_scores, _ = run_embedded_methods(Xtr, ytr, k=n_features)
        raw_scores.update({str(k): np.asarray(v).ravel() for k, v in f_scores.items()})
        raw_scores.update({str(k): np.asarray(v).ravel() for k, v in e_scores.items()})

    if HAS_GRAPH and run_graph_fs_methods:
        g_scores, _ = run_graph_fs_methods(Xtr, ytr, k=n_features, use_inffs=True, use_ugfs=True)
        raw_scores.update({str(k): np.asarray(v).ravel() for k, v in g_scores.items()})

    score_time = float(time.perf_counter() - t0)

    # ---- sanitize into usable scores (real + finite)
    scores: Dict[str, np.ndarray] = {k: _sanitize_scores(v, mode="abs") for k, v in raw_scores.items()}

    front_train_all: List[Dict[str, Any]] = []
    test_points_all: List[Dict[str, Any]] = []

    # ALL features point
    Sall = np.arange(n_features, dtype=int)

    front_train_all.extend(_front_train_rows(
        dataset_idx, dataset_name, "ALL_features", run_id,
        [Sall], Xtr, ytr, zData_full,
        extra_cols={"score_time": 0.0}
    ))

    t1 = time.perf_counter()
    test_points_all.append(_test_point_row(
        dataset_idx, dataset_name, "ALL_features", run_id,
        test_mode="all",
        S=Sall,
        Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
        zData_full_for_red=zData_full,
        time_total=float(time.perf_counter() - t1),
        ratio=1.0,
        extra={"score_time": 0.0}
    ))

    for mname, s in scores.items():
        # s is already real finite
        ranking = np.argsort(-np.abs(s))

        # ---- TRAIN front ratios (CV)
        fsets_front = []
        for ratio in RATIOS:
            k = max(1, min(int(round(ratio * n_features)), n_features))
            fsets_front.append(ranking[:k].astype(int))

        t_front = time.perf_counter()
        rows_front = _front_train_rows(
            dataset_idx, dataset_name, mname, run_id,
            fsets_front, Xtr, ytr, zData_full,
            extra_cols={"score_time": score_time}
        )
        front_time = float(time.perf_counter() - t_front)
        for rr in rows_front:
            rr["front_eval_time"] = front_time
        front_train_all.extend(rows_front)

        # ---- TEST ratios
        for ratio in RATIOS:
            k = max(1, min(int(round(ratio * n_features)), n_features))
            S = ranking[:k].astype(int)

            t_eval = time.perf_counter()
            acc = _knn_acc_test(Xtr, ytr, Xte, yte, S)
            eval_time = float(time.perf_counter() - t_eval)

            test_points_all.append({
                "dataset_idx": int(dataset_idx),
                "dataset": str(dataset_name),
                "method": str(mname),
                "run": int(run_id),
                "test_mode": "ratio",
                "ratio": float(ratio),
                "k_ref_used": "",
                "acc_test": float(acc),
                "eRate_test": float(1.0 - acc),
                "fnum": int(S.size),
                "fRatio": float(S.size / n_features),
                "red": float(_red(S, zData_full)),
                "time_total": float(score_time + eval_time),
                "score_time": float(score_time),
                "eval_time": float(eval_time),
                "fset": " ".join(map(str, S.tolist())),
            })

        # ---- TEST k_ref
        Sref = ranking[:k_ref].astype(int)
        t_eval = time.perf_counter()
        acc_ref = _knn_acc_test(Xtr, ytr, Xte, yte, Sref)
        eval_time = float(time.perf_counter() - t_eval)

        test_points_all.append({
            "dataset_idx": int(dataset_idx),
            "dataset": str(dataset_name),
            "method": str(mname),
            "run": int(run_id),
            "test_mode": "k_ref",
            "ratio": "",
            "k_ref_used": int(k_ref),
            "acc_test": float(acc_ref),
            "eRate_test": float(1.0 - acc_ref),
            "fnum": int(Sref.size),
            "fRatio": float(Sref.size / n_features),
            "red": float(_red(Sref, zData_full)),
            "time_total": float(score_time + eval_time),
            "score_time": float(score_time),
            "eval_time": float(eval_time),
            "fset": " ".join(map(str, Sref.tolist())),
        })

        # per-method logs
        method_dir = os.path.join(root, mname, f"run_{run_id:02d}")
        _ensure_dir(method_dir)
        _write_csv(os.path.join(method_dir, "front_train_cv.csv"), rows_front)
        tp = [r for r in test_points_all if r.get("method") == mname and r.get("run") == run_id]
        _write_csv(os.path.join(method_dir, "test_points.csv"), tp)
        _write_json(os.path.join(method_dir, "run_meta.json"), {
            "dataset_idx": int(dataset_idx),
            "dataset": str(dataset_name),
            "method": str(mname),
            "run": int(run_id),
            "k_ref_used": int(k_ref),
            "score_time": float(score_time),
            "front_eval_time": float(front_time),
            "note": "Scores sanitized to real (abs for complex) to avoid casting warning",
        })

    return front_train_all, test_points_all


# =========================================================
# 7) MAIN
# =========================================================

def main() -> None:
    _ensure_dir(OUT_ROOT)

    global_front_train: List[Dict[str, Any]] = []
    global_test_points: List[Dict[str, Any]] = []

    for data_idx in DATASET_INDICES:
        dataset_dir = os.path.join(OUT_ROOT, f"dataset_{int(data_idx):02d}")
        _ensure_dir(dataset_dir)

        # Load dataset using baseline runtime (safe; we only keep arrays afterward)
        _enter_runtime(BASELINE_ROOT, secondary_root=OOP_ROOT)
        myinput = _import("myinputdatasetXD").myinputdatasetXD

        dataset, labels, datasetName = myinput(int(data_idx))
        X_raw = np.asarray(dataset[:, 1:], dtype=float)
        y = np.asarray(labels, dtype=int).ravel()

        n_samples, n_features = X_raw.shape
        n_classes = int(len(np.unique(y)))

        tr_idx, te_idx = _fixed_split_indices(X_raw, y, TEST_SIZE, SPLIT_SEED)

        _write_json(os.path.join(dataset_dir, "dataset_info.json"), {
            "dataset_idx": int(data_idx),
            "dataset": str(datasetName),
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "n_classes": int(n_classes),
        })
        _write_json(os.path.join(dataset_dir, "split_indices.json"), {
            "test_size": float(TEST_SIZE),
            "split_seed": int(SPLIT_SEED),
            "train_idx": tr_idx.tolist(),
            "test_idx": te_idx.tolist(),
        })

        # ---------- (A) GBFS Enhanced ----------
        enh_front_rows: List[Dict[str, Any]] = []
        enh_test_rows: List[Dict[str, Any]] = []
        enh_k_refs: List[int] = []

        try:
            enh_front_rows, enh_test_rows, enh_k_refs, _ = run_gbfs_enhanced_oop(
                dataset_idx=int(data_idx),
                dataset_dir=dataset_dir,
                runs=RUNS,
            )
            _write_csv(os.path.join(dataset_dir, "gbfs_enhanced_front_train_cv.csv"), enh_front_rows)
            _write_csv(os.path.join(dataset_dir, "gbfs_enhanced_test_points.csv"), enh_test_rows)

            global_front_train.extend(enh_front_rows)
            global_test_points.extend(enh_test_rows)
        except Exception as e:
            _write_text(os.path.join(dataset_dir, "gbfs_enhanced_error.txt"), str(e))
            enh_k_refs = [0] * RUNS

        # ---------- (B) GBFS Baseline ----------
        base_root = os.path.join(dataset_dir, "gbfs_baseline")
        _ensure_dir(base_root)

        base_k_fallback: List[int] = []
        for r in range(1, RUNS + 1):
            try:
                front_rows, test_row, fnum_best = run_gbfs_baseline_one_run(
                    dataset_idx=int(data_idx),
                    dataset_name=str(datasetName),
                    X_raw=X_raw,
                    y=y,
                    tr_idx=tr_idx,
                    te_idx=te_idx,
                    out_dir=base_root,
                    run_id=r,
                    delt=BASE_DELT,
                    omega=BASE_OMEGA,
                    kNeigh=BASE_KNEIGH,
                    pop=BASE_POP,
                    gen=BASE_GEN,
                )
                base_k_fallback.append(int(fnum_best))

                _write_csv(os.path.join(base_root, f"run_{r:02d}", "front_train_cv.csv"), front_rows)
                _write_json(os.path.join(base_root, f"run_{r:02d}", "test_point.json"), test_row)

                global_front_train.extend(front_rows)
                global_test_points.append(test_row)
            except Exception as e:
                _write_text(os.path.join(base_root, f"run_{r:02d}", "error.txt"), str(e))
                base_k_fallback.append(0)

        # ---------- (C) Score-based methods ----------
        for r in range(1, RUNS + 1):
            k_ref = int(enh_k_refs[r - 1]) if (enh_k_refs and (r - 1) < len(enh_k_refs)) else 0
            if k_ref <= 0:
                k_ref = int(base_k_fallback[r - 1]) if (base_k_fallback and (r - 1) < len(base_k_fallback)) else 1
            if k_ref <= 0:
                k_ref = 1

            try:
                front_rows_trad, test_points_trad = run_score_based_methods_one_run(
                    dataset_idx=int(data_idx),
                    dataset_name=str(datasetName),
                    X_raw=X_raw,
                    y=y,
                    tr_idx=tr_idx,
                    te_idx=te_idx,
                    dataset_dir=dataset_dir,
                    run_id=r,
                    k_ref=k_ref,
                )
                global_front_train.extend(front_rows_trad)
                global_test_points.extend(test_points_trad)
            except Exception as e:
                _write_text(os.path.join(dataset_dir, "traditional_graph", f"run_{r:02d}_error.txt"), str(e))

    _write_csv(os.path.join(OUT_ROOT, "compare_front_train.csv"), global_front_train)
    _write_csv(os.path.join(OUT_ROOT, "compare_test_points.csv"), global_test_points)

    print("DONE.")
    print("TRAIN fronts:", os.path.join(OUT_ROOT, "compare_front_train.csv"))
    print("TEST points :", os.path.join(OUT_ROOT, "compare_test_points.csv"))


if __name__ == "__main__":
    main()
