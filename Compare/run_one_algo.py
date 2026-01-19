from __future__ import annotations

import os
import sys
import json
import csv
import time
import argparse
import importlib
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# -----------------------------
# ref_sizes lookup for AUTO fixed_fnum (when --fixed_fnum True/auto)
# -----------------------------
_REF_CACHE: dict[str, dict[tuple[int, int], int]] = {}

def _default_ref_sizes_path() -> str:
    try:
        return str(Path(__file__).with_name("ref_sizes.csv"))
    except Exception:
        return "ref_sizes.csv"

def _load_ref_sizes(ref_csv_path: str) -> dict[tuple[int, int], int]:
    apath = os.path.abspath(ref_csv_path)
    if apath in _REF_CACHE:
        return _REF_CACHE[apath]

    mp: dict[tuple[int, int], int] = {}
    with open(apath, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                did = int(float(str(row.get("dataset_idx", "")).strip()))
                run_id = int(float(str(row.get("run", "")).strip()))
                k = int(round(float(str(row.get("target_fnum", "")).strip())))
            except Exception:
                continue
            if did > 0 and run_id > 0 and k > 0:
                mp[(did, run_id)] = k

    _REF_CACHE[apath] = mp
    return mp

def _auto_fixed_fnum_from_ref(dataset_idx: int, run_id: int, ref_csv_path: str) -> int | None:
    mp = _load_ref_sizes(ref_csv_path)
    return mp.get((dataset_idx, run_id))


# -----------------------------
# utils I/O
# -----------------------------
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
    _ensure_dir(os.path.dirname(path))
    if not rows:
        return
    if fieldnames is None:
        fieldnames, seen = [], set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def _algo_tag(algo: str) -> str:
    # safe folder name
    return algo.replace(":", "__").replace("/", "_").replace("\\", "_").replace(" ", "_")

def _fmt_float_tag(x: float) -> str:
    # 0.5 -> "0p5", 2.0 -> "2", 0.25 -> "0p25"
    try:
        s = f"{float(x):g}"
    except Exception:
        s = str(x)
    s = s.replace("-", "m").replace(".", "p")
    return s


# -----------------------------
# collision control (optional but safe)
# -----------------------------
_PROJECT_NAMES = [
    "gbfs_globals", "myinputdatasetXD", "fisherScore", "helper",
    "newtry_ms", "copy_of_en_nsga_2_mating_strategy",
    "initialize_variables_f", "decodeNet", "encodeNet",
    "evaluate_objective_f", "evaluate_objective_f2",
    "graph_fs", "traditional_fs",
    "runner_oop", "oop_core", "sequential_strategies",
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
    if secondary_root:
        _purge_modules_from_root(secondary_root)
    _purge_modules_from_root(primary_root)
    _purge_module_names(_PROJECT_NAMES)
    _prioritize_path(primary_root)

def _import(name: str):
    return importlib.import_module(name)


# -----------------------------
# math helpers
# -----------------------------
def _mapminmax_zero_one(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
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

def _cv_acc_knn(X: np.ndarray, y: np.ndarray, S: np.ndarray, cv: int, seed: int, knn_k: int) -> float:
    S = np.asarray(S, dtype=int).ravel()
    if S.size == 0:
        return 0.0
    Xs = X[:, S]
    y = np.asarray(y, dtype=int).ravel()
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    accs = []
    for tr, va in skf.split(Xs, y):
        clf = KNeighborsClassifier(n_neighbors=knn_k)
        clf.fit(Xs[tr], y[tr])
        pred = clf.predict(Xs[va])
        accs.append(np.mean(pred == y[va]))
    return float(np.mean(accs)) if accs else 0.0

def _knn_acc_test(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray, S: np.ndarray, knn_k: int) -> float:
    S = np.asarray(S, dtype=int).ravel()
    if S.size == 0:
        return 0.0
    clf = KNeighborsClassifier(n_neighbors=knn_k)
    clf.fit(Xtr[:, S], ytr)
    pred = clf.predict(Xte[:, S])
    return float(np.mean(pred == yte))

def _sanitize_scores(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s).ravel()
    if np.iscomplexobj(s):
        s = np.abs(s)
    s = s.astype(float, copy=False)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    return s


# -----------------------------
# row builders
# -----------------------------
def _front_train_rows(
    dataset_idx: int,
    dataset_name: str,
    method: str,
    run_id: int,
    feature_sets: List[np.ndarray],
    X_train_for_cv: np.ndarray,
    y_train: np.ndarray,
    zData_full_for_red: np.ndarray,
    cv_folds: int,
    cv_seed: int,
    knn_k: int,
    extra_cols: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:

    n_features = X_train_for_cv.shape[1]
    best_by_fnum: Dict[int, Dict[str, Any]] = {}

    for sid, S in enumerate(feature_sets):
        S = np.asarray(S, dtype=int).ravel()
        if S.size == 0:
            continue

        acc_cv = _cv_acc_knn(X_train_for_cv, y_train, S, cv=cv_folds, seed=cv_seed, knn_k=knn_k)
        er_cv = float(1.0 - acc_cv)
        fnum = int(S.size)
        fr = float(fnum / n_features) if n_features else 0.0
        redv = _redundancy_rate_subset_fallback(S, zData_full_for_red)

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
    rows.sort(key=lambda d: float(d.get("fRatio", 0.0)))
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
    knn_k: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    S = np.asarray(S, dtype=int).ravel()
    acc = _knn_acc_test(Xtr, ytr, Xte, yte, S, knn_k=knn_k) if S.size else 0.0

    row = {
        "dataset_idx": int(dataset_idx),
        "dataset": str(dataset_name),
        "method": str(method),
        "run": int(run_id),
        "test_mode": str(test_mode),
        "acc_test": float(acc),
        "eRate_test": float(1.0 - acc),
        "fnum": int(S.size),
        "fRatio": float(S.size / Xtr.shape[1]) if Xtr.shape[1] else 0.0,
        "red": float(_redundancy_rate_subset_fallback(S, zData_full_for_red)) if S.size else 0.0,
        "time_total": float(time_total),
        "fset": " ".join(map(str, S.tolist())),
    }
    if extra:
        row.update(extra)
    return row


# -----------------------------
# runners
# -----------------------------
def run_enhanced_job(args, run_dir: str, seed_run: int) -> None:
    # parse enh:init:ks
    _, init_mode, ks_mode = args.algo.split(":", 2)
    method = f"GBFS_enhanced__init_{init_mode}__ks_{ks_mode}"

    _enter_runtime(args.oop_root, secondary_root=args.baseline_root)

    oop_core = _import("oop_core")
    runner_oop_mod = _import("runner_oop")
    myinput = _import("myinputdatasetXD").myinputdatasetXD

    ExperimentConfig = oop_core.ExperimentConfig
    InitParams = oop_core.InitParams
    KShellParams = oop_core.KShellParams
    BuddyParams = oop_core.BuddyParams
    LogParams = oop_core.LogParams
    GBFSRunner = runner_oop_mod.GBFSRunner

    # sensitivity params (default = paper)
    prob_beta = float(getattr(args, "prob_beta", 2.0))
    buddy_lambda = float(getattr(args, "buddy_lambda", 0.5))

    # IMPORTANT: runner_oop uses cfg.test_size/cfg.split_seed/cfg.knn_eval_k (top-level)
    cfg = ExperimentConfig(
        runs=1,
        test_size=float(args.test_size),
        split_seed=int(args.split_seed),

        pop=int(args.base_pop),
        gen=int(args.base_gen),
        knn_eval_k=int(args.knn_eval_k),

        init_modes=[init_mode],
        kshell_seq_modes=[ks_mode],
        post_seq_modes=["normal", "buddy"],

        log_init_mode=init_mode,
        log_kshell_seq_mode=ks_mode,
        log_post_seq_mode="buddy",

        init_params=InitParams(
            k_neigh=int(args.base_kneigh),
            k_min=1,
            quantile=0.8,
            extra_k=2,
            beta=float(prob_beta),          # <-- was 2.0
            seed=int(seed_run),
        ),
        kshell_params=KShellParams(
            max_add=int(args.ks_max_add),
            rc_tau=float(args.ks_rc_tau),   # <-- tau sweep already supported
        ),
        buddy_params=BuddyParams(
            max_per_core=1,
            lam_red=float(buddy_lambda),    # <-- was 0.5
            cv=int(args.cv_folds),
            knn_k=int(args.base_kneigh),
            seed=int(seed_run),
        ),
        log_params=LogParams(
            enabled=True,
            log_only_selected_combo=False,
            export_init_graph=False,
            export_solver_meta=True,
            export_solver_logs=False,  # VERY IMPORTANT: avoid 50+ gen files
            export_post_logs=True,
            export_plots=False,
            keep_run_logs_in_memory=False,
        )
    )

    enhanced_root = os.path.join(run_dir, "gbfs_enhanced_raw")
    runner = GBFSRunner(cfg, visual_root=enhanced_root)
    product, dataset_name = runner.run_dataset(int(args.dataset_idx))

    # Load raw dataset to recompute front/test consistently
    dataset, labels, _ = myinput(int(args.dataset_idx))
    X_raw = np.asarray(dataset[:, 1:], dtype=float)
    y = np.asarray(labels, dtype=int).ravel()

    zData_full = _mapminmax_zero_one(X_raw)
    tr_idx, te_idx = _fixed_split_indices(X_raw, y, float(args.test_size), int(args.split_seed))
    Xtr = zData_full[tr_idx, :]
    ytr = y[tr_idx]
    Xte = zData_full[te_idx, :]
    yte = y[te_idx]

    # Locate combo dir
    # runner_oop always uses run_01 when cfg.runs=1
    combo_dir = os.path.join(enhanced_root, f"{int(args.dataset_idx):02d}", "run_01", init_mode, f"ks_{ks_mode}")

    # Build feature sets from pareto_masks.npy
    fsets: List[np.ndarray] = []
    pm_path = os.path.join(combo_dir, "pareto_masks.npy")
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
        # fallback: S0_selected.txt
        s0_path = os.path.join(combo_dir, "S0_selected.txt")
        if os.path.exists(s0_path):
            txt = open(s0_path, "r", encoding="utf-8").read().strip()
            if txt:
                S0 = np.array([int(x) for x in txt.split()], dtype=int)
                if S0.size > 0:
                    fsets = [S0]

    front_rows = _front_train_rows(
        dataset_idx=int(args.dataset_idx),
        dataset_name=str(dataset_name),
        method=method,
        run_id=int(args.run),
        feature_sets=fsets,
        X_train_for_cv=Xtr,
        y_train=ytr,
        zData_full_for_red=zData_full,
        cv_folds=int(args.cv_folds),
        cv_seed=int(args.cv_seed),
        knn_k=int(args.knn_eval_k),
        extra_cols={
            "init_mode": init_mode,
            "ks_mode": ks_mode,
            "prob_beta": float(prob_beta),
            "ks_rc_tau": float(args.ks_rc_tau),
            "buddy_lambda": float(buddy_lambda),
        },
    )

    test_rows: List[Dict[str, Any]] = []
    for post_mode in ["normal", "buddy"]:
        post_dir = os.path.join(combo_dir, f"post_{post_mode}")
        sel_txt = os.path.join(post_dir, "selected_features.txt")
        pm_json = os.path.join(post_dir, "post_metrics.json")

        S = np.array([], dtype=int)
        if os.path.exists(sel_txt):
            txt = open(sel_txt, "r", encoding="utf-8").read().strip()
            if txt:
                S = np.array([int(x) for x in txt.split()], dtype=int)

        t_total = 0.0
        if os.path.exists(pm_json):
            meta = json.load(open(pm_json, "r", encoding="utf-8"))
            t_total = float(meta.get("time_total", meta.get("time_solver", 0.0) + meta.get("time_post_extra", 0.0)))

        test_rows.append(_test_point_row(
            dataset_idx=int(args.dataset_idx),
            dataset_name=str(dataset_name),
            method=method,
            run_id=int(args.run),
            test_mode="selected",
            S=S,
            Xtr=Xtr, ytr=ytr,
            Xte=Xte, yte=yte,
            zData_full_for_red=zData_full,
            time_total=t_total,
            knn_k=int(args.knn_eval_k),
            extra={
                "init_mode": init_mode,
                "ks_mode": ks_mode,
                "post_mode": post_mode,
                "prob_beta": float(prob_beta),
                "ks_rc_tau": float(args.ks_rc_tau),
                "buddy_lambda": float(buddy_lambda),
            },
        ))

    _write_csv(os.path.join(run_dir, "front_train_cv.csv"), front_rows)
    _write_csv(os.path.join(run_dir, "test_points.csv"), test_rows)
    _write_json(os.path.join(run_dir, "run_meta.json"), {
        "algo": args.algo,
        "method": method,
        "dataset_idx": int(args.dataset_idx),
        "dataset": str(dataset_name),
        "run": int(args.run),
        "seed_run": int(seed_run),
        "cfg": {
            "test_size": float(args.test_size),
            "split_seed": int(args.split_seed),
            "cv_folds": int(args.cv_folds),
            "cv_seed": int(args.cv_seed),
            "knn_eval_k": int(args.knn_eval_k),
            "pop": int(args.base_pop),
            "gen": int(args.base_gen),
            "k_neigh": int(args.base_kneigh),
            "ks_max_add": int(args.ks_max_add),
            "ks_rc_tau": float(args.ks_rc_tau),
            "prob_beta": float(prob_beta),
            "buddy_lambda": float(buddy_lambda),
        },
        "paths": {"combo_dir": combo_dir}
    })


def run_baseline_job(args, run_dir: str, seed_run: int) -> None:
    method = "GBFS_baseline"
    _enter_runtime(args.baseline_root, secondary_root=args.oop_root)

    GG = _import("gbfs_globals")
    fisherScore = _import("fisherScore").fisherScore
    baseline_newtry_ms = _import("newtry_ms").newtry_ms
    myinput = _import("myinputdatasetXD").myinputdatasetXD

    dataset, labels, datasetName = myinput(int(args.dataset_idx))
    X_raw = np.asarray(dataset[:, 1:], dtype=float)
    y = np.asarray(labels, dtype=int).ravel()

    tr_idx, te_idx = _fixed_split_indices(X_raw, y, float(args.test_size), int(args.split_seed))
    zData_full = _mapminmax_zero_one(X_raw)

    tr_mask = np.zeros(zData_full.shape[0], dtype=bool)
    tr_mask[tr_idx] = True

    # set globals
    GG.DELT = float(args.base_delt)
    GG.OMEGA = float(args.base_omega)
    GG.data = zData_full
    GG.label = y
    GG.featNum = zData_full.shape[1]
    GG.kNeigh = int(args.base_kneigh)

    GG.trIdx = tr_mask
    GG.trData = zData_full[tr_mask, :]
    GG.trLabel = y[tr_mask]
    GG.teData = zData_full[~tr_mask, :]
    GG.teLabel = y[~tr_mask]
    GG.assiNumInside = []

    tic = time.perf_counter()

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

    out = baseline_newtry_ms(kNeiAdj, int(args.base_pop), int(args.base_gen))
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

    # feature sets for front
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

    front_rows = _front_train_rows(
        dataset_idx=int(args.dataset_idx),
        dataset_name=str(datasetName),
        method=method,
        run_id=int(args.run),
        feature_sets=fsets,
        X_train_for_cv=GG.trData,
        y_train=GG.trLabel,
        zData_full_for_red=zData_full,
        cv_folds=int(args.cv_folds),
        cv_seed=int(args.cv_seed),
        knn_k=int(args.knn_eval_k),
    )

    test_rows = [_test_point_row(
        dataset_idx=int(args.dataset_idx),
        dataset_name=str(datasetName),
        method=method,
        run_id=int(args.run),
        test_mode="selected",
        S=S_best,
        Xtr=GG.trData, ytr=GG.trLabel,
        Xte=GG.teData, yte=GG.teLabel,
        zData_full_for_red=zData_full,
        time_total=t_total,
        knn_k=int(args.knn_eval_k),
        extra={"assiNum": float(np.sum(np.asarray(getattr(GG, "assiNumInside", []), dtype=float))) if hasattr(GG, "assiNumInside") else 0.0},
    )]

    _write_text(os.path.join(run_dir, "selected_features.txt"), " ".join(map(str, S_best.tolist())))
    _write_csv(os.path.join(run_dir, "front_train_cv.csv"), front_rows)
    _write_csv(os.path.join(run_dir, "test_points.csv"), test_rows)
    _write_json(os.path.join(run_dir, "run_meta.json"), {
        "algo": args.algo,
        "method": method,
        "dataset_idx": int(args.dataset_idx),
        "dataset": str(datasetName),
        "run": int(args.run),
        "seed_run": int(seed_run),
        "time_total": float(t_total),
        "params": {"delt": float(args.base_delt), "omega": float(args.base_omega), "kNeigh": int(args.base_kneigh), "pop": int(args.base_pop), "gen": int(args.base_gen)},
    })


def run_traditional_job(args, run_dir: str, seed_run: int) -> None:
    # algo = trad:METHOD
    _, method_name = args.algo.split(":", 1)
    method = method_name

    _enter_runtime(args.baseline_root, secondary_root=args.oop_root)
    myinput = _import("myinputdatasetXD").myinputdatasetXD
    trad = _import("traditional_fs")

    dataset, labels, datasetName = myinput(int(args.dataset_idx))
    X_raw = np.asarray(dataset[:, 1:], dtype=float)
    y = np.asarray(labels, dtype=int).ravel()

    tr_idx, te_idx = _fixed_split_indices(X_raw, y, float(args.test_size), int(args.split_seed))
    Xtr_raw = X_raw[tr_idx, :]
    Xte_raw = X_raw[te_idx, :]
    ytr = y[tr_idx]
    yte = y[te_idx]

    # consistent with your previous compare: scale once on full train
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr_raw)
    Xte = scaler.transform(Xte_raw)

    zData_full = _mapminmax_zero_one(X_raw)
    n_features = Xtr.shape[1]
    fixed_k = int(getattr(args, "fixed_fnum", 0))
    if fixed_k > 0:
        ratios = []
    else:
        ratios = [float(x) for x in str(args.ratios).split(",") if str(x).strip()]

    t0 = time.perf_counter()

    # 1) FILTER / EMBEDDED => score vector
    score_vec = None
    if hasattr(trad, "UNIVARIATE_FILTERS") and method in trad.UNIVARIATE_FILTERS:
        func = trad.UNIVARIATE_FILTERS[method]
        X_input = Xtr
        if "chi2" in method.lower():
            min_val = X_input.min()
            if min_val < 0:
                X_input = X_input - min_val + 1e-9
        score_vec = func(X_input, ytr)

    elif hasattr(trad, "EMBEDDED_METHODS") and method in trad.EMBEDDED_METHODS:
        func = trad.EMBEDDED_METHODS[method]
        score_vec = func(Xtr, ytr)

    score_time = float(time.perf_counter() - t0)

    front_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    if score_vec is not None:
        s = _sanitize_scores(np.asarray(score_vec).ravel())
        ranking = np.argsort(-np.abs(s))

        fsets = []
        if fixed_k > 0:
            k = max(1, min(fixed_k, n_features))
            fsets.append(ranking[:k].astype(int))
        else:
            for ratio in ratios:
                k = max(1, min(int(round(ratio * n_features)), n_features))
                fsets.append(ranking[:k].astype(int))

        t_front = time.perf_counter()
        front_rows = _front_train_rows(
            dataset_idx=int(args.dataset_idx),
            dataset_name=str(datasetName),
            method=method,
            run_id=int(args.run),
            feature_sets=fsets,
            X_train_for_cv=Xtr,
            y_train=ytr,
            zData_full_for_red=zData_full,
            cv_folds=int(args.cv_folds),
            cv_seed=int(args.cv_seed),
            knn_k=int(args.knn_eval_k),
            extra_cols={"ratio_grid": str(args.ratios), "score_time": score_time},
        )
        front_eval_time = float(time.perf_counter() - t_front)
        for rr in front_rows:
            rr["front_eval_time"] = front_eval_time

        # test points
        if fixed_k > 0:
            k = max(1, min(fixed_k, n_features))
            S = ranking[:k].astype(int)
            acc = _knn_acc_test(Xtr, ytr, Xte, yte, S, knn_k=int(args.knn_eval_k))
            test_rows.append({
                "dataset_idx": int(args.dataset_idx),
                "dataset": str(datasetName),
                "method": method,
                "run": int(args.run),
                "test_mode": "fixed_k",
                "fixed_fnum": int(k),
                "acc_test": float(acc),
                "eRate_test": float(1.0 - acc),
                "fnum": int(S.size),
                "fRatio": float(S.size / n_features),
                "red": float(_redundancy_rate_subset_fallback(S, zData_full)),
                "time_total": float(score_time),
                "fset": " ".join(map(str, S.tolist())),
            })
        else:
            for ratio in ratios:
                k = max(1, min(int(round(ratio * n_features)), n_features))
                S = ranking[:k].astype(int)
                t1 = time.perf_counter()
                acc = _knn_acc_test(Xtr, ytr, Xte, yte, S, knn_k=int(args.knn_eval_k))
                eval_time = float(time.perf_counter() - t1)

                test_rows.append({
                    "dataset_idx": int(args.dataset_idx),
                    "dataset": str(datasetName),
                    "method": method,
                    "run": int(args.run),
                    "test_mode": "ratio",
                    "ratio": float(ratio),
                    "acc_test": float(acc),
                    "eRate_test": float(1.0 - acc),
                    "fnum": int(S.size),
                    "fRatio": float(S.size / n_features),
                    "red": float(_redundancy_rate_subset_fallback(S, zData_full)),
                    "time_total": float(score_time + eval_time),
                    "score_time": float(score_time),
                    "eval_time": float(eval_time),
                    "fset": " ".join(map(str, S.tolist())),
                })

    else:
        if hasattr(trad, "WRAPPER_METHODS") and method in trad.WRAPPER_METHODS:
            func = trad.WRAPPER_METHODS[method]
            fsets = []
            t1 = time.perf_counter()
            for ratio in ratios:
                k = max(1, min(int(round(ratio * n_features)), n_features))
                if "SFS" in method:
                    idx = func(Xtr, ytr, k=k, cv=int(args.cv_folds))
                else:
                    idx = func(Xtr, ytr, k=k)
                fsets.append(np.asarray(idx, dtype=int))
            score_time = float(time.perf_counter() - t1)

            front_rows = _front_train_rows(
                dataset_idx=int(args.dataset_idx),
                dataset_name=str(datasetName),
                method=method,
                run_id=int(args.run),
                feature_sets=fsets,
                X_train_for_cv=Xtr,
                y_train=ytr,
                zData_full_for_red=zData_full,
                cv_folds=int(args.cv_folds),
                cv_seed=int(args.cv_seed),
                knn_k=int(args.knn_eval_k),
                extra_cols={"ratio_grid": str(args.ratios), "wrapper_time": score_time},
            )

            for S in fsets:
                t2 = time.perf_counter()
                acc = _knn_acc_test(Xtr, ytr, Xte, yte, S, knn_k=int(args.knn_eval_k))
                eval_time = float(time.perf_counter() - t2)
                test_rows.append(_test_point_row(
                    dataset_idx=int(args.dataset_idx),
                    dataset_name=str(datasetName),
                    method=method,
                    run_id=int(args.run),
                    test_mode="selected",
                    S=S,
                    Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
                    zData_full_for_red=zData_full,
                    time_total=float(score_time + eval_time),
                    knn_k=int(args.knn_eval_k),
                    extra={"post_mode": "", "note": "wrapper"},
                ))
        else:
            _write_text(os.path.join(run_dir, "error.txt"), f"Unknown traditional method: {method}")

    _write_csv(os.path.join(run_dir, "front_train_cv.csv"), front_rows)
    _write_csv(os.path.join(run_dir, "test_points.csv"), test_rows)
    _write_json(os.path.join(run_dir, "run_meta.json"), {
        "algo": args.algo,
        "method": method,
        "dataset_idx": int(args.dataset_idx),
        "dataset": str(datasetName),
        "run": int(args.run),
        "seed_run": int(seed_run),
        "ratios": [float(x) for x in str(args.ratios).split(",") if str(x).strip()],
        "test_size": float(args.test_size),
        "split_seed": int(args.split_seed),
        "knn_eval_k": int(args.knn_eval_k),
        "cv_folds": int(args.cv_folds),
        "cv_seed": int(args.cv_seed),
    })


def run_graph_job(args, run_dir: str, seed_run: int) -> None:
    # algo = graph:InfFS or graph:UGFS
    _, gname = args.algo.split(":", 1)
    method = gname

    _enter_runtime(args.baseline_root, secondary_root=args.oop_root)
    myinput = _import("myinputdatasetXD").myinputdatasetXD
    gfs = _import("graph_fs")

    dataset, labels, datasetName = myinput(int(args.dataset_idx))
    X_raw = np.asarray(dataset[:, 1:], dtype=float)
    y = np.asarray(labels, dtype=int).ravel()

    tr_idx, te_idx = _fixed_split_indices(X_raw, y, float(args.test_size), int(args.split_seed))
    Xtr_raw = X_raw[tr_idx, :]
    Xte_raw = X_raw[te_idx, :]
    ytr = y[tr_idx]
    yte = y[te_idx]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr_raw)
    Xte = scaler.transform(Xte_raw)

    zData_full = _mapminmax_zero_one(X_raw)
    n_features = Xtr.shape[1]
    fixed_k = int(getattr(args, "fixed_fnum", 0))
    if fixed_k > 0:
        ratios = []
    else:
        ratios = [float(x) for x in str(args.ratios).split(",") if str(x).strip()]

    t0 = time.perf_counter()
    if method.lower() == "inffs":
        scores, ranking, _ = gfs.inffs_rank(Xtr, ytr, standardize=False, verbose=0)
    elif method.lower() == "ugfs":
        scores, ranking, _ = gfs.ugfs_rank(Xtr, ytr, standardize=False, verbose=0)
    else:
        _write_text(os.path.join(run_dir, "error.txt"), f"Unknown graph method: {method}")
        return
    score_time = float(time.perf_counter() - t0)

    ranking = np.asarray(ranking, dtype=int).ravel()
    fsets = []
    if fixed_k > 0:
        k = max(1, min(fixed_k, n_features))
        fsets.append(ranking[:k].astype(int))
    else:
        for ratio in ratios:
            k = max(1, min(int(round(ratio * n_features)), n_features))
            fsets.append(ranking[:k].astype(int))

    t_front = time.perf_counter()
    front_rows = _front_train_rows(
        dataset_idx=int(args.dataset_idx),
        dataset_name=str(datasetName),
        method=method,
        run_id=int(args.run),
        feature_sets=fsets,
        X_train_for_cv=Xtr,
        y_train=ytr,
        zData_full_for_red=zData_full,
        cv_folds=int(args.cv_folds),
        cv_seed=int(args.cv_seed),
        knn_k=int(args.knn_eval_k),
        extra_cols={"ratio_grid": str(args.ratios), "score_time": score_time},
    )
    front_eval_time = float(time.perf_counter() - t_front)
    for rr in front_rows:
        rr["front_eval_time"] = front_eval_time

    test_rows = []
    if fixed_k > 0:
        k = max(1, min(fixed_k, n_features))
        S = ranking[:k].astype(int)
        acc = _knn_acc_test(Xtr, ytr, Xte, yte, S, knn_k=int(args.knn_eval_k))
        test_rows.append({
            "dataset_idx": int(args.dataset_idx),
            "dataset": str(datasetName),
            "method": method,
            "run": int(args.run),
            "test_mode": "fixed_k",
            "fixed_fnum": int(k),
            "acc_test": float(acc),
            "eRate_test": float(1.0 - acc),
            "fnum": int(S.size),
            "fRatio": float(S.size / n_features),
            "red": float(_redundancy_rate_subset_fallback(S, zData_full)),
            "time_total": float(score_time),
            "fset": " ".join(map(str, S.tolist())),
        })
    else:
        for ratio in ratios:
            k = max(1, min(int(round(ratio * n_features)), n_features))
            S = ranking[:k].astype(int)
            t1 = time.perf_counter()
            acc = _knn_acc_test(Xtr, ytr, Xte, yte, S, knn_k=int(args.knn_eval_k))
            eval_time = float(time.perf_counter() - t1)

            test_rows.append({
                "dataset_idx": int(args.dataset_idx),
                "dataset": str(datasetName),
                "method": method,
                "run": int(args.run),
                "test_mode": "ratio",
                "ratio": float(ratio),
                "acc_test": float(acc),
                "eRate_test": float(1.0 - acc),
                "fnum": int(S.size),
                "fRatio": float(S.size / n_features),
                "red": float(_redundancy_rate_subset_fallback(S, zData_full)),
                "time_total": float(score_time + eval_time),
                "score_time": float(score_time),
                "eval_time": float(eval_time),
                "fset": " ".join(map(str, S.tolist())),
            })

    _write_csv(os.path.join(run_dir, "front_train_cv.csv"), front_rows)
    _write_csv(os.path.join(run_dir, "test_points.csv"), test_rows)
    _write_json(os.path.join(run_dir, "run_meta.json"), {
        "algo": args.algo,
        "method": method,
        "dataset_idx": int(args.dataset_idx),
        "dataset": str(datasetName),
        "run": int(args.run),
        "seed_run": int(seed_run),
        "ratios": ratios,
        "test_size": float(args.test_size),
        "split_seed": int(args.split_seed),
        "knn_eval_k": int(args.knn_eval_k),
        "cv_folds": int(args.cv_folds),
        "cv_seed": int(args.cv_seed),
    })


def run_allfeatures_job(args, run_dir: str, seed_run: int) -> None:
    method = "KNN_ALL"

    _enter_runtime(args.baseline_root, secondary_root=args.oop_root)
    myinput = _import("myinputdatasetXD").myinputdatasetXD

    dataset, labels, datasetName = myinput(int(args.dataset_idx))
    X_raw = np.asarray(dataset[:, 1:], dtype=float)
    y = np.asarray(labels, dtype=int).ravel()

    tr_idx, te_idx = _fixed_split_indices(X_raw, y, float(args.test_size), int(args.split_seed))

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_raw[tr_idx, :])
    Xte = scaler.transform(X_raw[te_idx, :])
    ytr = y[tr_idx]
    yte = y[te_idx]

    zData_full = _mapminmax_zero_one(X_raw)

    n_features = Xtr.shape[1]
    S_all = np.arange(n_features, dtype=int)

    DO_CV_MAX_FEATS = 2000
    DO_CV_MAX_SAMPLES = 8000
    do_cv = (Xtr.shape[0] <= DO_CV_MAX_SAMPLES) and (n_features <= DO_CV_MAX_FEATS)

    front_rows: List[Dict[str, Any]] = []
    if do_cv:
        front_rows = _front_train_rows(
            dataset_idx=int(args.dataset_idx),
            dataset_name=str(datasetName),
            method=method,
            run_id=int(args.run),
            feature_sets=[S_all],
            X_train_for_cv=Xtr,
            y_train=ytr,
            zData_full_for_red=zData_full,
            cv_folds=int(args.cv_folds),
            cv_seed=int(args.cv_seed),
            knn_k=int(args.knn_eval_k),
            extra_cols={"note": "ALL features", "scaler": "StandardScaler"},
        )
    else:
        fnum = int(n_features)
        redv = _redundancy_rate_subset_fallback(S_all, zData_full)
        front_rows = [{
            "dataset_idx": int(args.dataset_idx),
            "dataset": str(datasetName),
            "method": method,
            "run": int(args.run),
            "point_id": 0,
            "fnum": fnum,
            "fRatio": 1.0,
            "acc_train_cv": float("nan"),
            "eRate_train_cv": float("nan"),
            "red": float(redv),
            "fset": " ".join(map(str, S_all.tolist())),
            "note": "ALL features (train CV skipped: too heavy)",
            "scaler": "StandardScaler",
        }]

    t1 = time.perf_counter()
    acc = _knn_acc_test(Xtr, ytr, Xte, yte, S_all, knn_k=int(args.knn_eval_k))
    eval_time = float(time.perf_counter() - t1)

    test_rows = [{
        "dataset_idx": int(args.dataset_idx),
        "dataset": str(datasetName),
        "method": method,
        "run": int(args.run),
        "test_mode": "all",
        "acc_test": float(acc),
        "eRate_test": float(1.0 - acc),
        "fnum": int(n_features),
        "fRatio": 1.0,
        "red": float(_redundancy_rate_subset_fallback(S_all, zData_full)),
        "time_total": float(eval_time),
        "eval_time": float(eval_time),
        "fset": " ".join(map(str, S_all.tolist())),
        "note": "ALL features",
        "scaler": "StandardScaler",
    }]

    _write_csv(os.path.join(run_dir, "front_train_cv.csv"), front_rows)
    _write_csv(os.path.join(run_dir, "test_points.csv"), test_rows)
    _write_json(os.path.join(run_dir, "run_meta.json"), {
        "algo": args.algo,
        "method": method,
        "dataset_idx": int(args.dataset_idx),
        "dataset": str(datasetName),
        "run": int(args.run),
        "seed_run": int(seed_run),
        "test_size": float(args.test_size),
        "split_seed": int(args.split_seed),
        "knn_eval_k": int(args.knn_eval_k),
        "cv_folds": int(args.cv_folds),
        "cv_seed": int(args.cv_seed),
        "train_cv_computed": bool(do_cv),
        "scaler": "StandardScaler",
    })


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_idx", type=int, required=True)
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--algo", type=str, required=True)

    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--baseline_root", type=str, required=True)
    ap.add_argument("--oop_root", type=str, required=True)

    ap.add_argument("--test_size", type=float, default=0.30)
    ap.add_argument("--split_seed", type=int, default=42)

    ap.add_argument("--knn_eval_k", type=int, default=5)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--cv_seed", type=int, default=42)

    # GBFS baseline/enhanced params
    ap.add_argument("--base_delt", type=float, default=10.0)
    ap.add_argument("--base_omega", type=float, default=0.8)
    ap.add_argument("--base_kneigh", type=int, default=5)
    ap.add_argument("--base_pop", type=int, default=20)
    ap.add_argument("--base_gen", type=int, default=50)

    # enhanced kshell params
    ap.add_argument("--ks_max_add", type=int, default=5)
    ap.add_argument("--ks_rc_tau", type=float, default=0.3)

    ap.add_argument("--prob_beta", type=float, default=2.0, help="ProbInit beta (default 2.0).")
    ap.add_argument("--buddy_lambda", type=float, default=0.5, help="Buddy-SFS redundancy weight lambda (default 0.5).")

    ap.add_argument("--fixed_fnum", type=str, default="0",
                    help="Integer K, or True/auto to load target_fnum from ref_sizes.csv by (dataset_idx, run).")
    ap.add_argument("--ref_sizes_csv", type=str, default=_default_ref_sizes_path(),
                    help="CSV containing dataset_idx, run, target_fnum. Used when --fixed_fnum is True/auto.")
    ap.add_argument("--ratios", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")

    return ap.parse_args()

def main() -> None:
    args = parse_args()

    dataset_idx = int(args.dataset_idx)
    run_id = int(args.run)

    raw_ff = str(getattr(args, "fixed_fnum", "0")).strip().lower()
    auto_mode = raw_ff in ("true", "t", "yes", "y", "auto", "1")

    if auto_mode:
        if args.algo.startswith("trad:") or args.algo.startswith("graph:"):
            ref_path = str(getattr(args, "ref_sizes_csv", _default_ref_sizes_path())).strip()
            if not ref_path or not os.path.exists(ref_path):
                _write_text(os.path.join(str(args.out_root), "error_ref_sizes.txt"),
                            f"ref_sizes_csv not found: {ref_path}\n")
                raise SystemExit(2)

            k = _auto_fixed_fnum_from_ref(dataset_idx=dataset_idx, run_id=run_id, ref_csv_path=ref_path)
            if k is None:
                algo_tag_tmp = _algo_tag(args.algo)
                run_dir_tmp = os.path.join(str(args.out_root), f"dataset_{dataset_idx:02d}", algo_tag_tmp, f"run_{run_id:02d}")
                _ensure_dir(run_dir_tmp)
                _write_text(os.path.join(run_dir_tmp, "error.txt"),
                            f"[AUTO fixed_fnum] Missing target_fnum for dataset_idx={dataset_idx}, run={run_id}\n")
                print("SKIP (no target_fnum):", dataset_idx, run_id, args.algo)
                return
            args.fixed_fnum = int(k)
        else:
            args.fixed_fnum = 0
    else:
        try:
            args.fixed_fnum = int(float(str(getattr(args, "fixed_fnum", "0")).strip()))
        except Exception:
            args.fixed_fnum = 0

    # deterministic seed per (dataset, run) independent of algo
    seed_run = 1000 * dataset_idx + run_id
    random.seed(seed_run)
    np.random.seed(seed_run)

    algo_tag = _algo_tag(args.algo)
    if args.algo.startswith("enh:"):
        algo_tag += (
            f"__beta_{_fmt_float_tag(float(args.prob_beta))}"
            f"__tau_{_fmt_float_tag(float(args.ks_rc_tau))}"
            f"__lam_{_fmt_float_tag(float(args.buddy_lambda))}"
        )

    run_dir = os.path.join(str(args.out_root), f"dataset_{dataset_idx:02d}", algo_tag, f"run_{run_id:02d}")
    _ensure_dir(run_dir)

    _write_json(os.path.join(run_dir, "run_info.json"), {
        "dataset_idx": dataset_idx,
        "run": run_id,
        "algo": args.algo,
        "seed_run": seed_run,
        "test_size": float(args.test_size),
        "split_seed": int(args.split_seed),
        "cv_folds": int(args.cv_folds),
        "cv_seed": int(args.cv_seed),
        "knn_eval_k": int(args.knn_eval_k),
        "fixed_fnum": int(args.fixed_fnum),
        "prob_beta": float(args.prob_beta),
        "ks_rc_tau": float(args.ks_rc_tau),
        "buddy_lambda": float(args.buddy_lambda),
    })

    if args.algo == "all":
        run_allfeatures_job(args, run_dir, seed_run)
    elif args.algo == "baseline":
        run_baseline_job(args, run_dir, seed_run)
    elif args.algo.startswith("enh:"):
        run_enhanced_job(args, run_dir, seed_run)
    elif args.algo.startswith("trad:"):
        run_traditional_job(args, run_dir, seed_run)
    elif args.algo.startswith("graph:"):
        run_graph_job(args, run_dir, seed_run)
    else:
        _write_text(os.path.join(run_dir, "error.txt"), f"Unknown algo format: {args.algo}")
        raise SystemExit(2)

    print("DONE:", run_dir)


if __name__ == "__main__":
    main()

# $REPO = "D:\PhD\The First Paper\Code Implement\GBFS-SND"
# $BASELINE = "$REPO\Python Code"
# $OOP = "$REPO\OOP Code"
# $OUT = "$REPO\_local_out"

# python "$REPO\Compare\run_one_algo.py" `
#   --dataset_idx 1 --run 1 --algo baseline `
#   --out_root "$OUT" `
#   --baseline_root "$BASELINE" --oop_root "$OOP" `
#   --base_pop 20 --base_gen 50 `
#   --test_size 0.3 --split_seed 42 `
#   --cv_folds 3 --cv_seed 42 `
#   --knn_eval_k 5

# python "$REPO\Compare\run_one_algo.py" `
#   --dataset_idx 1 --run 1 --algo "enh:knn:normal" `
#   --out_root "$OUT" `
#   --baseline_root "$BASELINE" --oop_root "$OOP" `
#   --base_pop 20 --base_gen 50 `
#   --ks_max_add 5 --ks_rc_tau 0.3 `
#   --prob_beta 2.0 --buddy_lambda 0.5 `
#   --test_size 0.3 --split_seed 42 `
#   --cv_folds 3 --cv_seed 42 `
#   --knn_eval_k 5

# python "$REPO\Compare\run_one_algo.py" --dataset_idx 2 --run 1 --algo "trad:FILTER_pearson" --out_root "$OUT" --baseline_root "$BASELINE" --oop_root "$OOP" --cv_folds 3 --knn_eval_k 5 --fixed_fnum True
