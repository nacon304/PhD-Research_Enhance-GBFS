# runner_oop.py
import os, shutil
import numpy as np
import pandas as pd
from time import perf_counter
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

from myinputdatasetXD import myinputdatasetXD
from fisherScore import fisherScore
from helper import (
    _mapminmax_zero_one,
    redundancy_rate_subset,
    compute_single_feature_accuracy,
    compute_complementarity_matrix,
)
from sequential_strategies import apply_post_sequential

from newtry_ms import newtry_ms
from oop_core import ExperimentConfig, get_initializer
from gg_adapter import sync_context_to_GG

@dataclass
class GBFSContext:
    dataset_name: str
    zData: np.ndarray
    label: np.ndarray

    trIdx: np.ndarray
    trData: np.ndarray
    trLabel: np.ndarray
    teData: np.ndarray
    teLabel: np.ndarray

    fisher_raw: np.ndarray
    Zout: np.ndarray
    Weight: np.ndarray
    vWeight: np.ndarray
    vWeight1: np.ndarray

    # will be set per init mode
    A_init: np.ndarray = None
    neigh_list: list = None

    acc_single: np.ndarray = None
    C_matrix: np.ndarray = None

class GBFSRunner:
    def __init__(self, cfg: ExperimentConfig, visual_root: str):
        self.cfg = cfg
        self.visual_root = visual_root

    def _prepare_context_one_run(self, data_idx: int, run_id: int) -> GBFSContext:
        condata, label, datasetName = myinputdatasetXD(data_idx)
        conData = condata[:, 1:]
        zData = _mapminmax_zero_one(conData)

        # split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.cfg.test_size, random_state=self.cfg.split_seed)
        row = zData.shape[0]
        for train_index, test_index in sss.split(zData, label):
            trIdx = np.zeros(row, dtype=bool)
            trIdx[train_index] = True
            trData, trLabel = zData[trIdx], label[trIdx]
            teData, teLabel = zData[~trIdx], label[~trIdx]

        # fisher (raw)
        _, vWeight0 = fisherScore(trData, trLabel)
        vWeight = 1.0 + _mapminmax_zero_one(vWeight0.reshape(-1, 1)).ravel()
        vWeight1 = vWeight

        # similarity/correlation
        adj = 1.0 - pdist(trData.T, metric="correlation")
        adj = np.nan_to_num(adj, nan=0.0)
        adj = np.abs(adj)
        Weight = squareform(adj)
        Zout = squareform(adj)

        need_acc_single = ("buddy" in self.cfg.post_seq_modes) or ("rc_greedy" in self.cfg.kshell_seq_modes)
        acc_single = None
        if need_acc_single:
            acc_single = compute_single_feature_accuracy(
                trData, trLabel,
                n_neighbors=self.cfg.buddy_knn_k,
                cv=self.cfg.buddy_cv,
                use_cv=True,
                random_state=self.cfg.buddy_seed
            )
            
        return GBFSContext(
            dataset_name=datasetName,
            zData=zData,
            label=label,
            trIdx=trIdx,
            trData=trData,
            trLabel=trLabel,
            teData=teData,
            teLabel=teLabel,
            fisher_raw=vWeight0,
            Zout=Zout,
            Weight=Weight,
            vWeight=vWeight,
            vWeight1=vWeight1,
            acc_single=acc_single,
        )

    def run_dataset(self, data_idx: int):
        RUNS = self.cfg.runs
        init_modes = list(self.cfg.init_modes)
        kshell_seq_modes = list(self.cfg.kshell_seq_modes)
        post_seq_modes = list(self.cfg.post_seq_modes)

        combos = [(im, ks, ps) for im in init_modes for ks in kshell_seq_modes for ps in post_seq_modes]
        keys = [f"{im}__ks{ks}__post{ps}" for (im, ks, ps) in combos]

        out_dir = os.path.join(self.visual_root, f"{data_idx:02d}")
        os.makedirs(out_dir, exist_ok=True)

        metrics = {
            k: {"ACC": np.zeros(RUNS), "FNUM": np.zeros(RUNS), "RED": np.zeros(RUNS), "TIME": np.zeros(RUNS)}
            for k in keys
        }

        FSET = [[] for _ in range(RUNS + 2)]

        for r in range(1, RUNS + 1):
            tic = perf_counter()
            ctx = self._prepare_context_one_run(data_idx, r)
            prep_time = perf_counter() - tic

            # cache C per (run, init_mode) vì candidate_pairs=neigh_list phụ thuộc init
            C_cache = {}

            for init_mode in init_modes:
                initializer = get_initializer(init_mode)
                init_res = initializer.build(ctx.Zout, ctx.fisher_raw, self.cfg)
                ctx.A_init = init_res.A_init
                ctx.neigh_list = init_res.neighbors

                # Build C_matrix if needed by either kshell rc_greedy or post buddy
                need_C = ("buddy" in post_seq_modes) or ("rc_greedy" in kshell_seq_modes)
                if need_C:
                    ctx.C_matrix = compute_complementarity_matrix(
                        ctx.trData, ctx.trLabel,
                        acc_single=ctx.acc_single,
                        n_neighbors=self.cfg.buddy_knn_k,
                        cv=self.cfg.buddy_cv,
                        use_cv=True,
                        candidate_pairs=ctx.neigh_list,   # ✅ variable degree
                        random_state=self.cfg.buddy_seed
                    )
                    C_cache[init_mode] = ctx.C_matrix
                else:
                    ctx.C_matrix = None

                for ks_mode in kshell_seq_modes:
                    mode_dir = os.path.join(out_dir, f"run_{r:02d}", init_mode, f"ks_{ks_mode}")
                    shutil.rmtree(mode_dir, ignore_errors=True)
                    os.makedirs(mode_dir, exist_ok=True)

                    t0 = perf_counter()

                    # sync + set GG.kshell_seq_mode + rc_tau/max_add
                    sync_context_to_GG(ctx, cfg=self.cfg, kshell_seq_mode=ks_mode)

                    kNeiAdj = squareform(ctx.A_init, force="tovector", checks=False)
                    featIdx, _, _, _ = newtry_ms(kNeiAdj, self.cfg.pop, self.cfg.gen, mode_dir)

                    t_after_solver = perf_counter()
                    solver_time = prep_time + (t_after_solver - t0)

                    featIdx = np.asarray(featIdx)
                    S0 = np.where(featIdx != 0)[0].astype(int)

                    post_selected = {}

                    # --- base test eval (normal post) ---
                    if S0.size == 0:
                        acc_S0, red_S0 = 0.0, 0.0
                    else:
                        knn0 = KNeighborsClassifier(n_neighbors=self.cfg.knn_eval_k)
                        knn0.fit(ctx.trData[:, S0], ctx.trLabel)
                        pred0 = knn0.predict(ctx.teData[:, S0])
                        acc_S0 = float(np.mean(pred0 == ctx.teLabel))
                        red_S0 = float(redundancy_rate_subset(S0, ctx.zData))

                    for ps_mode in post_seq_modes:
                        print(f"Run {r}/{RUNS}, Init: {init_mode}, SeqKShell: {ks_mode}, Post: {ps_mode}")
                        key = f"{init_mode}__ks{ks_mode}__post{ps_mode}"

                        t_post = perf_counter()

                        ps = str(ps_mode).lower()

                        S_post = apply_post_sequential(
                            S0=S0,
                            mode=ps,
                            X=ctx.trData,
                            y=ctx.trLabel,
                            C_matrix=C_cache.get(init_mode, ctx.C_matrix),
                            R_matrix=ctx.Zout,
                            buddy_kwargs=None
                        )
                        S_post = np.asarray(S_post, dtype=int)

                        # ---- Evaluate test acc + redundancy ----
                        if S_post.size == 0:
                            acc_post, red_post = 0.0, 0.0
                        else:
                            # reuse cached eval for normal (optional)
                            if ps in ["normal", "none"] and np.array_equal(np.sort(S_post), np.sort(S0)):
                                acc_post, red_post = acc_S0, red_S0
                            else:
                                knn = KNeighborsClassifier(n_neighbors=self.cfg.knn_eval_k)
                                knn.fit(ctx.trData[:, S_post], ctx.trLabel)
                                pred = knn.predict(ctx.teData[:, S_post])
                                acc_post = float(np.mean(pred == ctx.teLabel))
                                red_post = float(redundancy_rate_subset(S_post, ctx.zData))

                        post_extra_time = perf_counter() - t_post
                        total_time = solver_time + post_extra_time

                        # ---- Store metrics ----
                        metrics[key]["ACC"][r - 1] = acc_post
                        metrics[key]["FNUM"][r - 1] = int(S_post.size)
                        metrics[key]["RED"][r - 1] = red_post
                        metrics[key]["TIME"][r - 1] = total_time

                        post_selected[ps_mode] = S_post

                    # ---- store FSET for product (chosen combo) ----
                    if (init_mode == self.cfg.log_init_mode) and (ks_mode == self.cfg.log_kshell_seq_mode):
                        chosen = post_selected.get(self.cfg.log_post_seq_mode, S0)
                        FSET[r - 1] = chosen