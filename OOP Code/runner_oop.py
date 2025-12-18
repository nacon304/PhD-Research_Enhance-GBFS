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
from sequential_strategies import buddy_sfs

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

        need_buddy = ("buddy" in self.cfg.seq_modes)
        acc_single = None
        if need_buddy:
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
        seq_modes = list(self.cfg.seq_modes)

        combos = [(im, sm) for im in init_modes for sm in seq_modes]
        combo_keys = [f"{im}__{sm}" for (im, sm) in combos]

        out_dir = os.path.join(self.visual_root, f"{data_idx:02d}")
        os.makedirs(out_dir, exist_ok=True)

        metrics = {
            key: {
                "ACC": np.zeros(RUNS),
                "FNUM": np.zeros(RUNS),
                "RED": np.zeros(RUNS),
                "TIME": np.zeros(RUNS),
            }
            for key in combo_keys
        }

        FSET = [[] for _ in range(RUNS + 2)]

        for r in range(1, RUNS + 1):
            tic = perf_counter()

            ctx = self._prepare_context_one_run(data_idx, r)
            prep_time = perf_counter() - tic

            C_cache = {}
            for init_mode in init_modes:
                print(f"Run {r}, Init mode: {init_mode}, Seq modes: {seq_modes}")
                init_dir = os.path.join(out_dir, f"run_{r:02d}", init_mode)
                shutil.rmtree(init_dir, ignore_errors=True)
                os.makedirs(init_dir, exist_ok=True)

                t_base0 = perf_counter()

                initializer = get_initializer(init_mode)
                init_res = initializer.build(ctx.Zout, ctx.fisher_raw, self.cfg)
                ctx.A_init = init_res.A_init
                ctx.neigh_list = init_res.neighbors

                # sync to GG and call legacy solver
                sync_context_to_GG(ctx)

                kNeiAdj = squareform(ctx.A_init, force="tovector", checks=False)
                featIdx, _, _, _ = newtry_ms(kNeiAdj, self.cfg.pop, self.cfg.gen, init_dir)
                featIdx = np.asarray(featIdx)
                S0 = np.where(featIdx != 0)[0].astype(int)

                # base eval on test
                if S0.size == 0:
                    acc0, red0 = 0.0, 0.0
                else:
                    knn = KNeighborsClassifier(n_neighbors=self.cfg.knn_eval_k)
                    knn.fit(ctx.trData[:, S0], ctx.trLabel)
                    pred = knn.predict(ctx.teData[:, S0])
                    acc0 = float(np.mean(pred == ctx.teLabel))
                    red0 = float(redundancy_rate_subset(S0, ctx.zData))
                time_base = prep_time + (perf_counter() - t_base0)
                
                if "normal" in seq_modes:
                    key = f"{init_mode}__normal"
                    metrics[key]["ACC"][r - 1] = acc0
                    metrics[key]["FNUM"][r - 1] = int(S0.size)
                    metrics[key]["RED"][r - 1] = red0
                    metrics[key]["TIME"][r - 1] = time_base

                # seq_mode == buddy
                if "buddy" in seq_modes:
                    t_b0 = perf_counter()

                    if S0.size == 0:
                        accB, redB = 0.0, 0.0
                        SB = S0
                    else:
                        # build C_matrix for this init_mode if not yet
                        if init_mode not in C_cache:
                            C_cache[init_mode] = compute_complementarity_matrix(
                                ctx.trData, ctx.trLabel,
                                acc_single=ctx.acc_single,
                                n_neighbors=self.cfg.buddy_knn_k,
                                cv=self.cfg.buddy_cv,
                                use_cv=True,
                                candidate_pairs=ctx.neigh_list,
                                random_state=self.cfg.buddy_seed
                            )
                        C = C_cache[init_mode]

                        SB, _ = buddy_sfs(
                            ctx.trData, ctx.trLabel,
                            S0=S0,
                            C_matrix=C,
                            R_matrix=ctx.Zout,
                            max_buddy_per_core=self.cfg.buddy_max_per_core,
                            n_neighbors=self.cfg.buddy_knn_k,
                            cv=self.cfg.buddy_cv,
                            lam_red=self.cfg.buddy_lam_red
                        )
                        SB = np.asarray(SB, dtype=int)

                        if SB.size == 0:
                            accB, redB = 0.0, 0.0
                        else:
                            knn2 = KNeighborsClassifier(n_neighbors=self.cfg.knn_eval_k)
                            knn2.fit(ctx.trData[:, SB], ctx.trLabel)
                            pred2 = knn2.predict(ctx.teData[:, SB])
                            accB = float(np.mean(pred2 == ctx.teLabel))
                            redB = float(redundancy_rate_subset(SB, ctx.zData))

                    time_buddy = time_base + (perf_counter() - t_b0)

                    key = f"{init_mode}__buddy"
                    metrics[key]["ACC"][r - 1] = accB
                    metrics[key]["FNUM"][r - 1] = int(SB.size)
                    metrics[key]["RED"][r - 1] = redB
                    metrics[key]["TIME"][r - 1] = time_buddy

                # store FSET for product (chosen combo)
                if init_mode == self.cfg.log_mode:
                    if self.cfg.log_seq_mode == "normal":
                        chosen_set = S0
                        chosen_acc = acc0
                        chosen_red = red0
                        chosen_time = time_base
                    elif self.cfg.log_seq_mode == "buddy" and ("buddy" in seq_modes):
                        chosen_set = SB
                        chosen_acc = accB
                        chosen_red = redB
                        chosen_time = time_buddy
                    else:
                        chosen_set = S0
                        chosen_acc = acc0
                        chosen_red = red0
                        chosen_time = time_base

                    FSET[r - 1] = chosen_set

        # ===== ablation summary CSV =====
        rows = []
        for i in range(RUNS):
            row = {"run": i + 1}
            for key in combo_keys:
                row[f"acc_{key}"]  = float(metrics[key]["ACC"][i])
                row[f"fnum_{key}"] = float(metrics[key]["FNUM"][i])
                row[f"red_{key}"]  = float(metrics[key]["RED"][i])
                row[f"time_{key}"] = float(metrics[key]["TIME"][i])
            rows.append(row)

        row_mean = {"run": "mean"}
        row_std  = {"run": "std"}
        for key in combo_keys:
            row_mean[f"acc_{key}"]  = float(metrics[key]["ACC"].mean())
            row_mean[f"fnum_{key}"] = float(metrics[key]["FNUM"].mean())
            row_mean[f"red_{key}"]  = float(metrics[key]["RED"].mean())
            row_mean[f"time_{key}"] = float(metrics[key]["TIME"].mean())

            row_std[f"acc_{key}"]  = float(metrics[key]["ACC"].std())
            row_std[f"fnum_{key}"] = float(metrics[key]["FNUM"].std())
            row_std[f"red_{key}"]  = float(metrics[key]["RED"].std())
            row_std[f"time_{key}"] = float(metrics[key]["TIME"].std())

        rows += [row_mean, row_std]
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, "ablation_summary.csv"), index=False)

        # ===== product output (format cũ) from chosen combo =====
        log_key = f"{self.cfg.log_mode}__{self.cfg.log_seq_mode}"
        if log_key not in metrics:
            # fallback: use log_mode__normal
            log_key = f"{self.cfg.log_mode}__normal"

        ACC = np.concatenate([metrics[log_key]["ACC"], [metrics[log_key]["ACC"].mean(), metrics[log_key]["ACC"].std()]])
        FNUM = np.concatenate([metrics[log_key]["FNUM"], [metrics[log_key]["FNUM"].mean(), metrics[log_key]["FNUM"].std()]])
        RED = np.concatenate([metrics[log_key]["RED"], [metrics[log_key]["RED"].mean(), metrics[log_key]["RED"].std()]])
        T   = np.concatenate([metrics[log_key]["TIME"], [metrics[log_key]["TIME"].mean(), metrics[log_key]["TIME"].std()]])

        product = []
        for i in range(RUNS + 2):
            product.append([ACC[i], FNUM[i], FSET[i], RED[i], T[i]])

        return product, ctx.dataset_name
