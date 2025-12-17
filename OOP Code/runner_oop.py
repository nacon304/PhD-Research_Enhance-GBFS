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
from helper import _mapminmax_zero_one, redundancy_rate_subset

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
        )

    def run_dataset(self, data_idx: int):
        RUNS = self.cfg.runs
        init_modes = list(self.cfg.init_modes)

        out_dir = os.path.join(self.visual_root, f"{data_idx:02d}")
        os.makedirs(out_dir, exist_ok=True)

        metrics = {m: {"ACC": np.zeros(RUNS), "FNUM": np.zeros(RUNS), "RED": np.zeros(RUNS), "TIME": np.zeros(RUNS)}
                   for m in init_modes}

        FSET = [[] for _ in range(RUNS + 2)]
        assiNum = np.zeros(RUNS, float)

        for r in range(1, RUNS + 1):
            tic = perf_counter()

            ctx = self._prepare_context_one_run(data_idx, r)
            prep_time = perf_counter() - tic

            for mode in init_modes:
                mode_dir = os.path.join(out_dir, f"run_{r:02d}", mode)  # ✅ tránh overwrite
                shutil.rmtree(mode_dir, ignore_errors=True)
                os.makedirs(mode_dir, exist_ok=True)

                t0 = perf_counter()

                initializer = get_initializer(mode)
                init_res = initializer.build(ctx.Zout, ctx.fisher_raw, self.cfg)
                ctx.A_init = init_res.A_init
                ctx.neigh_list = init_res.neighbors

                # sync to GG and call legacy solver
                sync_context_to_GG(ctx)

                kNeiAdj = squareform(ctx.A_init, force="tovector", checks=False)
                featIdx, _, _, _ = newtry_ms(kNeiAdj, self.cfg.pop, self.cfg.gen, mode_dir)
                featIdx = np.asarray(featIdx)
                selected = np.where(featIdx != 0)[0]
                fnum = selected.size

                # eval
                if fnum == 0:
                    acc, red = 0.0, 0.0
                else:
                    knn = KNeighborsClassifier(n_neighbors=self.cfg.knn_eval_k)
                    knn.fit(ctx.trData[:, selected], ctx.trLabel)
                    pred = knn.predict(ctx.teData[:, selected])
                    acc = float(np.mean(pred == ctx.teLabel))
                    red = float(redundancy_rate_subset(selected, ctx.zData))

                elapsed = perf_counter() - t0
                metrics[mode]["ACC"][r - 1] = acc
                metrics[mode]["FNUM"][r - 1] = fnum
                metrics[mode]["RED"][r - 1] = red
                metrics[mode]["TIME"][r - 1] = prep_time + elapsed

                if mode == self.cfg.log_mode:
                    FSET[r - 1] = selected
                    # assiNum: bạn đang lấy từ GG.assiNumInside; để y hệt thì bạn sync thêm ở adapter

        # summary df giống bạn đang làm
        rows = []
        for i in range(RUNS):
            row = {"run": i + 1}
            for mode in init_modes:
                row[f"acc_{mode}"] = metrics[mode]["ACC"][i]
                row[f"fnum_{mode}"] = metrics[mode]["FNUM"][i]
                row[f"red_{mode}"] = metrics[mode]["RED"][i]
                row[f"time_{mode}"] = metrics[mode]["TIME"][i]
            rows.append(row)

        row_mean = {"run": "mean"}
        row_std = {"run": "std"}
        for mode in init_modes:
            row_mean[f"acc_{mode}"] = float(metrics[mode]["ACC"].mean())
            row_mean[f"fnum_{mode}"] = float(metrics[mode]["FNUM"].mean())
            row_mean[f"red_{mode}"] = float(metrics[mode]["RED"].mean())
            row_mean[f"time_{mode}"] = float(metrics[mode]["TIME"].mean())

            row_std[f"acc_{mode}"] = float(metrics[mode]["ACC"].std())
            row_std[f"fnum_{mode}"] = float(metrics[mode]["FNUM"].std())
            row_std[f"red_{mode}"] = float(metrics[mode]["RED"].std())
            row_std[f"time_{mode}"] = float(metrics[mode]["TIME"].std())

        rows += [row_mean, row_std]
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, "init_summary.csv"), index=False)

        # build product theo format cũ: dùng log_mode
        base = metrics[self.cfg.log_mode]
        ACC = np.concatenate([base["ACC"], [base["ACC"].mean(), base["ACC"].std()]])
        FNUM = np.concatenate([base["FNUM"], [base["FNUM"].mean(), base["FNUM"].std()]])
        RED = np.concatenate([base["RED"], [base["RED"].mean(), base["RED"].std()]])
        T = np.concatenate([base["TIME"], [base["TIME"].mean(), base["TIME"].std()]])

        product = []
        for i in range(RUNS + 2):
            product.append([ACC[i], FNUM[i], FSET[i], RED[i], T[i]])

        return product, ctx.dataset_name
