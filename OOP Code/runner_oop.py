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
    plot_knn_graph_with_selected,
    _safe_mkdir,
    _save_neighbors_json,
    _export_legacy_solver_logs,
    _save_selected_txt,
    _dump_post_metrics_json,
)
from sequential_strategies import apply_post_sequential

from newtry_ms import newtry_ms
from oop_core import ExperimentConfig, get_initializer
from gg_adapter import sync_context_to_GG
import gbfs_globals as GG

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

    A_init: np.ndarray = None
    neigh_list: list = None

    acc_single: np.ndarray = None
    C_matrix: np.ndarray = None

class GBFSRunner:
    def __init__(self, cfg: ExperimentConfig, visual_root: str):
        self.cfg = cfg
        self.visual_root = visual_root

    def _prepare_context_one_run(self, data_idx: int, run_id: int,
                                 init_modes_lc, kshell_modes_lc, post_modes_lc) -> GBFSContext:
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
                n_neighbors=self.cfg.buddy_params.knn_k,
                cv=self.cfg.buddy_params.cv,
                use_cv=True,
                random_state=self.cfg.buddy_params.seed
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

        combo_log_counter = 0
        if not hasattr(GG, "run_logs") or GG.run_logs is None:
            GG.run_logs = {}
        
        def should_log_heavy(init_mode_lc: str, ks_mode_lc: str) -> bool:
            LP = self.cfg.log_params
            if not LP.enabled:
                return False
            if not LP.log_only_selected_combo:
                return True
            return (init_mode_lc == str(self.cfg.log_init_mode).lower()
                    and ks_mode_lc == str(self.cfg.log_kshell_seq_mode).lower())

        def should_log_post(init_mode_lc: str, ks_mode_lc: str, ps_mode_lc: str) -> bool:
            LP = self.cfg.log_params
            if not LP.enabled:
                return False
            if not LP.log_only_selected_combo:
                return True
            return (init_mode_lc == str(self.cfg.log_init_mode).lower()
                    and ks_mode_lc == str(self.cfg.log_kshell_seq_mode).lower()
                    and ps_mode_lc == str(self.cfg.log_post_seq_mode).lower())
        
        # =========================
        # RUN LOOP
        # =========================
        for r in range(1, RUNS + 1):
            tic = perf_counter()
            ctx = self._prepare_context_one_run(
                data_idx=data_idx,
                run_id=r,
                init_modes_lc=init_modes,
                kshell_modes_lc=kshell_seq_modes,
                post_modes_lc=post_seq_modes
            )
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
                        n_neighbors=self.cfg.buddy_params.knn_k,
                        cv=self.cfg.buddy_params.cv,
                        use_cv=True,
                        candidate_pairs=ctx.neigh_list,
                        random_state=self.cfg.buddy_params.seed
                    )
                    C_cache[init_mode] = ctx.C_matrix
                else:
                    ctx.C_matrix = None

                for ks_mode in kshell_seq_modes:
                    mode_dir = os.path.join(out_dir, f"run_{r:02d}", init_mode, f"ks_{ks_mode}")
                    shutil.rmtree(mode_dir, ignore_errors=True)
                    _safe_mkdir(mode_dir)

                    heavy_ok = should_log_heavy(init_mode, ks_mode)
                    LP = self.cfg.log_params

                    # ---- init GG logs like old code (UNIQUE per (run, init, ks)) ----
                    combo_log_counter += 1
                    log_id = combo_log_counter
                    GG.current_run = log_id
                    GG.run_logs[log_id] = {"pareto_fronts": [], "pop_metrics": []}
                    GG.assiNumInside = []

                    t0 = perf_counter()

                    # sync + set GG.kshell_seq_mode + rc_tau/max_add
                    sync_context_to_GG(ctx, cfg=self.cfg, kshell_seq_mode=ks_mode)

                    # ---- Save init graph info ----
                    if LP.enabled and heavy_ok and LP.export_init_graph:
                        np.save(os.path.join(mode_dir, "A_init.npy"), np.asarray(ctx.A_init, dtype=float))
                        _save_neighbors_json(ctx.neigh_list, os.path.join(mode_dir, "neighbors.json"))

                    kNeiAdj = squareform(ctx.A_init, force="tovector", checks=False)
                    featIdx, pareto_masks, pareto_objs, kNeigh_chosen = newtry_ms(
                        kNeiAdj, self.cfg.pop, self.cfg.gen, mode_dir
                    )
                    try:
                        if pareto_masks is not None:
                            np.save(os.path.join(mode_dir, "pareto_masks.npy"), np.asarray(pareto_masks))
                        if pareto_objs is not None:
                            np.save(os.path.join(mode_dir, "pareto_objs.npy"), np.asarray(pareto_objs))
                    except Exception:
                        pass

                    t_after_solver = perf_counter()
                    solver_time = prep_time + (t_after_solver - t0)

                    featIdx = np.asarray(featIdx)
                    S0 = np.where(featIdx != 0)[0].astype(int)

                    # ---- Export legacy solver logs to files (like old code) ----
                    logs = GG.run_logs.get(log_id, {})
                    if LP.enabled and heavy_ok and LP.export_solver_logs:
                        _export_legacy_solver_logs(mode_dir, logs)

                    # ---- save solver outputs ----
                    if LP.enabled and heavy_ok and LP.export_solver_meta:
                        _save_selected_txt(S0, os.path.join(mode_dir, "S0_selected.txt"))
                        np.save(os.path.join(mode_dir, "kNeigh_chosen.npy"), np.asarray(kNeigh_chosen, dtype=int))

                        assi_num = float(np.sum(np.asarray(GG.assiNumInside, dtype=float))) if len(GG.assiNumInside) else 0.0
                        _dump_post_metrics_json(os.path.join(mode_dir, "solver_meta.json"), {
                            "run": int(r),
                            "init_mode": init_mode,
                            "kshell_seq_mode": ks_mode,
                            "log_id": int(log_id),
                            "assi_num": float(assi_num),
                            "solver_time": float(solver_time),
                            "fnum_S0": int(S0.size),
                        })
                    
                    if LP.enabled and heavy_ok and LP.export_plots:
                        try:
                            plot_knn_graph_with_selected(
                                ctx.neigh_list,
                                kNeigh_chosen,
                                S0,
                                mode_dir,
                                run_id=r,
                                filename_prefix="knn_graph_S0"
                            )
                        except Exception as e:
                            with open(os.path.join(mode_dir, "plot_error_S0.txt"), "w", encoding="utf-8") as f:
                                f.write(str(e))

                    # ---- cache eval S0 ----
                    if S0.size == 0:
                        acc_S0, red_S0 = 0.0, 0.0
                    else:
                        knn0 = KNeighborsClassifier(n_neighbors=self.cfg.knn_eval_k)
                        knn0.fit(ctx.trData[:, S0], ctx.trLabel)
                        pred0 = knn0.predict(ctx.teData[:, S0])
                        acc_S0 = float(np.mean(pred0 == ctx.teLabel))
                        red_S0 = float(redundancy_rate_subset(S0, ctx.zData))

                    post_selected = {}

                    for ps_mode in post_seq_modes:
                        print(f"Run {r}, Init {init_mode}, K-shell Seq {ks_mode}, Post Seq {ps_mode}")
                        ps = str(ps_mode).lower()
                        key = f"{init_mode}__ks{ks_mode}__post{ps_mode}"

                        post_dir = os.path.join(mode_dir, f"post_{ps}")
                        _safe_mkdir(post_dir)

                        t_post = perf_counter()

                        buddy_kwargs = None
                        if ps == "buddy":
                            buddy_kwargs = dict(
                                max_buddy_per_core=self.cfg.buddy_params.max_per_core,
                                n_neighbors=self.cfg.buddy_params.knn_k,
                                cv=self.cfg.buddy_params.cv,
                                lam_red=self.cfg.buddy_params.lam_red,
                                random_state=self.cfg.buddy_params.seed
                            )

                        S_post = apply_post_sequential(
                            S0=S0,
                            mode=ps,
                            X=ctx.trData,
                            y=ctx.trLabel,
                            C_matrix=C_cache.get(init_mode, ctx.C_matrix),
                            R_matrix=ctx.Zout,
                            buddy_kwargs=buddy_kwargs
                        )
                        S_post = np.asarray(S_post, dtype=int)

                        # ---- Evaluate ----
                        if S_post.size == 0:
                            acc_post, red_post = 0.0, 0.0
                        else:
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

                        # ---- Store metrics (ablation) ----
                        metrics[key]["ACC"][r - 1] = acc_post
                        metrics[key]["FNUM"][r - 1] = int(S_post.size)
                        metrics[key]["RED"][r - 1] = red_post
                        metrics[key]["TIME"][r - 1] = total_time

                        post_selected[ps] = S_post

                        # ---- Export per-post logs ----
                        post_ok = should_log_post(init_mode, ks_mode, ps)
                        if LP.enabled and post_ok and LP.export_post_logs:
                            _save_selected_txt(S_post, os.path.join(post_dir, "selected_features.txt"))
                            _dump_post_metrics_json(os.path.join(post_dir, "post_metrics.json"), {
                                "run": int(r),
                                "init_mode": init_mode,
                                "kshell_seq_mode": ks_mode,
                                "post_mode": ps,
                                "acc_test": float(acc_post),
                                "red": float(red_post),
                                "fnum": int(S_post.size),
                                "time_total": float(total_time),
                                "time_solver": float(solver_time),
                                "time_post_extra": float(post_extra_time),
                            })

                        if LP.enabled and post_ok and LP.export_plots:
                            try:
                                plot_knn_graph_with_selected(
                                    ctx.neigh_list,
                                    kNeigh_chosen,
                                    S_post,
                                    post_dir,
                                    run_id=r,
                                    filename_prefix=f"knn_graph_{ps}"
                                )
                            except Exception as e:
                                with open(os.path.join(post_dir, f"plot_error_{ps}.txt"), "w", encoding="utf-8") as f:
                                    f.write(str(e))

                    # ---- store FSET for product (chosen combo) ----
                    if (init_mode == str(self.cfg.log_init_mode).lower()) and (ks_mode == str(self.cfg.log_kshell_seq_mode).lower()):
                        chosen = post_selected.get(str(self.cfg.log_post_seq_mode).lower(), S0)
                        FSET[r - 1] = chosen
                    
                    if LP.enabled and heavy_ok and (not getattr(LP, "keep_run_logs_in_memory", False)):
                        GG.run_logs.pop(log_id, None)
        
        rows = []
        for i in range(RUNS):
            row = {"run": i + 1}
            for k in keys:
                row[f"acc_{k}"] = float(metrics[k]["ACC"][i])
                row[f"fnum_{k}"] = float(metrics[k]["FNUM"][i])
                row[f"red_{k}"] = float(metrics[k]["RED"][i])
                row[f"time_{k}"] = float(metrics[k]["TIME"][i])
            rows.append(row)

        row_mean = {"run": "mean"}
        row_std  = {"run": "std"}
        for k in keys:
            row_mean[f"acc_{k}"]  = float(metrics[k]["ACC"].mean())
            row_mean[f"fnum_{k}"] = float(metrics[k]["FNUM"].mean())
            row_mean[f"red_{k}"]  = float(metrics[k]["RED"].mean())
            row_mean[f"time_{k}"] = float(metrics[k]["TIME"].mean())

            row_std[f"acc_{k}"]  = float(metrics[k]["ACC"].std())
            row_std[f"fnum_{k}"] = float(metrics[k]["FNUM"].std())
            row_std[f"red_{k}"]  = float(metrics[k]["RED"].std())
            row_std[f"time_{k}"] = float(metrics[k]["TIME"].std())

        rows += [row_mean, row_std]
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, "ablation_summary.csv"), index=False)

        log_key = f"{str(self.cfg.log_init_mode).lower()}__ks{str(self.cfg.log_kshell_seq_mode).lower()}__post{str(self.cfg.log_post_seq_mode).lower()}"
        if log_key not in metrics:
            log_key = f"{str(self.cfg.log_init_mode).lower()}__ks{str(self.cfg.log_kshell_seq_mode).lower()}__postnormal"

        ACC = np.concatenate([metrics[log_key]["ACC"], [metrics[log_key]["ACC"].mean(), metrics[log_key]["ACC"].std()]])
        FNUM = np.concatenate([metrics[log_key]["FNUM"], [metrics[log_key]["FNUM"].mean(), metrics[log_key]["FNUM"].std()]])
        RED = np.concatenate([metrics[log_key]["RED"], [metrics[log_key]["RED"].mean(), metrics[log_key]["RED"].std()]])
        T   = np.concatenate([metrics[log_key]["TIME"], [metrics[log_key]["TIME"].mean(), metrics[log_key]["TIME"].std()]])

        product = []
        for i in range(RUNS + 2):
            product.append([float(ACC[i]), float(FNUM[i]), FSET[i], float(RED[i]), float(T[i])])

        return product, ctx.dataset_name