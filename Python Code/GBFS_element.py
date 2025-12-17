# GBFS_element.py
# ============================================================
# Clean OOP + Pymoo NSGA-II + Parallel Evaluation (thread/process)
# - Decision vars: edge chromosome x in {0,1}^{V_f}, V_f = featNum*kNeigh
# - Decode: build adjacency from GG.kNeiMatrix + GG.kNeiZout
# - Objective: reuse evaluate_objective_f(M, indiv_adj) exactly
# - Returns (compatible with Copy_of_js_ms):
#     featidx_best: full feature mask (0/1) length featNum
#     pareto_masks: feature masks on Pareto front
#     pareto_objs : objectives on Pareto front
#     kNeigh_best : best edge chromosome (length V_f)
# ============================================================

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import gbfs_globals as GG
from evaluate_objective_f import evaluate_objective_f
from kshell_2 import kshell_2

# ---------------- pymoo imports (robust across versions) ----------------
from pymoo.core.problem import ElementwiseProblem

try:
    # pymoo >= 0.6.x
    from pymoo.parallelization.starmap import StarmapParallelization
except Exception:
    # older fallback (may not exist)
    StarmapParallelization = None

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.callback import Callback


# ============================================================
# 1) Helpers: state export/import for process workers (Windows spawn)
# ============================================================

def _export_gg_state():
    """Export exactly what workers need for decode + objective."""
    return {
        "M": int(getattr(GG, "M", 2)),
        "featNum": int(GG.featNum),
        "kNeigh": int(GG.kNeigh),
        "kNeiMatrix": GG.kNeiMatrix,
        "kNeiZout": GG.kNeiZout,
        "data": GG.data,
        "trData": GG.trData,
        "trLabel": GG.trLabel,
    }


def _init_worker_gg(state):
    """Initializer for multiprocessing workers: set GG.* inside worker process."""
    import gbfs_globals as _GG
    _GG.M = state["M"]
    _GG.featNum = state["featNum"]
    _GG.kNeigh = state["kNeigh"]
    _GG.kNeiMatrix = state["kNeiMatrix"]
    _GG.kNeiZout = state["kNeiZout"]
    _GG.data = state["data"]
    _GG.trData = state["trData"]
    _GG.trLabel = state["trLabel"]


def _default_n_jobs():
    c = os.cpu_count() or 1
    return max(1, c - 1)


# ============================================================
# 2) Safe decode (avoid decodeNet bug: neighbor index 0)
# ============================================================

def _decode_adj_from_edgechrom(edge_chrom_01: np.ndarray) -> np.ndarray:
    """
    edge_chrom_01: shape (featNum*kNeigh,) in {0,1}
    Build adjacency matrix (featNum x featNum) using:
      - GG.kNeiMatrix: (featNum, kNeigh) neighbor indices (0-based)
      - GG.kNeiZout:   (featNum, featNum) similarity weights on kNN graph
    """
    featNum = int(GG.featNum)
    kNeigh = int(GG.kNeigh)

    x = np.asarray(edge_chrom_01, dtype=int).ravel()
    if x.size != featNum * kNeigh:
        raise ValueError("edge chromosome length mismatch")

    # reshape to (featNum, kNeigh): 1 means choose that neighbor position
    mask = x.reshape(featNum, kNeigh)

    MODE = np.zeros((featNum, featNum), dtype=float)

    # IMPORTANT: choose by mask positions, not by multiplying indices
    for i in range(featNum):
        chosen_pos = np.where(mask[i, :] != 0)[0]
        if chosen_pos.size > 0:
            neigh_idx = GG.kNeiMatrix[i, chosen_pos].astype(int)
            MODE[i, neigh_idx] = 1.0

    # weighted adjacency
    K = np.asarray(GG.kNeiZout, dtype=float)
    return MODE * K


def _decode_features_only(edge_chrom_01: np.ndarray):
    """Decode edge chromosome -> adjacency -> kshell indices -> full feature mask."""
    adj = _decode_adj_from_edgechrom(edge_chrom_01)
    corr = np.asarray(adj, dtype=float)
    n_features = GG.data.shape[1] if getattr(GG, "data", None) is not None else int(GG.featNum)

    if not np.any(corr):
        mask = np.ones(n_features, dtype=bool)
        return mask, np.arange(n_features, dtype=int)

    feat_idx = np.asarray(kshell_2(corr)).astype(int).ravel()
    feat_idx = feat_idx[(feat_idx >= 0) & (feat_idx < n_features)]

    mask = np.zeros(n_features, dtype=bool)
    mask[feat_idx] = True
    return mask, feat_idx


# ============================================================
# 3) Problem: Elementwise evaluation (pymoo will parallelize this)
# ============================================================

class GBFSEdgeProblem(ElementwiseProblem):
    def __init__(self, elementwise_runner=None):
        if GG.featNum is None or GG.kNeigh is None:
            raise ValueError("GBFSEdgeProblem: GG.featNum / GG.kNeigh not set.")
        if GG.kNeiMatrix is None or GG.kNeiZout is None:
            raise ValueError("GBFSEdgeProblem: GG.kNeiMatrix / GG.kNeiZout not set.")
        if GG.trData is None or GG.trLabel is None:
            raise ValueError("GBFSEdgeProblem: GG.trData / GG.trLabel not set.")
        if getattr(GG, "M", None) is None:
            GG.M = 2

        self.V_f = int(GG.featNum * GG.kNeigh)
        self.M = int(GG.M)

        super().__init__(
            n_var=self.V_f,
            n_obj=self.M,
            n_ieq_constr=0,
            xl=0,
            xu=1,
            elementwise_evaluation=True,
            elementwise_runner=elementwise_runner
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # IMPORTANT: do NOT return inf/nan (it breaks crowding/termination sometimes)
        PEN_F1 = 1.0                  # f1 = negative accuracy, range usually [-1,0]
        PEN_F2 = float(GG.featNum)    # penalize by selecting many features

        x01 = np.asarray(x, dtype=int).ravel()
        if x01.size != self.V_f:
            out["F"] = np.array([PEN_F1, PEN_F2], dtype=float)
            return

        try:
            indiv_adj = _decode_adj_from_edgechrom(x01)
            f, _ = evaluate_objective_f(self.M, indiv_adj)
            f = np.asarray(f, dtype=float).ravel()
            f = np.nan_to_num(f, nan=PEN_F1, posinf=PEN_F1, neginf=PEN_F1)
            if f.size >= 2 and (not np.isfinite(f[1])):
                f[1] = PEN_F2
            out["F"] = f
        except Exception:
            out["F"] = np.array([PEN_F1, PEN_F2], dtype=float)


# ============================================================
# 4) Optional logging callback (writes into GG.run_logs[current_run])
# ============================================================

class GBFSLogger(Callback):
    def notify(self, algorithm):
        gen = int(getattr(algorithm, "n_gen", -1))
        pop = algorithm.pop
        X = pop.get("X")
        F = pop.get("F")
        if X is None or F is None or len(F) == 0:
            return

        finite = np.isfinite(F).all(axis=1)
        Xf = X[finite]
        Ff = F[finite]
        if Ff.shape[0] == 0:
            return

        acc = (-Ff[:, 0]).astype(float)
        fnum = (Ff[:, 1]).astype(float)

        metric_row = {
            "gen": gen,
            "pop_size_finite": int(Ff.shape[0]),
            "acc_best": float(np.max(acc)),
            "acc_mean": float(np.mean(acc)),
            "acc_std": float(np.std(acc)),
            "fnum_best": float(np.min(fnum)),
            "fnum_mean": float(np.mean(fnum)),
            "fnum_std": float(np.std(fnum)),
        }

        # store pareto front (objectives only, cheap)
        nd_idx = NonDominatedSorting().do(Ff, only_non_dominated_front=True)
        Fnd = Ff[nd_idx]

        run_id = int(getattr(GG, "current_run", -1))
        if not hasattr(GG, "run_logs") or GG.run_logs is None:
            GG.run_logs = {}
        if run_id not in GG.run_logs:
            GG.run_logs[run_id] = {"pareto_fronts": [], "pop_metrics": []}

        GG.run_logs[run_id]["pop_metrics"].append(metric_row)
        GG.run_logs[run_id]["pareto_fronts"].append({
            "gen": gen,
            "df": pd.DataFrame({
                "gen": [gen] * len(Fnd),
                "f1": Fnd[:, 0].astype(float),
                "f2": Fnd[:, 1].astype(float),
                "acc": (-Fnd[:, 0]).astype(float),
                "fnum": (Fnd[:, 1]).astype(float),
            })
        })


# ============================================================
# 5) Runner (OOP) + compatibility wrapper newtry_ms
# ============================================================

class PymooGBFSRunner:
    def __init__(self, pop=20, n_gen=50, n_jobs=None, parallel_backend="thread", seed=1, verbose=True):
        self.pop = int(pop)
        self.n_gen = int(n_gen)
        self.seed = int(seed)
        self.verbose = bool(verbose)

        if n_jobs is None:
            n_jobs = _default_n_jobs()
        self.n_jobs = int(max(1, n_jobs))

        backend = str(parallel_backend).lower().strip()
        if backend not in ("thread", "process", "none"):
            raise ValueError("parallel_backend must be 'thread', 'process', or 'none'")
        self.backend = backend

    def _make_runner(self):
        if self.backend == "none" or self.n_jobs <= 1:
            return None, None

        if StarmapParallelization is None:
            raise ImportError("Your pymoo version does not provide StarmapParallelization in parallelization.starmap")

        if self.backend == "thread":
            from multiprocessing.pool import ThreadPool
            pool = ThreadPool(processes=self.n_jobs)
            runner = StarmapParallelization(pool.starmap)
            return pool, runner

        # process backend (Windows spawn)
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        state = _export_gg_state()
        pool = ctx.Pool(
            processes=self.n_jobs,
            initializer=_init_worker_gg,
            initargs=(state,)
        )
        runner = StarmapParallelization(pool.starmap)
        return pool, runner

    @staticmethod
    def _pick_best_scalar(F, alpha=0.9):
        """
        Similar to your old scalarization idea:
          fits = alpha*abs(f1) + (1-alpha)*(1 - f2/featNum)
        Here f1 is negative accuracy -> abs(f1)=accuracy
        """
        F = np.asarray(F, dtype=float)
        acc = np.abs(F[:, 0])
        fnum = F[:, 1].astype(float)
        n_feat_total = max(1.0, float(GG.featNum))
        fits = alpha * acc + (1.0 - alpha) * (1.0 - fnum / n_feat_total)
        return int(np.argmax(fits))

    def run(self):
        pool, runner = self._make_runner()
        try:
            problem = GBFSEdgeProblem(elementwise_runner=runner)

            algorithm = NSGA2(
                pop_size=self.pop,
                sampling=BinaryRandomSampling(),
                crossover=TwoPointCrossover(prob=0.9),
                mutation=BitflipMutation(prob=0.01),
                eliminate_duplicates=True,
            )

            callback = GBFSLogger()

            # IMPORTANT: use positional termination to avoid version differences
            res = minimize(
                problem,
                algorithm,
                ("n_gen", int(self.n_gen)),
                seed=self.seed,
                verbose=self.verbose,
                callback=callback,
            )

            X = np.asarray(res.X)
            F = np.asarray(res.F)

            if X.ndim == 1:
                X = X.reshape(1, -1)
            if F.ndim == 1:
                F = F.reshape(1, -1)

            finite = np.isfinite(F).all(axis=1)
            Xf = X[finite]
            Ff = F[finite]

            if Xf.shape[0] == 0:
                featidx_best = np.zeros(GG.featNum, dtype=float)
                pareto_masks = np.zeros((0, GG.featNum), dtype=float)
                pareto_objs = np.zeros((0, 2), dtype=float)
                kNeigh_best = np.zeros(problem.V_f, dtype=float)
                return featidx_best, pareto_masks, pareto_objs, kNeigh_best

            # Pareto front among finite solutions
            nd = NonDominatedSorting().do(Ff, only_non_dominated_front=True)
            Xp = Xf[nd]
            pareto_objs = Ff[nd].astype(float)

            # decode pareto to feature masks (cheap)
            pareto_masks = []
            for i in range(Xp.shape[0]):
                mask, _idx = _decode_features_only(np.asarray(Xp[i], dtype=int))
                pareto_masks.append(mask.astype(float))
            pareto_masks = np.vstack(pareto_masks) if len(pareto_masks) else np.zeros((0, GG.featNum), dtype=float)

            # pick best by scalarization
            idx_best = self._pick_best_scalar(Ff, alpha=0.9)
            kNeigh_best = np.asarray(Xf[idx_best]).astype(float).ravel()

            best_mask, _ = _decode_features_only(np.asarray(Xf[idx_best], dtype=int))
            featidx_best = best_mask.astype(float).ravel()

            return featidx_best, pareto_masks, pareto_objs, kNeigh_best

        finally:
            if pool is not None:
                pool.close()
                pool.join()


def newtry_ms(inputAdj, pop=20, times=50, run_dir=None,
             n_jobs=None, parallel_backend="thread", seed=1, verbose=True):
    """
    Compatibility wrapper: same signature style you use in Copy_of_js_ms.py
    """
    runner = PymooGBFSRunner(
        pop=pop,
        n_gen=times,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        seed=seed,
        verbose=verbose
    )
    return runner.run()
