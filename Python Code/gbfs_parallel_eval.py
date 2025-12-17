# gbfs_parallel_eval.py
import os
import numpy as np
import gbfs_globals as GG

from decodeNet import decodeNet
from evaluate_objective_f import evaluate_objective_f

# ---- globals inside worker (for process backend) ----
_WORKER_M = None
_WORKER_TEMPLATE = None
_WORKER_GG_STATE = None


def _pack_gg_state():
    """Pick only what worker really needs."""
    return {
        "data": GG.data,
        "trData": GG.trData,
        "trLabel": GG.trLabel,
        "featNum": GG.featNum,
        "kNeigh": GG.kNeigh,
        "kNeiMatrix": GG.kNeiMatrix,
        "kNeiZout": GG.kNeiZout,
        "M": GG.M,
    }


def _apply_gg_state(state: dict):
    GG.data = state["data"]
    GG.trData = state["trData"]
    GG.trLabel = state["trLabel"]
    GG.featNum = int(state["featNum"])
    GG.kNeigh = int(state["kNeigh"])
    GG.kNeiMatrix = state["kNeiMatrix"]
    GG.kNeiZout = state["kNeiZout"]
    GG.M = int(state.get("M", 2))


def _init_worker(state, M, templateAdj):
    # avoid oversubscription (optional but recommended)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    global _WORKER_GG_STATE, _WORKER_M, _WORKER_TEMPLATE
    _WORKER_GG_STATE = state
    _WORKER_M = int(M)
    _WORKER_TEMPLATE = templateAdj

    _apply_gg_state(_WORKER_GG_STATE)


def _eval_one(edge_row):
    """edge_row: shape (V,), 0/1"""
    try:
        indiv_net = decodeNet(edge_row, _WORKER_TEMPLATE)
        f, feat_mask = evaluate_objective_f(_WORKER_M, indiv_net)

        f = np.asarray(f, dtype=float).ravel()

        # OPTIONAL: avoid inf that can break some routines
        if not np.isfinite(f).all():
            # worst penalty: f1=1 (acc=-1), f2=featNum
            f = np.array([1.0, float(GG.featNum)], dtype=float)

        return f, np.asarray(feat_mask, dtype=bool).ravel()

    except Exception:
        f = np.array([1.0, float(GG.featNum)], dtype=float)
        feat_mask = np.zeros(int(GG.featNum), dtype=bool)
        return f, feat_mask


class GBFSParallelEvaluator:
    """
    Reusable evaluator for batches of edge chromosomes.
    backend:
      - "thread": safe, no GG-state copy (recommended first)
      - "process": faster sometimes, but needs __main__ guard (Windows)
    """

    def __init__(self, M, templateAdj, n_jobs=8, backend="thread"):
        self.M = int(M)
        self.templateAdj = templateAdj
        self.n_jobs = int(max(1, n_jobs))
        self.backend = str(backend).lower().strip()

        self.pool = None
        self.runner = None

    def __enter__(self):
        from pymoo.parallelization.starmap import StarmapParallelization  # :contentReference[oaicite:2]{index=2}

        if self.backend == "thread" or self.n_jobs <= 1:
            from multiprocessing.pool import ThreadPool
            self.pool = ThreadPool(processes=self.n_jobs)
            self.runner = StarmapParallelization(self.pool.starmap)
            # thread shares GG state, no initializer needed
            return self

        if self.backend == "process":
            import multiprocessing as mp
            ctx = mp.get_context("spawn")  # Windows-safe
            state = _pack_gg_state()
            self.pool = ctx.Pool(
                processes=self.n_jobs,
                initializer=_init_worker,
                initargs=(state, self.M, self.templateAdj),
            )
            self.runner = StarmapParallelization(self.pool.starmap)
            return self

        raise ValueError("backend must be 'thread' or 'process'")

    def __exit__(self, exc_type, exc, tb):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()

    def evaluate_batch(self, X_edges: np.ndarray):
        """
        X_edges: (N, V) binary
        returns:
          F: (N, M)
          masks: (N, featNum)
        """
        X_edges = np.asarray(X_edges, dtype=int)
        if X_edges.ndim != 2:
            raise ValueError("X_edges must be 2D (N,V)")

        # For thread backend: set globals directly once
        if self.backend == "thread":
            _init_worker(_pack_gg_state(), self.M, self.templateAdj)

        tasks = [(X_edges[i, :],) for i in range(X_edges.shape[0])]
        results = self.runner(_eval_one, tasks)  # uses starmap interface :contentReference[oaicite:3]{index=3}

        F = np.vstack([r[0] for r in results]).astype(float)
        masks = np.vstack([r[1] for r in results]).astype(bool)
        return F, masks
