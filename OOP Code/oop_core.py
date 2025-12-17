# oop_core.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple
import numpy as np

# dùng lại các hàm init hiện có
from init_strategies import (
    init_graph_local_threshold,
    init_graph_fisher_redundancy_degree,
    init_graph_mst_plus_local,
    init_graph_probabilistic,
)

@dataclass
class InitParams:
    # shared
    exclude_self: bool = True
    symmetrize: bool = True

    # knn
    k_neigh: int = 5

    # local_thresh
    k_min: int = 1
    quantile: float = 0.8

    # fisher/prob
    k_max: Optional[int] = None
    beta: float = 2.0
    seed: int = 42

    # mst_plus_local
    extra_k: int = 2

@dataclass
class ExperimentConfig:
    runs: int = 10
    test_size: float = 0.3
    split_seed: int = 42

    pop: int = 20
    gen: int = 50

    knn_eval_k: int = 5

    # init strategies bật/tắt ở đây
    init_modes: List[str] = field(default_factory=lambda: ["knn"])

    # mode nào được dùng để fill FSET/assiNum (giữ format output cũ)
    log_mode: str = "knn"

    init_params: InitParams = field(default_factory=InitParams)

@dataclass
class GraphInitResult:
    A_init: np.ndarray                 # (n,n)
    neighbors: List[np.ndarray]        # neighbors[i] array
    degrees: Optional[np.ndarray] = None

class GraphInitializer(Protocol):
    name: str
    def build(self, Z: np.ndarray, fisher_raw: Optional[np.ndarray], cfg: ExperimentConfig) -> GraphInitResult: ...

class KNNInitializer:
    name = "knn"
    def build(self, Z, fisher_raw, cfg):
        Z = np.asarray(Z, float)
        n = Z.shape[0]
        k = cfg.init_params.k_neigh
        A = np.zeros_like(Z, float)
        neigh = []
        for i in range(n):
            row = Z[i].copy()
            row[i] = 0.0
            idx = np.argsort(-row)[:k]
            neigh.append(idx.astype(int))
            A[i, idx] = Z[i, idx]
        return GraphInitResult(A_init=A, neighbors=neigh)

class LocalThreshInitializer:
    name = "local_thresh"
    def build(self, Z, fisher_raw, cfg):
        p = cfg.init_params
        A, neigh = init_graph_local_threshold(
            Z,
            k_min=p.k_min,
            quantile=p.quantile,
            exclude_self=p.exclude_self,
            symmetrize=p.symmetrize,
        )
        return GraphInitResult(A_init=A, neighbors=neigh)

class FisherRedDegInitializer:
    name = "fisher_red_deg"
    def build(self, Z, fisher_raw, cfg):
        if fisher_raw is None:
            raise ValueError("fisher_raw is required for fisher_red_deg")
        p = cfg.init_params
        n = Z.shape[0]
        k_max = p.k_max if p.k_max is not None else min(p.k_neigh * 3, n - 1)
        A, degs, neigh = init_graph_fisher_redundancy_degree(
            Z,
            fisher_scores=fisher_raw,
            k_min=p.k_min,
            k_max=k_max,
            exclude_self=p.exclude_self,
            symmetrize=p.symmetrize,
        )
        return GraphInitResult(A_init=A, neighbors=neigh, degrees=degs)

class MSTPlusLocalInitializer:
    name = "mst_plus_local"
    def build(self, Z, fisher_raw, cfg):
        p = cfg.init_params
        A, neigh = init_graph_mst_plus_local(
            Z,
            extra_k=p.extra_k,
            exclude_self=p.exclude_self,
            symmetrize=p.symmetrize,
        )
        return GraphInitResult(A_init=A, neighbors=neigh)

class ProbabilisticInitializer:
    name = "probabilistic"
    def build(self, Z, fisher_raw, cfg):
        if fisher_raw is None:
            raise ValueError("fisher_raw is required for probabilistic")
        p = cfg.init_params
        n = Z.shape[0]
        k_max = p.k_max if p.k_max is not None else min(p.k_neigh * 3, n - 1)
        A, neigh = init_graph_probabilistic(
            Z,
            fisher_scores=fisher_raw,
            k_min=p.k_min,
            k_max=k_max,
            beta=p.beta,
            exclude_self=p.exclude_self,
            symmetrize=p.symmetrize,
            random_state=p.seed,
        )
        return GraphInitResult(A_init=A, neighbors=neigh)

INIT_REGISTRY: Dict[str, GraphInitializer] = {
    "knn": KNNInitializer(),
    "local_thresh": LocalThreshInitializer(),
    "fisher_red_deg": FisherRedDegInitializer(),
    "mst_plus_local": MSTPlusLocalInitializer(),
    "probabilistic": ProbabilisticInitializer(),
}

def get_initializer(name: str) -> GraphInitializer:
    if name not in INIT_REGISTRY:
        raise KeyError(f"Unknown initializer: {name}")
    return INIT_REGISTRY[name]
