from oop_core import ExperimentConfig, InitParams
from runner_oop import GBFSRunner

cfg = ExperimentConfig(
    runs=3,
    pop=20,
    gen=50,

    init_modes=["knn", "probabilistic"],
    seq_modes=["normal", "buddy"],

    log_mode="knn",
    log_seq_mode="normal",

    buddy_max_per_core=1,
    buddy_lam_red=0.5,
    buddy_cv=5,
    buddy_knn_k=5,
    buddy_seed=42,

    init_params=InitParams(
        k_neigh=5,
        k_min=1,
        quantile=0.8,
        extra_k=2,
        beta=2.0,
        seed=42,
    )
)

runner = GBFSRunner(
    cfg,
    visual_root="D:/PhD/The First Paper/Code Implement/GBFS-SND/Evaluation/Visualize Implementation Final"
)

product, name = runner.run_dataset(1)
print(f"Dataset: {name}")
print("Results (chosen combo log_mode/log_seq_mode):")
for row in product:
    print(row)
