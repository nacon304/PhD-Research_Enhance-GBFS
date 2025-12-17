from oop_core import ExperimentConfig, InitParams
from runner_oop import GBFSRunner

cfg = ExperimentConfig(
    runs=30,
    pop=20,
    gen=50,
    init_modes=["knn"],  # bật/tắt ở đây
    log_mode="knn",
    init_params=InitParams(
        k_neigh=5,
        k_min=1,
        quantile=0.8,
        extra_k=2,
        beta=2.0,
        seed=42,
    )
)

runner = GBFSRunner(cfg, visual_root="D:/PhD/The First Paper/Code Implement/GBFS-SND/Evaluation/Visualize Implementation Final")
product, name = runner.run_dataset(1)
print(f"Dataset: {name}")
print("Results:")
for row in product:
    print(row)
