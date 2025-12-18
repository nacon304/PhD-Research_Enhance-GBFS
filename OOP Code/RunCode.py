from oop_core import ExperimentConfig, InitParams
from runner_oop import GBFSRunner

cfg = ExperimentConfig(
    runs=3,
    pop=20,
    gen=50,

    init_modes=["knn", "probabilistic"],
    kshell_seq_modes=["normal", "rc_greedy"],
    post_seq_modes=["normal", "buddy"],

    kshell_max_add=5,
    kshell_rc_tau=0.3,

    log_init_mode="probabilistic",
    log_kshell_seq_mode="rc_greedy",
    log_post_seq_mode="buddy",

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

runner = GBFSRunner(cfg, visual_root="D:/PhD/The First Paper/Code Implement/GBFS-SND/Evaluation/Visualize Implementation Final")
product, name = runner.run_dataset(1)
print("Dataset:", name)
for row in product:
    print(row)
