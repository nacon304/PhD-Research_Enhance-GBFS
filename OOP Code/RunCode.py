from oop_core import ExperimentConfig, InitParams, KShellParams, BuddyParams, EvalParams, LogParams
from runner_oop import GBFSRunner

cfg = ExperimentConfig(
    runs=30,
    pop=20,
    gen=50,

    init_modes=["knn", "probabilistic"],
    kshell_seq_modes=["normal", "rc_greedy"],
    post_seq_modes=["normal", "buddy"],

    log_init_mode="probabilistic",
    log_kshell_seq_mode="rc_greedy",
    log_post_seq_mode="buddy",

    init_params=InitParams(
        k_neigh=5,
        k_min=1,
        quantile=0.8,
        extra_k=2,
        beta=2.0,
        seed=42,
    ),

    kshell_params=KShellParams(
        max_add=5,
        rc_tau=0.3,
    ),

    buddy_params=BuddyParams(
        max_per_core=1,
        lam_red=0.5,
        cv=5,
        knn_k=5,
        seed=42,
    ),

    eval_params=EvalParams(
        test_size=0.3,
        split_seed=42,
        knn_eval_k=5,
    ),

    log_params=LogParams(
        enabled=True,
        log_only_selected_combo=False,   # True nếu chỉ muốn export log nặng cho combo log_*
        export_init_graph=True,
        export_solver_meta=True,
        export_solver_logs=True,
        export_post_logs=True,
        export_plots=False,
        keep_run_logs_in_memory=False,
    )
)

runner = GBFSRunner(cfg, visual_root="D:/PhD/The First Paper/Code Implement/GBFS-SND/Evaluation/Visualize Implementation Final v2")
product, name = runner.run_dataset(1)
print("Dataset:", name)
for row in product:
    print(row)