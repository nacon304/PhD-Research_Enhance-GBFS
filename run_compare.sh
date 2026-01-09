#!/bin/bash
set -euo pipefail

OUT_ROOT='/nfs/scratch/<user>/gbfs_compare_results'

# mapping = {
#     1:  ("glass.mat",      "Glass"),
#     2:  ("Urban.mat",      "Urban"),
#     3:  ("Musk1.mat",      "Musk1"),
#     4:  ("USPS.mat",       "USPS"),
#     5:  ("madelon.mat",    "Madelon"),
#     6:  ("ISOLET.mat",     "ISOLET"),
#     7:  ("GINA_01.mat",    "GINA"),
#     8:  ("Bioresponse.mat","Bioresponse"),
#     9:  ("Colon.mat",      "Colon"),
#     10: ("PIE10P.mat",     "PIE10P"),
#     11: ("BASEHOCK.mat",   "BASEHOCK"),
#     12: ("GISETTE.mat",    "GISETTE"),
#     13: ("TOX171.mat",     "TOX_171"),
#     14: ("ARCENE.mat",     "ARCENE"),
# }
datasets=(1 2 3 4 5 6 7 8 9 10 11 12 13 14)
RUNS=10

algos=(
  "baseline"
  "enh:knn:normal"
  "enh:knn:rc_greedy"
  "enh:probabilistic:normal"
  "enh:probabilistic:rc_greedy"

  "all"
  "trad:FILTER_pearson"
  "trad:FILTER_kendall"
  "trad:FILTER_reliefF"
  "trad:FILTER_chi2"
  "trad:FILTER_info_gain"
  "trad:EMB_L1_LogReg"
  "trad:EMB_RF_importance"

  "graph:InfFS"
  "graph:UGFS"
)

for dataset_idx in "${datasets[@]}"; do
  for algo in "${algos[@]}"; do
    safe_algo="${algo//[:]/_}"
    jobname="GBFS_${safe_algo}_d${dataset_idx}"

    echo "Submitting: dataset=${dataset_idx}, algo=${algo}, runs=1..${RUNS}"

    sbatch \
      --job-name="$jobname" \
      --array=1-"$RUNS" \
      job_slurm_one_dataset.sh "$dataset_idx" "$algo"
  done
done
