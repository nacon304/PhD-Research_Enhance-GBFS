#!/bin/bash
set -euo pipefail

homeDir='/nfs/scratch/<user>/GBFS-SND'
cd "$homeDir/Compare" || exit 1

# -----------------------------
# List of dataset indices
# -----------------------------
# Dataset index mapping (1–10):
#  1: Glass        (glass.mat)
#  2: Urban        (Urban.mat)
#  3: Musk1        (Musk1.mat)
#  4: Madelon      (madelon.mat)
#  5: Bioresponse  (Bioresponse.mat)
#  6: Colon        (Colon.mat)
#  7: PIE10P       (PIE10P.mat)
#  8: BASEHOCK     (BASEHOCK.mat)
#  9: GISETTE      (GISETTE.mat)
# 10: TOX_171      (TOX171.mat)
datasets=(
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
)

RUNS=30

for dataset_idx in "${datasets[@]}"; do
    echo "Submitting: dataset_idx=${dataset_idx}, runs=1..${RUNS}"
    sbatch --array=1-"$RUNS" job_slurm_one_dataset.sh "$dataset_idx"
done
