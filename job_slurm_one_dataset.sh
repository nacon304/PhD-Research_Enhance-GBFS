#!/bin/bash
#SBATCH --job-name=GBFS_ONE
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01-00:00:00
#SBATCH --partition=parallel
#SBATCH --output=/nfs/scratch/<user>/gbfs_compare_results/Out/%x_%A_%a.out
#SBATCH --error=/nfs/scratch/<user>/gbfs_compare_results/Err/%x_%A_%a.err

set -euo pipefail

homeDir='/nfs/scratch/<user>/GBFS-SND'
OUT_ROOT='/nfs/scratch/<user>/gbfs_compare_results_v2'

dataset_idx="$1"
algo="$2"
shift 2
EXTRA_ARGS=("$@")

RUN_ID="$SLURM_ARRAY_TASK_ID"
SEED="$SLURM_ARRAY_TASK_ID"

mkdir -p "$OUT_ROOT/Out" "$OUT_ROOT/Err"

cd "$homeDir" || exit 1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gbfs_enhance

export MPLBACKEND=Agg
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# One job = one dataset + one run + one algo
python Compare/run_one_algo.py \
  --dataset_idx "$dataset_idx" \
  --run "$RUN_ID" \
  --algo "$algo" \
  --out_root "$OUT_ROOT" \
  --baseline_root "$homeDir/Python Code" \
  --oop_root "$homeDir/OOP Code" \
  "${EXTRA_ARGS[@]}"
