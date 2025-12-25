#!/bin/bash
#SBATCH --job-name=GBFS_COMPARE
#SBATCH --output=out_array_%A_%a.out
#SBATCH --error=out_array_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01-00:00:00
#SBATCH --partition=parallel

set -euo pipefail

homeDir='/nfs/scratch/<user>/GBFS-SND'
OUT_ROOT='/nfs/scratch/<user>/gbfs_compare_results'

dataset_idx="$1"

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

# ---------- run one job = one dataset + one run ----------
python Compare/compare_and_log_methods.py \
  --dataset_idx "$dataset_idx" \
  --run "$RUN_ID" \
  --out_root "$OUT_ROOT"

mv "out_array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "$OUT_ROOT/Out/" 2>/dev/null || true
mv "out_array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "$OUT_ROOT/Err/" 2>/dev/null || true
