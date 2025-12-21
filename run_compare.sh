#!/bin/bash
#SBATCH --job-name=GBFS_COMPARE
#SBATCH --output=out_%A.out
#SBATCH --error=err_%A.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01-00:00:00
#SBATCH --partition=parallel

cd /nfs/scratch/<user>/GBFS-SND || exit 1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gbfs_enhance

export MPLBACKEND=Agg
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

python Compare/compare_and_log_methods.py