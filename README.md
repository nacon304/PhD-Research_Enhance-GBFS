# GBFS-Enhance

## 0) Repository structure

- Compare/ : comparison pipeline (train CV fronts + test evaluation)
- OOP Code/ : enhanced GBFS (OOP implementation)
- Python Code/ : baseline GBFS + traditional/graph methods
- Data/ : datasets

## 1) Create the Conda environment (portable)

On HPC (Linux):

    source ~/miniconda3/etc/profile.d/conda.sh
    conda env create -f environment.yml -n gbfs_enhance
    conda activate gbfs_enhance

## 2) Dataset location

Default dataset directory:

- <repo_root>/Data

Link data: [Google Drive](https://drive.google.com/file/d/1vzxGjahPR2vf3dAXPLEjAuZTip6PGgD7/view?usp=sharing)

## 3) Run the comparison script

From the repo root:

    export MPLBACKEND=Agg
    python Compare/compare_and_log_methods.py

Outputs are written under the OUT_ROOT configured inside compare_and_log_methods.py.

## 4) Run on HPC via SLURM

Submit:
bash run_compare.sh
