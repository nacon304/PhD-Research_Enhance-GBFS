#!/bin/bash
set -euo pipefail

OUT_ROOT='/nfs/scratch/<user>/gbfs_compare_results_v2'

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
datasets_fixed_fnum=(1 2 3 4 5 6 7 8 9 10 12 13)
datasets_sensitivity=(2 3 5 6 7 8 9 10)
RUNS_fixed_fnum=40

algos_fixed_fnum=(
  # "baseline"
  # "enh:knn:normal"
  # "enh:knn:rc_greedy"
  # "enh:probabilistic:normal"
  # "enh:probabilistic:rc_greedy"

  # "all"
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

for dataset_idx in "${datasets_fixed_fnum[@]}"; do
  for algo in "${algos_fixed_fnum[@]}"; do
    safe_algo="${algo//[:]/_}"
    jobname="GBFS_${safe_algo}_d${dataset_idx}"

    echo "Submitting: dataset=${dataset_idx}, algo=${algo}, runs=1..${RUNS_fixed_fnum}"

    sbatch \
      --job-name="$jobname" \
      --array=1-"$RUNS_fixed_fnum" \
      job_slurm_one_dataset.sh "$dataset_idx" "$algo" \
      --fixed_fnum True
  done
done

algo_enhanced="enh:probabilistic:rc_greedy"
# Default
BETA0="2.0"
TAU0="0.3"
LAMBDA0="0.5"
RUNS_sensitivity=10

BETAS=(1.4 1.7 2.0 2.3)
TAUS=(0.0 0.3 0.6 0.9)
LAMBDAS=(0.2 0.5 0.7 1.0)

# -------- beta sweep --------
for dataset_idx in "${datasets_sensitivity[@]}"; do
  for b in "${BETAS[@]}"; do
    tag="beta${b//./p}"
    jobname="SENS_${tag}_d${dataset_idx}"
    echo "Submitting SENS(beta): d=${dataset_idx} beta=${b} runs=1..${RUNS_sensitivity}"

    sbatch \
      --job-name="$jobname" \
      --array=1-"$RUNS_sensitivity" \
      job_slurm_one_dataset.sh "$dataset_idx" "$algo_enhanced" \
      --prob_beta "$b" --ks_rc_tau "$TAU0" --buddy_lambda "$LAMBDA0"
  done
done

# -------- tau sweep --------
for dataset_idx in "${datasets_sensitivity[@]}"; do
  for t in "${TAUS[@]}"; do
    tag="tau${t//./p}"
    jobname="SENS_${tag}_d${dataset_idx}"
    echo "Submitting SENS(tau): d=${dataset_idx} tau=${t} runs=1..${RUNS_sensitivity}"

    sbatch \
      --job-name="$jobname" \
      --array=1-"$RUNS_sensitivity" \
      job_slurm_one_dataset.sh "$dataset_idx" "$algo_enhanced" \
      --prob_beta "$BETA0" --ks_rc_tau "$t" --buddy_lambda "$LAMBDA0"
  done
done

# -------- lambda sweep --------
for dataset_idx in "${datasets_sensitivity[@]}"; do
  for l in "${LAMBDAS[@]}"; do
    tag="lam${l//./p}"
    jobname="SENS_${tag}_d${dataset_idx}"
    echo "Submitting SENS(lambda): d=${dataset_idx} lambda=${l} runs=1..${RUNS_sensitivity}"

    sbatch \
      --job-name="$jobname" \
      --array=1-"$RUNS_sensitivity" \
      job_slurm_one_dataset.sh "$dataset_idx" "$algo_enhanced" \
      --prob_beta "$BETA0" --ks_rc_tau "$TAU0" --buddy_lambda "$l"
  done
done
