#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --account=smartipc
#SBATCH --qos=high
#SBATCH --job-name=pybpr_sweep
#SBATCH --output=logs/pybpr_sweep-%j.out
##SBATCH --partition=debug

# Usage (with or without SLURM):
#   mkdir -p logs
#   sbatch submit_sweep.sh ml-100k
#   sbatch submit_sweep.sh ml-10m  --hero
#   sbatch submit_sweep.sh ml-100k --experiment my-exp
#   bash   submit_sweep.sh ml-100k            # run locally
#   bash   submit_sweep.sh ml-10m  --hero

# cd into movielens dir; works with sbatch (SLURM_SUBMIT_DIR) or bash
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$SUBMIT_DIR" || exit 1

# Parse dataset (required first positional arg), --hero, --experiment
DATASET=""
USE_HERO=0
EXPERIMENT=""
args=("$@")
i=0
while [[ $i -lt ${#args[@]} ]]; do
    arg="${args[$i]}"
    if [[ "$arg" == "--hero" ]]; then
        USE_HERO=1
    elif [[ "$arg" == "--experiment" ]]; then
        i=$(( i + 1 ))
        EXPERIMENT="${args[$i]}"
    elif [[ "$arg" != --* ]] && [[ -z "$DATASET" ]]; then
        DATASET="$arg"
    fi
    i=$(( i + 1 ))
done

if [[ -z "$DATASET" ]]; then
    echo "Error: dataset argument required (e.g. ml-100k or ml-10m)"
    exit 1
fi

# Build extra args for runner
EXTRA_ARGS="--sweep"
[[ $USE_HERO -eq 1 ]]      && EXTRA_ARGS="$EXTRA_ARGS --hero"
[[ -n "$EXPERIMENT" ]]     && EXTRA_ARGS="$EXTRA_ARGS --experiment $EXPERIMENT"

TOTAL_CPUS=${SLURM_CPUS_ON_NODE:-$(nproc)}

echo "======================================"
echo "Job ID   : ${SLURM_JOB_ID:-local}"
echo "Node     : ${SLURMD_NODENAME:-$(hostname)}"
echo "CPUs     : $TOTAL_CPUS"
echo "Dataset  : $DATASET"
echo "Experiment: ${EXPERIMENT:-<from config>}"
echo "Hero     : $USE_HERO"
echo "Start    : $(date)"
echo "======================================"

# Launch sweep for the specified dataset
uv run python run_movielens.py \
    --dataset "$DATASET" \
    --n-jobs "$TOTAL_CPUS" \
    $EXTRA_ARGS

echo "======================================"
echo "End: $(date)"
echo "======================================"
