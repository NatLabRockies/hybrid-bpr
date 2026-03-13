#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --account=smartipc
#SBATCH --qos=high
#SBATCH --job-name=pybpr_zazzle_sweep
#SBATCH --output=logs/pybpr_zazzle_sweep-%j.out
##SBATCH --partition=debug

# Usage (with or without SLURM):
#   mkdir -p logs
#   sbatch submit_sweep.sh            # standard MLflow
#   sbatch submit_sweep.sh --hero     # Hero MLflow backend
#   bash   submit_sweep.sh            # run locally
#   bash   submit_sweep.sh --hero

# cd to project root; works with sbatch (SLURM_SUBMIT_DIR) or bash
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$(dirname "$SUBMIT_DIR")" || exit 1

# Parse --hero flag
USE_HERO=0
for arg in "$@"; do
    [[ "$arg" == "--hero" ]] && USE_HERO=1
done

# Build extra args for runner
EXTRA_ARGS="--sweep"
[[ $USE_HERO -eq 1 ]] && EXTRA_ARGS="$EXTRA_ARGS --hero"

# Divide available CPUs evenly across event types; run all in parallel
EVENT_TYPES=("clicks" "orders")
N_EVENTS=${#EVENT_TYPES[@]}
TOTAL_CPUS=${SLURM_CPUS_ON_NODE:-$(nproc)}
CPUS_PER_EVENT=$(( TOTAL_CPUS / N_EVENTS ))

echo "======================================"
echo "Job ID   : ${SLURM_JOB_ID:-local}"
echo "Node     : ${SLURMD_NODENAME:-$(hostname)}"
echo "CPUs     : $TOTAL_CPUS total / $CPUS_PER_EVENT per event type"
echo "Hero     : $USE_HERO"
echo "Start    : $(date)"
echo "======================================"

# Launch all event types in parallel; each gets its CPU share
for EVENT in "${EVENT_TYPES[@]}"; do
    echo "--- Launching: event_type=$EVENT ---"
    uv run python zazzle/run_zazzle.py \
        --event-type "$EVENT" \
        --n-jobs "$CPUS_PER_EVENT" \
        $EXTRA_ARGS &
done

# Wait for all background jobs to finish
wait

echo "======================================"
echo "End: $(date)"
echo "======================================"
