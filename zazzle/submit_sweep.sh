#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=6:00:00
#SBATCH --account=smartipc
#SBATCH --qos=high
#SBATCH --job-name=pybpr_zazzle_sweep
#SBATCH --output=logs/pybpr_zazzle_sweep-%j.out

# Usage (with or without SLURM):
#   mkdir -p logs
#   sbatch submit_sweep.sh   # via SLURM
#   bash   submit_sweep.sh   # run locally
# SLURM_SUBMIT_DIR is zazzle/; project root is one level up
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$(dirname "$SUBMIT_DIR")" || exit 1

# Divide CPUs evenly across event types; run in parallel
EVENT_TYPES=("clicks" "orders")
N_EVENTS=${#EVENT_TYPES[@]}
TOTAL_CPUS=${SLURM_CPUS_ON_NODE:-$(nproc)}
CPUS_PER_EVENT=$(( TOTAL_CPUS / N_EVENTS ))

echo "======================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "CPUs     : $TOTAL_CPUS total / $CPUS_PER_EVENT per event type"
echo "Start    : $(date)"
echo "======================================"

# Launch each event type in parallel
for EVENT in "${EVENT_TYPES[@]}"; do
    echo "--- Launching: event_type=$EVENT ---"
    uv run python zazzle/run_zazzle.py \
        --event-type "$EVENT" \
        --n-jobs "$CPUS_PER_EVENT" \
        --sweep \
        --hero &
done

# Wait for all background jobs to finish
wait

echo "======================================"
echo "End: $(date)"
echo "======================================"
