#!/usr/bin/env bash
# ==============================================================================
# ParallelCFD: Thread Scaling & CPU Affinity Experiment Runner
# Tests strong scaling and investigates OMP_PROC_BIND and OMP_PLACES
# ==============================================================================

set -e

GRID_SIZE=${1:-128}
echo "=========================================================================="
echo " Running Scaling Study on Grid ${GRID_SIZE}^3"
echo "=========================================================================="

echo "--- 1. Baseline Scaling (No Affinity Enforced) ---"
./build/benchmark_cfd_kernels "$GRID_SIZE"

echo -e "\n--- 2. CPU Affinity Experiment (OMP_PROC_BIND=close, OMP_PLACES=cores) ---"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
./build/benchmark_cfd_kernels "$GRID_SIZE"

echo -e "\n--- 3. CPU Affinity Experiment (OMP_PROC_BIND=spread, OMP_PLACES=sockets) ---"
export OMP_PROC_BIND=spread
export OMP_PLACES=sockets
./build/benchmark_cfd_kernels "$GRID_SIZE"

unset OMP_PROC_BIND
unset OMP_PLACES

echo -e "\nRegenerating scaling plots..."
.venv/bin/python benchmarks/run_and_plot.py
echo "Done!"
