#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " Running ParallelCFD Complete Benchmark & Test Suite"
echo "=========================================================="

# 1. Build C++ and Python extensions
echo "[1/5] Building project via CMake..."
cmake -B build -S . -DPython3_EXECUTABLE=$(pwd)/.venv/bin/python
cmake --build build -j$(nproc)

# 2. Run C++ Unit Tests
echo "[2/5] Running C++ unit tests..."
./build/test_cfd_kernels
./build/test_experiments

# 3. Run Python Unit Tests
echo "[3/5] Running Python unit tests..."
.venv/bin/pytest tests/test_python_api.py -v

# 4. Run Benchmarks
echo "[4/5] Executing scaling and architecture benchmarks..."
./build/benchmark_cfd_kernels 128
./build/benchmark_experiments

echo "Running Hybrid MPI+OpenMP sweeps on 16 cores..."
rm -f results/hybrid_scaling.csv
OMP_NUM_THREADS=1 mpirun -n 16 ./build/benchmark_hybrid
OMP_NUM_THREADS=2 mpirun -n 8 ./build/benchmark_hybrid
OMP_NUM_THREADS=4 mpirun -n 4 ./build/benchmark_hybrid
OMP_NUM_THREADS=8 mpirun -n 2 ./build/benchmark_hybrid
OMP_NUM_THREADS=16 mpirun -n 1 ./build/benchmark_hybrid

# 5. Generate Visualizations and Plots
echo "[5/5] Generating PyVista renders and Matplotlib plots..."
.venv/bin/python python/examples/02_pyvista_qcriterion_vis.py
.venv/bin/python benchmarks/run_and_plot.py

echo "=========================================================="
echo " ParallelCFD Suite Completed Successfully! Results in results/"
echo "=========================================================="
