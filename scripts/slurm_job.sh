#!/usr/bin/env bash
#SBATCH --job-name=parallelcfd-bench
#SBATCH --output=results/slurm_%j.out
#SBATCH --error=results/slurm_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --partition=standard

# ==============================================================================
# ParallelCFD: HPC SLURM Job Submission Script
# Configured for Hybrid MPI + OpenMP Execution on Multi-Core Compute Nodes
# ==============================================================================

echo "=========================================================================="
echo " Starting Job ID: $SLURM_JOB_ID on $(hostname)"
echo " Allocated Nodes: $SLURM_JOB_NODELIST"
echo " Total Tasks:     $SLURM_NTASKS"
echo " CPUs per Task:   $SLURM_CPUS_PER_TASK"
echo "=========================================================================="

# 1. Environment & Module Load (adjust module names per cluster, e.g. Betzy/Fram)
# module purge
# module load GCC/13.2.0 OpenMPI/4.1.6 CMake/3.27.6 Python/3.11.5

# 2. OpenMP Thread Pinning and Affinity
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "OpenMP Configuration:"
echo "  OMP_NUM_THREADS = $OMP_NUM_THREADS"
echo "  OMP_PROC_BIND   = $OMP_PROC_BIND"
echo "  OMP_PLACES      = $OMP_PLACES"

# 3. Step 1: Shared-Memory OpenMP Scaling Benchmark on Node 1
echo -e "\n[Step 1] Running Shared-Memory OpenMP Benchmark on 1 Node..."
./build/benchmark_cfd_kernels 256
./build/benchmark_experiments

# 4. Step 2: Hybrid MPI + OpenMP Across Allocated Nodes
echo -e "\n[Step 2] Running Hybrid MPI + OpenMP Domain Decomposition..."
# Using srun to bind MPI ranks and pass OpenMP cpus
srun --ntasks=$SLURM_NTASKS --cpus-per-task=$SLURM_CPUS_PER_TASK ./build/benchmark_hybrid 256 256 256

# 5. Step 3: Python Post-Processing and Figure Generation
echo -e "\n[Step 3] Generating Scaling Figures..."
python3 benchmarks/run_and_plot.py

echo "=========================================================================="
echo " Job Completed Successfully!"
echo "=========================================================================="
