#pragma once

#include "cfd/common.hpp"
#include <vector>

namespace cfd {

struct HybridConfig {
    int mpi_rank{0};
    int mpi_size{1};
    int omp_threads{1};
};

struct HybridBenchmarkResult {
    int mpi_ranks;
    int omp_threads_per_rank;
    int total_threads;
    double computation_time;
    double communication_time;
    double total_time;
    double speedup;
};

// Initializes MPI with thread support (MPI_THREAD_FUNNELED or MPI_THREAD_SERIALIZED)
bool init_hybrid_environment(int* argc, char*** argv, int& rank, int& size);

// Decomposes 3D grid along x-axis and computes vorticity using MPI domain distribution + local OpenMP
void hybrid_vorticity(const double* global_u, const double* global_v, const double* global_w,
                      const Grid3D& global_grid,
                      double* global_wx, double* global_wy, double* global_wz,
                      int omp_threads = 0);

// Hybrid Q-criterion
void hybrid_qcriterion(const double* global_u, const double* global_v, const double* global_w,
                       const Grid3D& global_grid,
                       double* global_q,
                       int omp_threads = 0);

} // namespace cfd
