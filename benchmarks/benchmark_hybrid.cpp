#include "cfd/common.hpp"
#include "cfd/grid.hpp"
#include "cfd/hybrid_mpi.hpp"
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cstdlib>

int main(int argc, char** argv) {
    int rank = 0;
    int size = 1;
    cfd::init_hybrid_environment(&argc, &argv, rank, size);

    int omp_threads = omp_get_max_threads();
    const char* env_threads = std::getenv("OMP_NUM_THREADS");
    if (env_threads) {
        omp_threads = std::atoi(env_threads);
    }

    size_t nx = 128, ny = 128, nz = 128;
    if (argc > 1) nx = std::stoul(argv[1]);
    if (argc > 2) ny = std::stoul(argv[2]);
    if (argc > 3) nz = std::stoul(argv[3]);

    const cfd::Grid3D global_grid(nx, ny, nz, 0.01, 0.01, 0.01);
    const size_t total_cells = global_grid.total_cells();

    if (rank == 0) {
        std::cout << "========================================================================\n";
        std::cout << " ParallelCFD: Hybrid MPI + OpenMP Benchmark\n";
        std::cout << "========================================================================\n";
        std::cout << "MPI Ranks: " << size << " | OpenMP Threads per Rank: " << omp_threads
                  << " | Total Compute Cores: " << size * omp_threads << "\n";
        std::cout << "Global Grid: " << nx << "x" << ny << "x" << nz
                  << " (" << total_cells << " cells)\n\n";
    }

    std::vector<double> u, v, w, q;
    if (rank == 0) {
        cfd::Field3D field = cfd::generate_taylor_green_vortex(global_grid);
        u = std::move(field.u);
        v = std::move(field.v);
        w = std::move(field.w);
        q.resize(total_cells, 0.0);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    const int reps = 3;
    for (int r = 0; r < reps; ++r) {
        cfd::hybrid_qcriterion(u.data(), v.data(), w.data(), global_grid, q.data(), omp_threads);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    double total_time = (t_end - t_start) / reps;

    if (rank == 0) {
        double throughput = (total_cells / 1e6) / total_time;
        std::cout << "Completed in " << std::fixed << std::setprecision(4) << total_time * 1000.0 << " ms\n";
        std::cout << "Throughput:   " << std::fixed << std::setprecision(2) << throughput << " Mcells/s\n";

        // Append to results/hybrid_scaling.csv
        std::ofstream csv("results/hybrid_scaling.csv", std::ios::app);
        if (csv.is_open()) {
            csv << size << "," << omp_threads << "," << size * omp_threads << ","
                << total_time << "," << throughput << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
