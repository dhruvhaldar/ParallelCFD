#include "cfd/common.hpp"
#include "cfd/grid.hpp"
#include "cfd/vorticity.hpp"
#include "cfd/qcriterion.hpp"
#include "cfd/hybrid_mpi.hpp"

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

int main(int argc, char** argv) {
    int rank = 0;
    int size = 1;
    cfd::init_hybrid_environment(&argc, &argv, rank, size);

    const size_t nx = 64, ny = 64, nz = 64;
    const cfd::Grid3D grid(nx, ny, nz, 0.05, 0.05, 0.05);
    const size_t total_cells = grid.total_cells();

    std::vector<double> u, v, w;
    std::vector<double> q_hybrid(total_cells, 0.0);
    std::vector<double> q_reference(total_cells, 0.0);

    if (rank == 0) {
        std::cout << "========================================================\n";
        std::cout << " Running Hybrid MPI + OpenMP Correctness Test\n";
        std::cout << " MPI Ranks: " << size << ", Grid: " << nx << "x" << ny << "x" << nz << "\n";
        std::cout << "========================================================\n";

        cfd::Field3D field = cfd::generate_taylor_green_vortex(grid);
        u = std::move(field.u);
        v = std::move(field.v);
        w = std::move(field.w);

        // Compute local single-node reference
        cfd::q_criterion_serial(u.data(), v.data(), w.data(), grid, q_reference.data());
    }

    // Run hybrid MPI + OpenMP calculation
    cfd::hybrid_qcriterion(u.data(), v.data(), w.data(), grid, q_hybrid.data(), 2);

    if (rank == 0) {
        double max_diff = 0.0;
        for (size_t i = 0; i < total_cells; ++i) {
            double diff = std::abs(q_hybrid[i] - q_reference[i]);
            if (diff > max_diff) max_diff = diff;
        }

        std::cout << "  Max difference between Hybrid and Single-Node Reference: " << max_diff << "\n";
        assert(max_diff < 1e-12);
        std::cout << "\n>>> HYBRID MPI + OPENMP TEST PASSED SUCCESSFULLY! <<<\n";
    }

    MPI_Finalize();
    return 0;
}
