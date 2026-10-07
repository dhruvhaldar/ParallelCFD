#include "cfd/hybrid_mpi.hpp"
#include "cfd/vorticity.hpp"
#include "cfd/qcriterion.hpp"
#include <mpi.h>
#include <vector>
#include <iostream>
#include <algorithm>

namespace cfd {

bool init_hybrid_environment(int* argc, char*** argv, int& rank, int& size) {
    int provided = 0;
    MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return (provided >= MPI_THREAD_FUNNELED);
}

void hybrid_vorticity(const double* global_u, const double* global_v, const double* global_w,
                      const Grid3D& global_grid,
                      double* global_wx, double* global_wy, double* global_wz,
                      int omp_threads) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const size_t nx = global_grid.nx;
    const size_t ny = global_grid.ny;
    const size_t nz = global_grid.nz;
    const size_t plane_size = ny * nz;

    // Partition nx among MPI ranks
    std::vector<int> counts(size, 0);
    std::vector<int> displs(size, 0);

    const int base_nx = static_cast<int>(nx) / size;
    const int rem_nx  = static_cast<int>(nx) % size;

    int current_disp = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = base_nx + (r < rem_nx ? 1 : 0);
        displs[r] = current_disp;
        current_disp += counts[r];
    }

    const int local_nx = counts[rank];
    const int start_i  = displs[rank];

    // Local grid including ghost cells (1 on left, 1 on right if internal)
    const bool has_left_ghost  = (rank > 0);
    const bool has_right_ghost = (rank < size - 1);
    const int left_ghost_count  = has_left_ghost ? 1 : 0;
    const int right_ghost_count = has_right_ghost ? 1 : 0;
    const int local_nx_with_ghosts = local_nx + left_ghost_count + right_ghost_count;

    const size_t local_total_cells = static_cast<size_t>(local_nx_with_ghosts) * plane_size;

    std::vector<double> local_u(local_total_cells, 0.0);
    std::vector<double> local_v(local_total_cells, 0.0);
    std::vector<double> local_w(local_total_cells, 0.0);

    // Distribution vectors for MPI_Scatterv (measured in double elements)
    std::vector<int> send_counts(size, 0);
    std::vector<int> send_displs(size, 0);
    for (int r = 0; r < size; ++r) {
        send_counts[r] = counts[r] * static_cast<int>(plane_size);
        send_displs[r] = displs[r] * static_cast<int>(plane_size);
    }

    // Offset in local buffer for interior cells
    double* local_u_interior = local_u.data() + left_ghost_count * plane_size;
    double* local_v_interior = local_v.data() + left_ghost_count * plane_size;
    double* local_w_interior = local_w.data() + left_ghost_count * plane_size;

    MPI_Scatterv(global_u, send_counts.data(), send_displs.data(), MPI_DOUBLE,
                 local_u_interior, send_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(global_v, send_counts.data(), send_displs.data(), MPI_DOUBLE,
                 local_v_interior, send_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(global_w, send_counts.data(), send_displs.data(), MPI_DOUBLE,
                 local_w_interior, send_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Halo exchange for left/right ghost layers
    const int prev_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int next_rank = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    // Send left interior plane to prev_rank, receive right ghost plane from next_rank
    auto exchange_halos = [&](std::vector<double>& field) {
        double* field_interior = field.data() + left_ghost_count * plane_size;
        double* send_left = field_interior; // first interior plane
        double* recv_right = field_interior + local_nx * plane_size; // ghost plane on right

        MPI_Sendrecv(send_left, static_cast<int>(plane_size), MPI_DOUBLE, prev_rank, 101,
                     recv_right, static_cast<int>(plane_size), MPI_DOUBLE, next_rank, 101,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Send right interior plane to next_rank, receive left ghost plane from prev_rank
        double* send_right = field_interior + (local_nx - 1) * plane_size; // last interior plane
        double* recv_left = field.data(); // ghost plane on left

        MPI_Sendrecv(send_right, static_cast<int>(plane_size), MPI_DOUBLE, next_rank, 102,
                     recv_left, static_cast<int>(plane_size), MPI_DOUBLE, prev_rank, 102,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    };

    exchange_halos(local_u);
    exchange_halos(local_v);
    exchange_halos(local_w);

    // Subgrid configuration for local domain
    Grid3D local_grid(local_nx_with_ghosts, ny, nz, global_grid.dx, global_grid.dy, global_grid.dz,
                      global_grid.x0 + (start_i - left_ghost_count) * global_grid.dx,
                      global_grid.y0, global_grid.z0);

    std::vector<double> local_wx(local_total_cells, 0.0);
    std::vector<double> local_wy(local_total_cells, 0.0);
    std::vector<double> local_wz(local_total_cells, 0.0);

    // Compute vorticity using OpenMP threads on local domain
    vorticity_openmp(local_u.data(), local_v.data(), local_w.data(), local_grid,
                     local_wx.data(), local_wy.data(), local_wz.data(), nullptr,
                     2, omp_threads);

    // Extract interior result and gather back to rank 0 using MPI_Gatherv
    double* local_wx_interior = local_wx.data() + left_ghost_count * plane_size;
    double* local_wy_interior = local_wy.data() + left_ghost_count * plane_size;
    double* local_wz_interior = local_wz.data() + left_ghost_count * plane_size;

    MPI_Gatherv(local_wx_interior, send_counts[rank], MPI_DOUBLE,
                global_wx, send_counts.data(), send_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_wy_interior, send_counts[rank], MPI_DOUBLE,
                global_wy, send_counts.data(), send_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_wz_interior, send_counts[rank], MPI_DOUBLE,
                global_wz, send_counts.data(), send_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void hybrid_qcriterion(const double* global_u, const double* global_v, const double* global_w,
                       const Grid3D& global_grid,
                       double* global_q,
                       int omp_threads) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const size_t nx = global_grid.nx;
    const size_t ny = global_grid.ny;
    const size_t nz = global_grid.nz;
    const size_t plane_size = ny * nz;

    std::vector<int> counts(size, 0);
    std::vector<int> displs(size, 0);

    const int base_nx = static_cast<int>(nx) / size;
    const int rem_nx  = static_cast<int>(nx) % size;

    int current_disp = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = base_nx + (r < rem_nx ? 1 : 0);
        displs[r] = current_disp;
        current_disp += counts[r];
    }

    const int local_nx = counts[rank];
    const int start_i  = displs[rank];

    const bool has_left_ghost  = (rank > 0);
    const bool has_right_ghost = (rank < size - 1);
    const int left_ghost_count  = has_left_ghost ? 1 : 0;
    const int right_ghost_count = has_right_ghost ? 1 : 0;
    const int local_nx_with_ghosts = local_nx + left_ghost_count + right_ghost_count;

    const size_t local_total_cells = static_cast<size_t>(local_nx_with_ghosts) * plane_size;

    std::vector<double> local_u(local_total_cells, 0.0);
    std::vector<double> local_v(local_total_cells, 0.0);
    std::vector<double> local_w(local_total_cells, 0.0);

    std::vector<int> send_counts(size, 0);
    std::vector<int> send_displs(size, 0);
    for (int r = 0; r < size; ++r) {
        send_counts[r] = counts[r] * static_cast<int>(plane_size);
        send_displs[r] = displs[r] * static_cast<int>(plane_size);
    }

    double* local_u_interior = local_u.data() + left_ghost_count * plane_size;
    double* local_v_interior = local_v.data() + left_ghost_count * plane_size;
    double* local_w_interior = local_w.data() + left_ghost_count * plane_size;

    MPI_Scatterv(global_u, send_counts.data(), send_displs.data(), MPI_DOUBLE,
                 local_u_interior, send_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(global_v, send_counts.data(), send_displs.data(), MPI_DOUBLE,
                 local_v_interior, send_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(global_w, send_counts.data(), send_displs.data(), MPI_DOUBLE,
                 local_w_interior, send_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const int prev_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int next_rank = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    auto exchange_halos = [&](std::vector<double>& field) {
        double* field_interior = field.data() + left_ghost_count * plane_size;
        double* send_left = field_interior;
        double* recv_right = field_interior + local_nx * plane_size;

        MPI_Sendrecv(send_left, static_cast<int>(plane_size), MPI_DOUBLE, prev_rank, 201,
                     recv_right, static_cast<int>(plane_size), MPI_DOUBLE, next_rank, 201,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        double* send_right = field_interior + (local_nx - 1) * plane_size;
        double* recv_left = field.data();

        MPI_Sendrecv(send_right, static_cast<int>(plane_size), MPI_DOUBLE, next_rank, 202,
                     recv_left, static_cast<int>(plane_size), MPI_DOUBLE, prev_rank, 202,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    };

    exchange_halos(local_u);
    exchange_halos(local_v);
    exchange_halos(local_w);

    Grid3D local_grid(local_nx_with_ghosts, ny, nz, global_grid.dx, global_grid.dy, global_grid.dz,
                      global_grid.x0 + (start_i - left_ghost_count) * global_grid.dx,
                      global_grid.y0, global_grid.z0);

    std::vector<double> local_q(local_total_cells, 0.0);

    q_criterion_openmp(local_u.data(), local_v.data(), local_w.data(), local_grid,
                       local_q.data(), 2, omp_threads);

    double* local_q_interior = local_q.data() + left_ghost_count * plane_size;

    MPI_Gatherv(local_q_interior, send_counts[rank], MPI_DOUBLE,
                global_q, send_counts.data(), send_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

} // namespace cfd
