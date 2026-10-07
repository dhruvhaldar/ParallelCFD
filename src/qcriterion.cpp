#include "cfd/qcriterion.hpp"

namespace cfd {

namespace {

inline double diff_x(const double* phi, size_t i, size_t j, size_t k,
                     size_t nx, size_t ny, size_t nz, double inv_2dx, double inv_dx) {
    const size_t s_x = ny * nz;
    const size_t idx = (i * ny + j) * nz + k;
    if (i == 0) {
        return (nx >= 3) ? (-3.0 * phi[idx] + 4.0 * phi[idx + s_x] - phi[idx + 2 * s_x]) * inv_2dx
                         : (phi[idx + s_x] - phi[idx]) * inv_dx;
    } else if (i == nx - 1) {
        return (nx >= 3) ? (3.0 * phi[idx] - 4.0 * phi[idx - s_x] + phi[idx - 2 * s_x]) * inv_2dx
                         : (phi[idx] - phi[idx - s_x]) * inv_dx;
    } else {
        return (phi[idx + s_x] - phi[idx - s_x]) * inv_2dx;
    }
}

inline double diff_y(const double* phi, size_t i, size_t j, size_t k,
                     size_t /*nx*/, size_t ny, size_t nz, double inv_2dy, double inv_dy) {
    const size_t s_y = nz;
    const size_t idx = (i * ny + j) * nz + k;
    if (j == 0) {
        return (ny >= 3) ? (-3.0 * phi[idx] + 4.0 * phi[idx + s_y] - phi[idx + 2 * s_y]) * inv_2dy
                         : (phi[idx + s_y] - phi[idx]) * inv_dy;
    } else if (j == ny - 1) {
        return (ny >= 3) ? (3.0 * phi[idx] - 4.0 * phi[idx - s_y] + phi[idx - 2 * s_y]) * inv_2dy
                         : (phi[idx] - phi[idx - s_y]) * inv_dy;
    } else {
        return (phi[idx + s_y] - phi[idx - s_y]) * inv_2dy;
    }
}

inline double diff_z(const double* phi, size_t i, size_t j, size_t k,
                     size_t /*nx*/, size_t ny, size_t nz, double inv_2dz, double inv_dz) {
    const size_t idx = (i * ny + j) * nz + k;
    if (k == 0) {
        return (nz >= 3) ? (-3.0 * phi[idx] + 4.0 * phi[idx + 1] - phi[idx + 2]) * inv_2dz
                         : (phi[idx + 1] - phi[idx]) * inv_dz;
    } else if (k == nz - 1) {
        return (nz >= 3) ? (3.0 * phi[idx] - 4.0 * phi[idx - 1] + phi[idx - 2]) * inv_2dz
                         : (phi[idx] - phi[idx - 1]) * inv_dz;
    } else {
        return (phi[idx + 1] - phi[idx - 1]) * inv_2dz;
    }
}

inline double compute_cell_q(const double* u, const double* v, const double* w,
                            size_t i, size_t j, size_t k,
                            size_t nx, size_t ny, size_t nz,
                            double inv_2dx, double inv_dx,
                            double inv_2dy, double inv_dy,
                            double inv_2dz, double inv_dz) {
    const double du_dx = diff_x(u, i, j, k, nx, ny, nz, inv_2dx, inv_dx);
    const double du_dy = diff_y(u, i, j, k, nx, ny, nz, inv_2dy, inv_dy);
    const double du_dz = diff_z(u, i, j, k, nx, ny, nz, inv_2dz, inv_dz);

    const double dv_dx = diff_x(v, i, j, k, nx, ny, nz, inv_2dx, inv_dx);
    const double dv_dy = diff_y(v, i, j, k, nx, ny, nz, inv_2dy, inv_dy);
    const double dv_dz = diff_z(v, i, j, k, nx, ny, nz, inv_2dz, inv_dz);

    const double dw_dx = diff_x(w, i, j, k, nx, ny, nz, inv_2dx, inv_dx);
    const double dw_dy = diff_y(w, i, j, k, nx, ny, nz, inv_2dy, inv_dy);
    const double dw_dz = diff_z(w, i, j, k, nx, ny, nz, inv_2dz, inv_dz);

    // Symmetric strain tensor S
    const double S11 = du_dx;
    const double S22 = dv_dy;
    const double S33 = dw_dz;
    const double S12 = 0.5 * (du_dy + dv_dx);
    const double S13 = 0.5 * (du_dz + dw_dx);
    const double S23 = 0.5 * (dv_dz + dw_dy);

    const double norm_S_sq = S11 * S11 + S22 * S22 + S33 * S33 +
                            2.0 * (S12 * S12 + S13 * S13 + S23 * S23);

    // Antisymmetric rotation tensor Omega
    const double O12 = 0.5 * (du_dy - dv_dx);
    const double O13 = 0.5 * (du_dz - dw_dx);
    const double O23 = 0.5 * (dv_dz - dw_dy);

    const double norm_O_sq = 2.0 * (O12 * O12 + O13 * O13 + O23 * O23);

    return 0.5 * (norm_O_sq - norm_S_sq);
}

} // anonymous namespace

void q_criterion_serial(const double* u, const double* v, const double* w,
                        const Grid3D& grid, double* q_field) {
    const size_t nx = grid.nx;
    const size_t ny = grid.ny;
    const size_t nz = grid.nz;
    const double inv_2dx = 1.0 / (2.0 * grid.dx);
    const double inv_dx  = 1.0 / grid.dx;
    const double inv_2dy = 1.0 / (2.0 * grid.dy);
    const double inv_dy  = 1.0 / grid.dy;
    const double inv_2dz = 1.0 / (2.0 * grid.dz);
    const double inv_dz  = 1.0 / grid.dz;

    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            for (size_t k = 0; k < nz; ++k) {
                const size_t idx = (i * ny + j) * nz + k;
                q_field[idx] = compute_cell_q(u, v, w, i, j, k, nx, ny, nz,
                                              inv_2dx, inv_dx, inv_2dy, inv_dy, inv_2dz, inv_dz);
            }
        }
    }
}

void q_criterion_openmp(const double* u, const double* v, const double* w,
                        const Grid3D& grid, double* q_field,
                        int collapse_level, int num_threads) {
    const size_t nx = grid.nx;
    const size_t ny = grid.ny;
    const size_t nz = grid.nz;
    const double inv_2dx = 1.0 / (2.0 * grid.dx);
    const double inv_dx  = 1.0 / grid.dx;
    const double inv_2dy = 1.0 / (2.0 * grid.dy);
    const double inv_dy  = 1.0 / grid.dy;
    const double inv_2dz = 1.0 / (2.0 * grid.dz);
    const double inv_dz  = 1.0 / grid.dz;
    const int threads = (num_threads > 0) ? num_threads : omp_get_max_threads();

    if (collapse_level == 3) {
        #pragma omp parallel for collapse(3) num_threads(threads) schedule(static)
        for (size_t i = 0; i < nx; ++i) {
            for (size_t j = 0; j < ny; ++j) {
                for (size_t k = 0; k < nz; ++k) {
                    const size_t idx = (i * ny + j) * nz + k;
                    q_field[idx] = compute_cell_q(u, v, w, i, j, k, nx, ny, nz,
                                                  inv_2dx, inv_dx, inv_2dy, inv_dy, inv_2dz, inv_dz);
                }
            }
        }
    } else {
        #pragma omp parallel for collapse(2) num_threads(threads) schedule(static)
        for (size_t i = 0; i < nx; ++i) {
            for (size_t j = 0; j < ny; ++j) {
                #pragma omp simd
                for (size_t k = 0; k < nz; ++k) {
                    const size_t idx = (i * ny + j) * nz + k;
                    q_field[idx] = compute_cell_q(u, v, w, i, j, k, nx, ny, nz,
                                                  inv_2dx, inv_dx, inv_2dy, inv_dy, inv_2dz, inv_dz);
                }
            }
        }
    }
}

} // namespace cfd
