#include "cfd/divergence.hpp"

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

} // anonymous namespace

void divergence_serial(const double* u, const double* v, const double* w,
                       const Grid3D& grid, double* div) {
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
                const double du_dx = diff_x(u, i, j, k, nx, ny, nz, inv_2dx, inv_dx);
                const double dv_dy = diff_y(v, i, j, k, nx, ny, nz, inv_2dy, inv_dy);
                const double dw_dz = diff_z(w, i, j, k, nx, ny, nz, inv_2dz, inv_dz);
                div[idx] = du_dx + dv_dy + dw_dz;
            }
        }
    }
}

void divergence_openmp(const double* u, const double* v, const double* w,
                       const Grid3D& grid, double* div,
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
                    const double du_dx = diff_x(u, i, j, k, nx, ny, nz, inv_2dx, inv_dx);
                    const double dv_dy = diff_y(v, i, j, k, nx, ny, nz, inv_2dy, inv_dy);
                    const double dw_dz = diff_z(w, i, j, k, nx, ny, nz, inv_2dz, inv_dz);
                    div[idx] = du_dx + dv_dy + dw_dz;
                }
            }
        }
    } else if (collapse_level == 1) {
        #pragma omp parallel for num_threads(threads) schedule(static)
        for (size_t i = 0; i < nx; ++i) {
            for (size_t j = 0; j < ny; ++j) {
                #pragma omp simd
                for (size_t k = 0; k < nz; ++k) {
                    const size_t idx = (i * ny + j) * nz + k;
                    const double du_dx = diff_x(u, i, j, k, nx, ny, nz, inv_2dx, inv_dx);
                    const double dv_dy = diff_y(v, i, j, k, nx, ny, nz, inv_2dy, inv_dy);
                    const double dw_dz = diff_z(w, i, j, k, nx, ny, nz, inv_2dz, inv_dz);
                    div[idx] = du_dx + dv_dy + dw_dz;
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
                    const double du_dx = diff_x(u, i, j, k, nx, ny, nz, inv_2dx, inv_dx);
                    const double dv_dy = diff_y(v, i, j, k, nx, ny, nz, inv_2dy, inv_dy);
                    const double dw_dz = diff_z(w, i, j, k, nx, ny, nz, inv_2dz, inv_dz);
                    div[idx] = du_dx + dv_dy + dw_dz;
                }
            }
        }
    }
}

} // namespace cfd
