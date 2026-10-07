#include "cfd/gradients.hpp"
#include <algorithm>

namespace cfd {

namespace {

inline double diff_x(const double* phi, size_t i, size_t j, size_t k,
                     size_t nx, size_t ny, size_t nz, double inv_2dx, double inv_dx) {
    const size_t s_x = ny * nz;
    const size_t idx = (i * ny + j) * nz + k;
    if (i == 0) {
        if (nx >= 3) {
            return (-3.0 * phi[idx] + 4.0 * phi[idx + s_x] - phi[idx + 2 * s_x]) * inv_2dx;
        } else {
            return (phi[idx + s_x] - phi[idx]) * inv_dx;
        }
    } else if (i == nx - 1) {
        if (nx >= 3) {
            return (3.0 * phi[idx] - 4.0 * phi[idx - s_x] + phi[idx - 2 * s_x]) * inv_2dx;
        } else {
            return (phi[idx] - phi[idx - s_x]) * inv_dx;
        }
    } else {
        return (phi[idx + s_x] - phi[idx - s_x]) * inv_2dx;
    }
}

inline double diff_y(const double* phi, size_t i, size_t j, size_t k,
                     size_t /*nx*/, size_t ny, size_t nz, double inv_2dy, double inv_dy) {
    const size_t s_y = nz;
    const size_t idx = (i * ny + j) * nz + k;
    if (j == 0) {
        if (ny >= 3) {
            return (-3.0 * phi[idx] + 4.0 * phi[idx + s_y] - phi[idx + 2 * s_y]) * inv_2dy;
        } else {
            return (phi[idx + s_y] - phi[idx]) * inv_dy;
        }
    } else if (j == ny - 1) {
        if (ny >= 3) {
            return (3.0 * phi[idx] - 4.0 * phi[idx - s_y] + phi[idx - 2 * s_y]) * inv_2dy;
        } else {
            return (phi[idx] - phi[idx - s_y]) * inv_dy;
        }
    } else {
        return (phi[idx + s_y] - phi[idx - s_y]) * inv_2dy;
    }
}

inline double diff_z(const double* phi, size_t i, size_t j, size_t k,
                     size_t /*nx*/, size_t ny, size_t nz, double inv_2dz, double inv_dz) {
    const size_t idx = (i * ny + j) * nz + k;
    if (k == 0) {
        if (nz >= 3) {
            return (-3.0 * phi[idx] + 4.0 * phi[idx + 1] - phi[idx + 2]) * inv_2dz;
        } else {
            return (phi[idx + 1] - phi[idx]) * inv_dz;
        }
    } else if (k == nz - 1) {
        if (nz >= 3) {
            return (3.0 * phi[idx] - 4.0 * phi[idx - 1] + phi[idx - 2]) * inv_2dz;
        } else {
            return (phi[idx] - phi[idx - 1]) * inv_dz;
        }
    } else {
        return (phi[idx + 1] - phi[idx - 1]) * inv_2dz;
    }
}

} // anonymous namespace

void gradient_serial(const double* phi, const Grid3D& grid,
                     double* grad_x, double* grad_y, double* grad_z) {
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
                if (grad_x) grad_x[idx] = diff_x(phi, i, j, k, nx, ny, nz, inv_2dx, inv_dx);
                if (grad_y) grad_y[idx] = diff_y(phi, i, j, k, nx, ny, nz, inv_2dy, inv_dy);
                if (grad_z) grad_z[idx] = diff_z(phi, i, j, k, nx, ny, nz, inv_2dz, inv_dz);
            }
        }
    }
}

void gradient_openmp_collapse1(const double* phi, const Grid3D& grid,
                              double* grad_x, double* grad_y, double* grad_z,
                              int num_threads) {
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

    #pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            #pragma omp simd
            for (size_t k = 0; k < nz; ++k) {
                const size_t idx = (i * ny + j) * nz + k;
                if (grad_x) grad_x[idx] = diff_x(phi, i, j, k, nx, ny, nz, inv_2dx, inv_dx);
                if (grad_y) grad_y[idx] = diff_y(phi, i, j, k, nx, ny, nz, inv_2dy, inv_dy);
                if (grad_z) grad_z[idx] = diff_z(phi, i, j, k, nx, ny, nz, inv_2dz, inv_dz);
            }
        }
    }
}

void gradient_openmp_collapse2(const double* phi, const Grid3D& grid,
                              double* grad_x, double* grad_y, double* grad_z,
                              int num_threads) {
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

    #pragma omp parallel for collapse(2) num_threads(threads) schedule(static)
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            #pragma omp simd
            for (size_t k = 0; k < nz; ++k) {
                const size_t idx = (i * ny + j) * nz + k;
                if (grad_x) grad_x[idx] = diff_x(phi, i, j, k, nx, ny, nz, inv_2dx, inv_dx);
                if (grad_y) grad_y[idx] = diff_y(phi, i, j, k, nx, ny, nz, inv_2dy, inv_dy);
                if (grad_z) grad_z[idx] = diff_z(phi, i, j, k, nx, ny, nz, inv_2dz, inv_dz);
            }
        }
    }
}

void gradient_openmp_collapse3(const double* phi, const Grid3D& grid,
                              double* grad_x, double* grad_y, double* grad_z,
                              int num_threads) {
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

    #pragma omp parallel for collapse(3) num_threads(threads) schedule(static)
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            for (size_t k = 0; k < nz; ++k) {
                const size_t idx = (i * ny + j) * nz + k;
                if (grad_x) grad_x[idx] = diff_x(phi, i, j, k, nx, ny, nz, inv_2dx, inv_dx);
                if (grad_y) grad_y[idx] = diff_y(phi, i, j, k, nx, ny, nz, inv_2dy, inv_dy);
                if (grad_z) grad_z[idx] = diff_z(phi, i, j, k, nx, ny, nz, inv_2dz, inv_dz);
            }
        }
    }
}

void gradient_openmp(const double* phi, const Grid3D& grid,
                     double* grad_x, double* grad_y, double* grad_z,
                     int collapse_level, int num_threads) {
    if (collapse_level == 1) {
        gradient_openmp_collapse1(phi, grid, grad_x, grad_y, grad_z, num_threads);
    } else if (collapse_level == 3) {
        gradient_openmp_collapse3(phi, grid, grad_x, grad_y, grad_z, num_threads);
    } else {
        gradient_openmp_collapse2(phi, grid, grad_x, grad_y, grad_z, num_threads);
    }
}

} // namespace cfd
