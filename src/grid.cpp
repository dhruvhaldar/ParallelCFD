#include "cfd/grid.hpp"
#include <cmath>

namespace cfd {

Field3D generate_taylor_green_vortex(const Grid3D& grid, double U0, double L, double rho, double p0) {
    Field3D field(grid);
    const size_t nx = grid.nx;
    const size_t ny = grid.ny;
    const size_t nz = grid.nz;
    const double invL = 1.0 / L;
    const double p_scale = rho * U0 * U0 / 16.0;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            const double x = grid.x(i);
            const double y = grid.y(j);
            const double sin_x = std::sin(x * invL);
            const double cos_x = std::cos(x * invL);
            const double sin_y = std::sin(y * invL);
            const double cos_y = std::cos(y * invL);
            const double cos_2x = std::cos(2.0 * x * invL);
            const double cos_2y = std::cos(2.0 * y * invL);

            #pragma omp simd
            for (size_t k = 0; k < nz; ++k) {
                const double z = grid.z(k);
                const double cos_z = std::cos(z * invL);
                const double cos_2z = std::cos(2.0 * z * invL);
                const size_t idx = grid.index(i, j, k);

                field.u[idx] = U0 * sin_x * cos_y * cos_z;
                field.v[idx] = -U0 * cos_x * sin_y * cos_z;
                field.w[idx] = 0.0;
                field.pressure[idx] = p0 + p_scale * (cos_2x + cos_2y) * (cos_2z + 2.0);
            }
        }
    }

    return field;
}

Field3D generate_solid_body_rotation(const Grid3D& grid, double Omega, double rho, double p0) {
    Field3D field(grid);
    const size_t nx = grid.nx;
    const size_t ny = grid.ny;
    const size_t nz = grid.nz;
    const double p_scale = 0.5 * rho * Omega * Omega;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            const double x = grid.x(i);
            const double y = grid.y(j);
            const double r2 = x * x + y * y;

            #pragma omp simd
            for (size_t k = 0; k < nz; ++k) {
                const size_t idx = grid.index(i, j, k);
                field.u[idx] = -Omega * y;
                field.v[idx] = Omega * x;
                field.w[idx] = 0.0;
                field.pressure[idx] = p0 + p_scale * r2;
            }
        }
    }

    return field;
}

} // namespace cfd
