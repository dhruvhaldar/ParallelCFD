#pragma once

#include "cfd/common.hpp"
#include <vector>
#include <cmath>

namespace cfd {

struct Field3D {
    Grid3D grid;
    std::vector<double> u;
    std::vector<double> v;
    std::vector<double> w;
    std::vector<double> pressure;

    explicit Field3D(const Grid3D& g)
        : grid(g),
          u(g.total_cells(), 0.0),
          v(g.total_cells(), 0.0),
          w(g.total_cells(), 0.0),
          pressure(g.total_cells(), 0.0) {}
};

// Generate 3D Taylor-Green Vortex field
// u = U0 * sin(x/L) * cos(y/L) * cos(z/L)
// v = -U0 * cos(x/L) * sin(y/L) * cos(z/L)
// w = 0
// p = p0 + (rho*U0^2/16) * (cos(2x/L) + cos(2y/L)) * (cos(2z/L) + 2)
Field3D generate_taylor_green_vortex(const Grid3D& grid, double U0 = 1.0, double L = 1.0, double rho = 1.0, double p0 = 100.0);

// Generate Solid-Body Rotation field
// u = -Omega * y
// v = Omega * x
// w = 0
// p = p0 + 0.5 * rho * Omega^2 * (x^2 + y^2)
Field3D generate_solid_body_rotation(const Grid3D& grid, double Omega = 2.0, double rho = 1.0, double p0 = 100.0);

// Analytical solution helpers for verification
struct AnalyticalTGV {
    static double div(double, double, double, double = 1.0, double = 1.0) noexcept { return 0.0; }
    static void vorticity(double x, double y, double z, double& wx, double& wy, double& wz, double U0 = 1.0, double L = 1.0) noexcept {
        const double invL = 1.0 / L;
        wx = -U0 * invL * std::cos(x * invL) * std::sin(y * invL) * std::sin(z * invL);
        wy = -U0 * invL * std::sin(x * invL) * std::cos(y * invL) * std::sin(z * invL);
        wz =  2.0 * U0 * invL * std::sin(x * invL) * std::sin(y * invL) * std::cos(z * invL);
    }
};

} // namespace cfd
