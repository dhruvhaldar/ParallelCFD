#include "cfd/common.hpp"
#include "cfd/vorticity.hpp"
#include "cfd/qcriterion.hpp"
#include "cfd/grid.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

void test_solid_body_rotation() {
    std::cout << "  [1/4] Testing Solid-Body Rotation (curl u = 2*Omega, Q = Omega^2)...\n";
    const size_t nx = 32, ny = 32, nz = 32;
    const double Omega = 4.2;
    const cfd::Grid3D grid(nx, ny, nz, 0.05, 0.05, 0.05, -0.8, -0.8, -0.8);
    const size_t n = grid.total_cells();

    cfd::Field3D field = cfd::generate_solid_body_rotation(grid, Omega);

    std::vector<double> wx(n), wy(n), wz(n), wmag(n), q(n);
    cfd::vorticity_openmp(field.u.data(), field.v.data(), field.w.data(), grid,
                          wx.data(), wy.data(), wz.data(), wmag.data(), 2, 4);
    cfd::q_criterion_openmp(field.u.data(), field.v.data(), field.w.data(), grid, q.data(), 2, 4);

    for (size_t i = 2; i < nx - 2; ++i) {
        for (size_t j = 2; j < ny - 2; ++j) {
            for (size_t k = 2; k < nz - 2; ++k) {
                const size_t idx = grid.index(i, j, k);
                assert(std::abs(wx[idx]) < 1e-12);
                assert(std::abs(wy[idx]) < 1e-12);
                assert(std::abs(wz[idx] - (2.0 * Omega)) < 1e-12);
                assert(std::abs(wmag[idx] - (2.0 * Omega)) < 1e-12);
                assert(std::abs(q[idx] - (Omega * Omega)) < 1e-12);
            }
        }
    }
}

void test_pure_shear_couette_flow() {
    std::cout << "  [2/4] Testing Pure Shear Couette Flow (vorticity != 0, but Q == 0 identically)...\n";
    // u = gamma * y, v = 0, w = 0
    // Curl u = (0, 0, -gamma)
    // S_12 = gamma/2, Omega_12 = gamma/2 -> ||Omega||^2 == ||S||^2 -> Q = 0 everywhere
    const size_t nx = 24, ny = 24, nz = 24;
    const double gamma_dot = 5.0;
    const cfd::Grid3D grid(nx, ny, nz, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0);
    const size_t n = grid.total_cells();

    std::vector<double> u(n), v(n, 0.0), w(n, 0.0);
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            for (size_t k = 0; k < nz; ++k) {
                u[grid.index(i, j, k)] = gamma_dot * grid.y(j);
            }
        }
    }

    std::vector<double> wx(n), wy(n), wz(n), q(n);
    cfd::vorticity_openmp(u.data(), v.data(), w.data(), grid, wx.data(), wy.data(), wz.data(), nullptr, 2, 4);
    cfd::q_criterion_openmp(u.data(), v.data(), w.data(), grid, q.data(), 2, 4);

    for (size_t i = 2; i < nx - 2; ++i) {
        for (size_t j = 2; j < ny - 2; ++j) {
            for (size_t k = 2; k < nz - 2; ++k) {
                const size_t idx = grid.index(i, j, k);
                assert(std::abs(wx[idx]) < 1e-12);
                assert(std::abs(wy[idx]) < 1e-12);
                assert(std::abs(wz[idx] - (-gamma_dot)) < 1e-12);
                // Q-criterion MUST be zero in pure shear flow!
                assert(std::abs(q[idx]) < 1e-12);
            }
        }
    }
}

void test_tgv_analytical_vorticity() {
    std::cout << "  [3/4] Testing Taylor-Green Vortex vorticity against exact trigonometric solution...\n";
    const size_t nx = 32, ny = 32, nz = 32;
    const double L = 1.0;
    const double dx = 0.05, dy = 0.05, dz = 0.05;
    const cfd::Grid3D grid(nx, ny, nz, dx, dy, dz, 0.0, 0.0, 0.0);
    const size_t n = grid.total_cells();

    cfd::Field3D field = cfd::generate_taylor_green_vortex(grid, 1.0, L);
    std::vector<double> wx(n), wy(n), wz(n);
    cfd::vorticity_openmp(field.u.data(), field.v.data(), field.w.data(), grid,
                          wx.data(), wy.data(), wz.data(), nullptr, 2, 4);

    // Central difference stencil error is O(dx^2)
    double max_err = 0.0;
    for (size_t i = 2; i < nx - 2; ++i) {
        for (size_t j = 2; j < ny - 2; ++j) {
            for (size_t k = 2; k < nz - 2; ++k) {
                const double x = grid.x(i);
                const double y = grid.y(j);
                const double z = grid.z(k);
                const size_t idx = grid.index(i, j, k);

                double exact_wx, exact_wy, exact_wz;
                cfd::AnalyticalTGV::vorticity(x, y, z, exact_wx, exact_wy, exact_wz, 1.0, L);

                max_err = std::max(max_err, std::abs(wx[idx] - exact_wx));
                max_err = std::max(max_err, std::abs(wy[idx] - exact_wy));
                max_err = std::max(max_err, std::abs(wz[idx] - exact_wz));
            }
        }
    }
    // Truncation error for second order difference: O(dx^2) ~ 0.05^2 / 6 ~ 0.0004
    assert(max_err < 0.002);
    std::cout << "    Max TGV vorticity error: " << max_err << " (Within O(dx^2) discretization bound)\n";
}

void test_serial_vs_openmp_equivalence() {
    std::cout << "  [4/4] Testing serial vs OpenMP numerical equivalence for Q-criterion...\n";
    const size_t nx = 32, ny = 32, nz = 32;
    const cfd::Grid3D grid(nx, ny, nz, 0.1, 0.1, 0.1);
    const size_t n = grid.total_cells();
    cfd::Field3D field = cfd::generate_taylor_green_vortex(grid);

    std::vector<double> q_ser(n, 0.0), q_omp(n, 0.0);
    cfd::q_criterion_serial(field.u.data(), field.v.data(), field.w.data(), grid, q_ser.data());

    const int threads_sweep[] = {1, 2, 4, 8, 16};
    for (int t : threads_sweep) {
        cfd::q_criterion_openmp(field.u.data(), field.v.data(), field.w.data(), grid, q_omp.data(), 2, t);
        for (size_t i = 0; i < n; ++i) {
            assert(std::abs(q_ser[i] - q_omp[i]) < 1e-14);
        }
    }
}

int main() {
    std::cout << "========================================================\n";
    std::cout << " Running Vorticity & Q-Criterion Verification Tests\n";
    std::cout << "========================================================\n";

    test_solid_body_rotation();
    test_pure_shear_couette_flow();
    test_tgv_analytical_vorticity();
    test_serial_vs_openmp_equivalence();

    std::cout << "\n>>> ALL VORTICITY & Q-CRITERION TESTS PASSED! <<<\n";
    return 0;
}
