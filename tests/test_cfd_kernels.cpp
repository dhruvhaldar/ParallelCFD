#include "cfd/common.hpp"
#include "cfd/grid.hpp"
#include "cfd/velocity.hpp"
#include "cfd/reductions.hpp"
#include "cfd/gradients.hpp"
#include "cfd/divergence.hpp"
#include "cfd/vorticity.hpp"
#include "cfd/qcriterion.hpp"

#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

void test_velocity_magnitude() {
    std::cout << "[TEST] Running test_velocity_magnitude...\n";
    const size_t n = 100000;
    std::vector<double> u(n, 3.0);
    std::vector<double> v(n, 4.0);
    std::vector<double> w(n, 0.0);
    std::vector<double> mag_ser(n, 0.0);
    std::vector<double> mag_omp(n, 0.0);
    std::vector<double> mag_simd(n, 0.0);

    cfd::velocity_magnitude_serial(u.data(), v.data(), w.data(), mag_ser.data(), n);
    cfd::velocity_magnitude_openmp(u.data(), v.data(), w.data(), mag_omp.data(), n, 4);
    cfd::velocity_magnitude_simd(u.data(), v.data(), w.data(), mag_simd.data(), n, 4);

    for (size_t i = 0; i < n; ++i) {
        assert(std::abs(mag_ser[i] - 5.0) < 1e-12);
        assert(std::abs(mag_omp[i] - 5.0) < 1e-12);
        assert(std::abs(mag_simd[i] - 5.0) < 1e-12);
    }
    std::cout << "  -> PASSED: Serial, OpenMP, and SIMD velocity magnitude match exact analytical 5.0!\n";
}

void test_reductions() {
    std::cout << "[TEST] Running test_reductions...\n";
    const size_t n = 10000;
    std::vector<double> data(n);
    for (size_t i = 0; i < n; ++i) {
        data[i] = static_cast<double>(i + 1);
    }

    auto stats_ser = cfd::field_stats_serial(data.data(), n);
    auto stats_omp = cfd::field_stats_openmp(data.data(), n, 4);

    assert(std::abs(stats_ser.min_val - 1.0) < 1e-12);
    assert(std::abs(stats_omp.min_val - 1.0) < 1e-12);
    assert(std::abs(stats_ser.max_val - static_cast<double>(n)) < 1e-12);
    assert(std::abs(stats_omp.max_val - static_cast<double>(n)) < 1e-12);

    const double expected_sum = static_cast<double>(n) * (n + 1) / 2.0;
    assert(std::abs(stats_ser.sum_val - expected_sum) < 1e-6);
    assert(std::abs(stats_omp.sum_val - expected_sum) < 1e-6);
    assert(std::abs(stats_omp.mean_val - stats_ser.mean_val) < 1e-12);
    assert(std::abs(stats_omp.rms_val - stats_ser.rms_val) < 1e-12);

    std::cout << "  -> PASSED: Reductions min/max/mean/RMS match expected analytical values!\n";
}

void test_analytical_solid_body() {
    std::cout << "[TEST] Running test_analytical_solid_body...\n";
    const size_t nx = 32, ny = 32, nz = 32;
    const double dx = 0.1, dy = 0.1, dz = 0.1;
    const double Omega = 3.5;
    const cfd::Grid3D grid(nx, ny, nz, dx, dy, dz, -1.6, -1.6, -1.6);
    const size_t n = grid.total_cells();

    cfd::Field3D field = cfd::generate_solid_body_rotation(grid, Omega);

    // 1. Test Divergence (should be identically 0)
    std::vector<double> div_ser(n, 0.0);
    std::vector<double> div_omp(n, 0.0);
    cfd::divergence_serial(field.u.data(), field.v.data(), field.w.data(), grid, div_ser.data());
    cfd::divergence_openmp(field.u.data(), field.v.data(), field.w.data(), grid, div_omp.data(), 2, 4);

    for (size_t i = 0; i < n; ++i) {
        assert(std::abs(div_ser[i]) < 1e-10);
        assert(std::abs(div_omp[i]) < 1e-10);
        assert(std::abs(div_ser[i] - div_omp[i]) < 1e-12);
    }
    std::cout << "  -> PASSED: Solid-body divergence identically zero!\n";

    // 2. Test Vorticity (omega_z should be exactly 2 * Omega = 7.0, omega_x = 0, omega_y = 0)
    std::vector<double> wx(n, 0.0), wy(n, 0.0), wz(n, 0.0), wmag(n, 0.0);
    cfd::vorticity_openmp(field.u.data(), field.v.data(), field.w.data(), grid,
                          wx.data(), wy.data(), wz.data(), wmag.data(), 2, 4);

    // Check interior cells (avoid boundary discretization artifacts)
    for (size_t i = 2; i < nx - 2; ++i) {
        for (size_t j = 2; j < ny - 2; ++j) {
            for (size_t k = 2; k < nz - 2; ++k) {
                const size_t idx = grid.index(i, j, k);
                assert(std::abs(wx[idx]) < 1e-10);
                assert(std::abs(wy[idx]) < 1e-10);
                assert(std::abs(wz[idx] - (2.0 * Omega)) < 1e-10);
                assert(std::abs(wmag[idx] - (2.0 * Omega)) < 1e-10);
            }
        }
    }
    std::cout << "  -> PASSED: Solid-body vorticity matches exact analytical curl (2*Omega = 7.0)!\n";

    // 3. Test Q-Criterion (Q = Omega^2 = 12.25 everywhere)
    std::vector<double> q_ser(n, 0.0);
    std::vector<double> q_omp(n, 0.0);
    cfd::q_criterion_serial(field.u.data(), field.v.data(), field.w.data(), grid, q_ser.data());
    cfd::q_criterion_openmp(field.u.data(), field.v.data(), field.w.data(), grid, q_omp.data(), 2, 4);

    const double expected_q = Omega * Omega;
    for (size_t i = 2; i < nx - 2; ++i) {
        for (size_t j = 2; j < ny - 2; ++j) {
            for (size_t k = 2; k < nz - 2; ++k) {
                const size_t idx = grid.index(i, j, k);
                assert(std::abs(q_ser[idx] - expected_q) < 1e-10);
                assert(std::abs(q_omp[idx] - expected_q) < 1e-10);
                assert(std::abs(q_ser[idx] - q_omp[idx]) < 1e-12);
            }
        }
    }
    std::cout << "  -> PASSED: Solid-body Q-criterion matches exact analytical Omega^2!\n";
}

int main() {
    std::cout << "========================================================\n";
    std::cout << " ParallelCFD C++ Unit Tests\n";
    std::cout << "========================================================\n";

    test_velocity_magnitude();
    test_reductions();
    test_analytical_solid_body();

    std::cout << "\nALL C++ UNIT TESTS PASSED SUCCESSFULLY!\n";
    return 0;
}
