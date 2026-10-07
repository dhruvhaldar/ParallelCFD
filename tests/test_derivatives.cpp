#include "cfd/common.hpp"
#include "cfd/gradients.hpp"
#include "cfd/divergence.hpp"
#include "cfd/grid.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

void test_gradients_quadratic_polynomial() {
    std::cout << "  [1/4] Testing 3D central difference gradients on exact quadratic polynomial...\n";
    // phi(x,y,z) = 3x^2 + 2y^2 + 5z^2 + 4xy
    // Exact dphi/dx = 6x + 4y
    // Exact dphi/dy = 4y + 4x
    // Exact dphi/dz = 10z
    const size_t nx = 32, ny = 32, nz = 32;
    const double dx = 0.05, dy = 0.05, dz = 0.05;
    const cfd::Grid3D grid(nx, ny, nz, dx, dy, dz, -0.8, -0.8, -0.8);
    const size_t n = grid.total_cells();

    std::vector<double> phi(n);
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            for (size_t k = 0; k < nz; ++k) {
                const double x = grid.x(i);
                const double y = grid.y(j);
                const double z = grid.z(k);
                phi[grid.index(i, j, k)] = 3.0 * x * x + 2.0 * y * y + 5.0 * z * z + 4.0 * x * y;
            }
        }
    }

    std::vector<double> gx(n, 0.0), gy(n, 0.0), gz(n, 0.0);
    cfd::gradient_openmp(phi.data(), grid, gx.data(), gy.data(), gz.data(), 2, 4);

    // Verify interior cells (where central difference is exact to truncation order)
    double max_err_x = 0.0, max_err_y = 0.0, max_err_z = 0.0;
    for (size_t i = 2; i < nx - 2; ++i) {
        for (size_t j = 2; j < ny - 2; ++j) {
            for (size_t k = 2; k < nz - 2; ++k) {
                const double x = grid.x(i);
                const double y = grid.y(j);
                const double z = grid.z(k);
                const size_t idx = grid.index(i, j, k);

                const double exact_gx = 6.0 * x + 4.0 * y;
                const double exact_gy = 4.0 * y + 4.0 * x;
                const double exact_gz = 10.0 * z;

                max_err_x = std::max(max_err_x, std::abs(gx[idx] - exact_gx));
                max_err_y = std::max(max_err_y, std::abs(gy[idx] - exact_gy));
                max_err_z = std::max(max_err_z, std::abs(gz[idx] - exact_gz));
            }
        }
    }

    assert(max_err_x < 1e-12);
    assert(max_err_y < 1e-12);
    assert(max_err_z < 1e-12);
    std::cout << "    Max interior gradient error: " << max_err_x << " (Exact machine precision)\n";
}

void test_collapse_equivalence() {
    std::cout << "  [2/4] Testing exact equivalence between collapse(1), collapse(2), and collapse(3)...\n";
    const size_t nx = 32, ny = 32, nz = 32;
    const cfd::Grid3D grid(nx, ny, nz, 0.1, 0.1, 0.1);
    const size_t n = grid.total_cells();

    std::vector<double> phi(n);
    for (size_t i = 0; i < n; ++i) phi[i] = std::sin(static_cast<double>(i) * 0.1);

    std::vector<double> gx1(n, 0.0), gy1(n, 0.0), gz1(n, 0.0);
    std::vector<double> gx2(n, 0.0), gy2(n, 0.0), gz2(n, 0.0);
    std::vector<double> gx3(n, 0.0), gy3(n, 0.0), gz3(n, 0.0);

    cfd::gradient_openmp_collapse1(phi.data(), grid, gx1.data(), gy1.data(), gz1.data(), 8);
    cfd::gradient_openmp_collapse2(phi.data(), grid, gx2.data(), gy2.data(), gz2.data(), 8);
    cfd::gradient_openmp_collapse3(phi.data(), grid, gx3.data(), gy3.data(), gz3.data(), 8);

    for (size_t i = 0; i < n; ++i) {
        assert(gx1[i] == gx2[i] && gx2[i] == gx3[i]);
        assert(gy1[i] == gy2[i] && gy2[i] == gy3[i]);
        assert(gz1[i] == gz2[i] && gz2[i] == gz3[i]);
    }
}

void test_divergence_linear_field() {
    std::cout << "  [3/4] Testing divergence on linear velocity field (div = a + b + c)...\n";
    // u = 2x, v = 3y, w = 4z -> div u = 2 + 3 + 4 = 9.0 everywhere
    const size_t nx = 24, ny = 24, nz = 24;
    const cfd::Grid3D grid(nx, ny, nz, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0);
    const size_t n = grid.total_cells();

    std::vector<double> u(n), v(n), w(n), div(n, 0.0);
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            for (size_t k = 0; k < nz; ++k) {
                const size_t idx = grid.index(i, j, k);
                u[idx] = 2.0 * grid.x(i);
                v[idx] = 3.0 * grid.y(j);
                w[idx] = 4.0 * grid.z(k);
            }
        }
    }

    cfd::divergence_openmp(u.data(), v.data(), w.data(), grid, div.data(), 2, 4);

    for (size_t i = 2; i < nx - 2; ++i) {
        for (size_t j = 2; j < ny - 2; ++j) {
            for (size_t k = 2; k < nz - 2; ++k) {
                const size_t idx = grid.index(i, j, k);
                assert(std::abs(div[idx] - 9.0) < 1e-12);
            }
        }
    }
}

void test_divergence_taylor_green() {
    std::cout << "  [4/4] Testing incompressibility on 3D Taylor-Green vortex (div = 0)...\n";
    const size_t nx = 64, ny = 64, nz = 64;
    const double L = 3.141592653589793;
    const double dx = 2.0 * L / nx;
    const double dy = 2.0 * L / ny;
    const double dz = 2.0 * L / nz;
    const cfd::Grid3D grid(nx, ny, nz, dx, dy, dz, -L, -L, -L);
    const size_t n = grid.total_cells();

    cfd::Field3D field = cfd::generate_taylor_green_vortex(grid);

    std::vector<double> div_ser(n, 0.0);
    std::vector<double> div_omp(n, 0.0);

    cfd::divergence_serial(field.u.data(), field.v.data(), field.w.data(), grid, div_ser.data());
    cfd::divergence_openmp(field.u.data(), field.v.data(), field.w.data(), grid, div_omp.data(), 2, 4);

    for (size_t i = 0; i < n; ++i) {
        assert(std::abs(div_ser[i] - div_omp[i]) < 1e-14);
    }

    // Interior points should have divergence near machine epsilon
    double max_div = 0.0;
    for (size_t i = 2; i < nx - 2; ++i) {
        for (size_t j = 2; j < ny - 2; ++j) {
            for (size_t k = 2; k < nz - 2; ++k) {
                const size_t idx = grid.index(i, j, k);
                max_div = std::max(max_div, std::abs(div_omp[idx]));
            }
        }
    }
    assert(max_div < 1e-12);
    std::cout << "    Max interior divergence: " << max_div << " (Incompressible to machine precision)\n";
}

int main() {
    std::cout << "========================================================\n";
    std::cout << " Running Finite Difference Gradient & Divergence Tests\n";
    std::cout << "========================================================\n";

    test_gradients_quadratic_polynomial();
    test_collapse_equivalence();
    test_divergence_linear_field();
    test_divergence_taylor_green();

    std::cout << "\n>>> ALL DERIVATIVE TESTS PASSED SUCCESSFULLY! <<<\n";
    return 0;
}
