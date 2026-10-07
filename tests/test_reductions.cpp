#include "cfd/common.hpp"
#include "cfd/reductions.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>

void test_field_stats_arithmetic_series() {
    std::cout << "  [1/4] Testing field reductions against exact arithmetic series...\n";
    const size_t n = 100000;
    std::vector<double> data(n);
    for (size_t i = 0; i < n; ++i) {
        data[i] = static_cast<double>(i + 1);
    }

    const double expected_min = 1.0;
    const double expected_max = static_cast<double>(n);
    const double expected_sum = static_cast<double>(n) * (n + 1) / 2.0;
    const double expected_mean = (n + 1) / 2.0;
    // Sum of squares: n(n+1)(2n+1)/6
    const double expected_sum_sq = static_cast<double>(n) * (n + 1) * (2.0 * n + 1) / 6.0;
    const double expected_rms = std::sqrt(expected_sum_sq / static_cast<double>(n));

    auto stats_ser = cfd::field_stats_serial(data.data(), n);
    assert(stats_ser.min_val == expected_min);
    assert(stats_ser.max_val == expected_max);
    assert(std::abs(stats_ser.sum_val - expected_sum) < 1e-4);
    assert(std::abs(stats_ser.mean_val - expected_mean) < 1e-6);
    assert(std::abs(stats_ser.rms_val - expected_rms) < 1e-6);

    const int threads_sweep[] = {1, 2, 4, 8, 16, 32};
    for (int t : threads_sweep) {
        auto stats_omp = cfd::field_stats_openmp(data.data(), n, t);
        assert(stats_omp.min_val == expected_min);
        assert(stats_omp.max_val == expected_max);
        assert(std::abs(stats_omp.sum_val - expected_sum) < 1e-4);
        assert(std::abs(stats_omp.mean_val - expected_mean) < 1e-6);
        assert(std::abs(stats_omp.rms_val - expected_rms) < 1e-6);
    }
}

void test_min_max_extreme_values() {
    std::cout << "  [2/4] Testing min and max reduction with extreme values and negative numbers...\n";
    const size_t n = 50000;
    std::vector<double> data(n, 0.0);
    // Insert outliers
    data[13] = -999999.5;
    data[25000] = 888888.75;
    data[49999] = -12345.0;

    auto stats = cfd::field_stats_openmp(data.data(), n, 8);
    assert(stats.min_val == -999999.5);
    assert(stats.max_val == 888888.75);
}

void test_kinetic_energy() {
    std::cout << "  [3/4] Testing kinetic energy K = 0.5 * sum(u^2 + v^2 + w^2)...\n";
    const size_t n = 40000;
    // u = 1.0, v = 2.0, w = 3.0 -> u^2 + v^2 + w^2 = 1 + 4 + 9 = 14.0
    // Total K = 0.5 * n * 14 = 7 * n
    std::vector<double> u(n, 1.0), v(n, 2.0), w(n, 3.0);
    const double expected_k = 7.0 * static_cast<double>(n);

    double k_ser = cfd::kinetic_energy_serial(u.data(), v.data(), w.data(), n);
    assert(std::abs(k_ser - expected_k) < 1e-9);

    const int threads_sweep[] = {1, 2, 4, 8, 16};
    for (int t : threads_sweep) {
        double k_omp = cfd::kinetic_energy_openmp(u.data(), v.data(), w.data(), n, t);
        assert(std::abs(k_omp - expected_k) < 1e-9);
    }
}

void test_fp_nonassociativity_bounds() {
    std::cout << "  [4/4] Testing IEEE-754 floating point non-associativity bounds...\n";
    auto res = cfd::test_floating_point_nonassociativity(1000000, 8);
    // Differences should be small (order of machine precision accumulated over iterations)
    assert(res.rel_diff < 1e-10);
    std::cout << "    Verified: Serial = " << res.serial_sum
              << ", Parallel = " << res.parallel_sum
              << ", Relative Diff = " << res.rel_diff << "\n";
}

int main() {
    std::cout << "========================================================\n";
    std::cout << " Running Reduction Kernels & Precision Tests\n";
    std::cout << "========================================================\n";

    test_field_stats_arithmetic_series();
    test_min_max_extreme_values();
    test_kinetic_energy();
    test_fp_nonassociativity_bounds();

    std::cout << "\n>>> ALL REDUCTION TESTS PASSED SUCCESSFULLY! <<<\n";
    return 0;
}
