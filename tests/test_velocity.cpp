#include "cfd/common.hpp"
#include "cfd/velocity.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <limits>

void test_velocity_zero_and_constants() {
    std::cout << "  [1/5] Testing zero and uniform constant velocity fields...\n";
    const size_t n = 10000;
    std::vector<double> u(n, 0.0), v(n, 0.0), w(n, 0.0), mag(n, 1.0);

    cfd::velocity_magnitude_serial(u.data(), v.data(), w.data(), mag.data(), n);
    for (size_t i = 0; i < n; ++i) {
        assert(mag[i] == 0.0);
    }

    // 3-4-0 Pythagorean triple -> magnitude 5.0
    std::fill(u.begin(), u.end(), 3.0);
    std::fill(v.begin(), v.end(), 4.0);
    std::fill(w.begin(), w.end(), 0.0);

    cfd::velocity_magnitude_openmp(u.data(), v.data(), w.data(), mag.data(), n, 4);
    for (size_t i = 0; i < n; ++i) {
        assert(std::abs(mag[i] - 5.0) < 1e-14);
    }

    // 1-2-2 vector -> magnitude sqrt(1+4+4) = 3.0
    std::fill(u.begin(), u.end(), 1.0);
    std::fill(v.begin(), v.end(), 2.0);
    std::fill(w.begin(), w.end(), 2.0);

    cfd::velocity_magnitude_simd(u.data(), v.data(), w.data(), mag.data(), n, 8);
    for (size_t i = 0; i < n; ++i) {
        assert(std::abs(mag[i] - 3.0) < 1e-14);
    }
}

void test_velocity_thread_invariance() {
    std::cout << "  [2/5] Testing OpenMP thread count invariance (1, 2, 4, 8, 16, 32 threads)...\n";
    const size_t n = 250000;
    std::vector<double> u(n), v(n), w(n), mag_ref(n), mag_test(n);

    for (size_t i = 0; i < n; ++i) {
        u[i] = std::sin(static_cast<double>(i) * 0.01);
        v[i] = std::cos(static_cast<double>(i) * 0.02);
        w[i] = std::sin(static_cast<double>(i) * 0.03);
    }

    cfd::velocity_magnitude_serial(u.data(), v.data(), w.data(), mag_ref.data(), n);

    const int thread_sweep[] = {1, 2, 4, 8, 16, 32};
    for (int t : thread_sweep) {
        std::fill(mag_test.begin(), mag_test.end(), 0.0);
        cfd::velocity_magnitude_openmp(u.data(), v.data(), w.data(), mag_test.data(), n, t);

        double max_diff = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double diff = std::abs(mag_ref[i] - mag_test[i]);
            if (diff > max_diff) max_diff = diff;
        }
        assert(max_diff < 1e-14);
    }
}

void test_velocity_simd_equivalence() {
    std::cout << "  [3/5] Testing SIMD vectorization numerical equivalence...\n";
    const size_t n = 131072; // multiple of AVX2/AVX-512 register widths
    std::vector<double> u(n), v(n), w(n), mag_ser(n), mag_simd(n);

    for (size_t i = 0; i < n; ++i) {
        u[i] = static_cast<double>(i) * 0.001;
        v[i] = -static_cast<double>(i) * 0.0005;
        w[i] = 1.5;
    }

    cfd::velocity_magnitude_serial(u.data(), v.data(), w.data(), mag_ser.data(), n);
    cfd::velocity_magnitude_simd(u.data(), v.data(), w.data(), mag_simd.data(), n, 8);

    double max_diff = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = std::abs(mag_ser[i] - mag_simd[i]);
        if (diff > max_diff) max_diff = diff;
    }
    assert(max_diff < 1e-14);
}

void test_velocity_soa_vs_aos() {
    std::cout << "  [4/5] Testing Structure of Arrays (SoA) vs Array of Structures (AoS)...\n";
    const size_t n = 50000;
    std::vector<double> u(n), v(n), w(n);
    for (size_t i = 0; i < n; ++i) {
        u[i] = 2.5 + static_cast<double>(i % 100);
        v[i] = -1.2 + static_cast<double>(i % 50);
        w[i] = 0.8;
    }

    // Convert SoA to AoS
    auto aos = cfd::soa_to_aos(u.data(), v.data(), w.data(), n);
    assert(aos.size() == n);

    std::vector<double> mag_soa(n, 0.0);
    std::vector<double> mag_aos(n, 0.0);

    cfd::velocity_magnitude_openmp(u.data(), v.data(), w.data(), mag_soa.data(), n, 4);
    cfd::velocity_magnitude_aos_openmp(aos.data(), mag_aos.data(), n, 4);

    for (size_t i = 0; i < n; ++i) {
        double diff = std::abs(mag_soa[i] - mag_aos[i]);
        assert(diff < 1e-12);
    }

    // Convert back from AoS to SoA
    std::vector<double> u_back(n), v_back(n), w_back(n);
    cfd::aos_to_soa(aos.data(), u_back.data(), v_back.data(), w_back.data(), n);
    for (size_t i = 0; i < n; ++i) {
        assert(u_back[i] == u[i]);
        assert(v_back[i] == v[i]);
        assert(w_back[i] == w[i]);
    }
}

void test_velocity_edge_cases() {
    std::cout << "  [5/5] Testing boundary edge cases (single element, large values)...\n";
    double u_single = 1.0, v_single = 2.0, w_single = 2.0, mag_single = 0.0;
    cfd::velocity_magnitude_openmp(&u_single, &v_single, &w_single, &mag_single, 1, 4);
    assert(std::abs(mag_single - 3.0) < 1e-14);

    // Large floating point values
    double u_large = 1e150, v_large = 0.0, w_large = 0.0, mag_large = 0.0;
    cfd::velocity_magnitude_serial(&u_large, &v_large, &w_large, &mag_large, 1);
    assert(std::abs(mag_large - 1e150) < 1e136);
}

int main() {
    std::cout << "========================================================\n";
    std::cout << " Running Velocity Magnitude & Memory Layout Tests\n";
    std::cout << "========================================================\n";

    test_velocity_zero_and_constants();
    test_velocity_thread_invariance();
    test_velocity_simd_equivalence();
    test_velocity_soa_vs_aos();
    test_velocity_edge_cases();

    std::cout << "\n>>> ALL VELOCITY TESTS PASSED SUCCESSFULLY! <<<\n";
    return 0;
}
