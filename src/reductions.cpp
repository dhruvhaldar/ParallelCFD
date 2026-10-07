#include "cfd/reductions.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace cfd {

FieldStats field_stats_serial(const double* data, size_t n) {
    FieldStats stats;
    if (n == 0) return stats;

    double min_v = data[0];
    double max_v = data[0];
    double sum_v = 0.0;
    double sum_sq = 0.0;

    for (size_t i = 0; i < n; ++i) {
        const double val = data[i];
        if (val < min_v) min_v = val;
        if (val > max_v) max_v = val;
        sum_v += val;
        sum_sq += val * val;
    }

    stats.min_val = min_v;
    stats.max_val = max_v;
    stats.sum_val = sum_v;
    stats.mean_val = sum_v / static_cast<double>(n);
    stats.rms_val = std::sqrt(sum_sq / static_cast<double>(n));
    return stats;
}

FieldStats field_stats_openmp(const double* data, size_t n, int num_threads) {
    FieldStats stats;
    if (n == 0) return stats;

    double min_v = data[0];
    double max_v = data[0];
    double sum_v = 0.0;
    double sum_sq = 0.0;

    const int threads = (num_threads > 0) ? num_threads : omp_get_max_threads();

    #pragma omp parallel for num_threads(threads) schedule(static) \
        reduction(min:min_v) reduction(max:max_v) reduction(+:sum_v) reduction(+:sum_sq)
    for (size_t i = 0; i < n; ++i) {
        const double val = data[i];
        if (val < min_v) min_v = val;
        if (val > max_v) max_v = val;
        sum_v += val;
        sum_sq += val * val;
    }

    stats.min_val = min_v;
    stats.max_val = max_v;
    stats.sum_val = sum_v;
    stats.mean_val = sum_v / static_cast<double>(n);
    stats.rms_val = std::sqrt(sum_sq / static_cast<double>(n));
    return stats;
}

double kinetic_energy_serial(const double* u, const double* v, const double* w, size_t n) {
    double total = 0.0;
    for (size_t i = 0; i < n; ++i) {
        total += u[i] * u[i] + v[i] * v[i] + w[i] * w[i];
    }
    return 0.5 * total;
}

double kinetic_energy_openmp(const double* u, const double* v, const double* w, size_t n,
                             int num_threads) {
    double total = 0.0;
    const int threads = (num_threads > 0) ? num_threads : omp_get_max_threads();

    #pragma omp parallel for num_threads(threads) schedule(static) reduction(+:total)
    for (size_t i = 0; i < n; ++i) {
        total += u[i] * u[i] + v[i] * v[i] + w[i] * w[i];
    }
    return 0.5 * total;
}

NonAssociativityResult test_floating_point_nonassociativity(size_t n, int num_threads) {
    std::vector<double> vals(n);
    // Construct series with dynamic range where IEEE-754 order of summation matters:
    // harmonic-like series + small oscillatory perturbation
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        vals[i] = (1.0 / (static_cast<double>(i % 1000 + 1))) + ((i % 2 == 0) ? 1.0e-7 : -1.0e-7);
    }

    // Serial sum: strictly sequential accumulator
    double serial_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        serial_sum += vals[i];
    }

    // OpenMP parallel sum: partitioned across threads
    double parallel_sum = 0.0;
    const int threads = (num_threads > 0) ? num_threads : omp_get_max_threads();
    #pragma omp parallel for num_threads(threads) schedule(static) reduction(+:parallel_sum)
    for (size_t i = 0; i < n; ++i) {
        parallel_sum += vals[i];
    }

    NonAssociativityResult res;
    res.serial_sum = serial_sum;
    res.parallel_sum = parallel_sum;
    res.abs_diff = std::abs(serial_sum - parallel_sum);
    res.rel_diff = (std::abs(serial_sum) > 0.0) ? (res.abs_diff / std::abs(serial_sum)) : 0.0;
    return res;
}

} // namespace cfd
