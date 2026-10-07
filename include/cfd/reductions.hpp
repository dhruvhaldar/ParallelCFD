#pragma once

#include "cfd/common.hpp"

namespace cfd {

// Serial reductions
FieldStats field_stats_serial(const double* data, size_t n);

double kinetic_energy_serial(const double* u, const double* v, const double* w, size_t n);

// OpenMP parallel reductions
FieldStats field_stats_openmp(const double* data, size_t n, int num_threads = 0);

double kinetic_energy_openmp(const double* u, const double* v, const double* w, size_t n,
                             int num_threads = 0);

// Floating-point non-associativity demonstration
struct NonAssociativityResult {
    double serial_sum{0.0};
    double parallel_sum{0.0};
    double abs_diff{0.0};
    double rel_diff{0.0};
};

NonAssociativityResult test_floating_point_nonassociativity(size_t n = 10000000, int num_threads = 0);

} // namespace cfd
