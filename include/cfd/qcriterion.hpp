#pragma once

#include "cfd/common.hpp"

namespace cfd {

// Serial Q-criterion calculation: Q = 0.5 * (||Omega||^2 - ||S||^2)
void q_criterion_serial(const double* u, const double* v, const double* w,
                        const Grid3D& grid, double* q_field);

// OpenMP parallel Q-criterion calculation
void q_criterion_openmp(const double* u, const double* v, const double* w,
                        const Grid3D& grid, double* q_field,
                        int collapse_level = 2, int num_threads = 0);

} // namespace cfd
