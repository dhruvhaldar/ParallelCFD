#pragma once

#include "cfd/common.hpp"

namespace cfd {

// Serial divergence: computes div = du/dx + dv/dy + dw/dz
void divergence_serial(const double* u, const double* v, const double* w,
                       const Grid3D& grid, double* div);

// OpenMP parallel divergence with configurable collapse and threads
void divergence_openmp(const double* u, const double* v, const double* w,
                       const Grid3D& grid, double* div,
                       int collapse_level = 2, int num_threads = 0);

} // namespace cfd
