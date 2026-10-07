#pragma once

#include "cfd/common.hpp"

namespace cfd {

// Serial vorticity: computes wx, wy, wz, and optionally vorticity magnitude
void vorticity_serial(const double* u, const double* v, const double* w,
                      const Grid3D& grid,
                      double* wx, double* wy, double* wz, double* mag = nullptr);

// OpenMP parallel vorticity
void vorticity_openmp(const double* u, const double* v, const double* w,
                      const Grid3D& grid,
                      double* wx, double* wy, double* wz, double* mag = nullptr,
                      int collapse_level = 2, int num_threads = 0);

} // namespace cfd
