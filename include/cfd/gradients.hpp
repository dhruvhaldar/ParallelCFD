#pragma once

#include "cfd/common.hpp"

namespace cfd {

// Serial gradient kernel (2nd-order central differences inside, 2nd-order one-sided at boundaries)
void gradient_serial(const double* phi, const Grid3D& grid,
                     double* grad_x, double* grad_y, double* grad_z);

// OpenMP gradient kernel with selectable collapse level (1, 2, or 3)
void gradient_openmp(const double* phi, const Grid3D& grid,
                     double* grad_x, double* grad_y, double* grad_z,
                     int collapse_level = 2, int num_threads = 0);

// Specific collapse versions for benchmarking
void gradient_openmp_collapse1(const double* phi, const Grid3D& grid,
                              double* grad_x, double* grad_y, double* grad_z,
                              int num_threads = 0);

void gradient_openmp_collapse2(const double* phi, const Grid3D& grid,
                              double* grad_x, double* grad_y, double* grad_z,
                              int num_threads = 0);

void gradient_openmp_collapse3(const double* phi, const Grid3D& grid,
                              double* grad_x, double* grad_y, double* grad_z,
                              int num_threads = 0);

} // namespace cfd
