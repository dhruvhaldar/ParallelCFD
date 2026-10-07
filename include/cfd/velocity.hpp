#pragma once

#include "cfd/common.hpp"
#include <vector>

namespace cfd {

// SoA implementations
void velocity_magnitude_serial(const double* u, const double* v, const double* w,
                               double* mag, size_t n);

void velocity_magnitude_openmp(const double* u, const double* v, const double* w,
                               double* mag, size_t n, int num_threads = 0);

void velocity_magnitude_simd(const double* u, const double* v, const double* w,
                             double* mag, size_t n, int num_threads = 0);

// AoS implementations for memory layout studies
void velocity_magnitude_aos_serial(const VelocityAoS* vel, double* mag, size_t n);

void velocity_magnitude_aos_openmp(const VelocityAoS* vel, double* mag, size_t n,
                                   int num_threads = 0);

void velocity_magnitude_aos_simd(const VelocityAoS* vel, double* mag, size_t n,
                                 int num_threads = 0);

// Data structure conversion utilities
std::vector<VelocityAoS> soa_to_aos(const double* u, const double* v, const double* w, size_t n);
void aos_to_soa(const VelocityAoS* vel, double* u, double* v, double* w, size_t n);

} // namespace cfd
