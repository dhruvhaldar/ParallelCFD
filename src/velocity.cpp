#include "cfd/velocity.hpp"
#include <cmath>

namespace cfd {

void velocity_magnitude_serial(const double* u, const double* v, const double* w,
                               double* mag, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        mag[i] = std::sqrt(u[i] * u[i] + v[i] * v[i] + w[i] * w[i]);
    }
}

void velocity_magnitude_openmp(const double* u, const double* v, const double* w,
                               double* mag, size_t n, int num_threads) {
    const int threads = (num_threads > 0) ? num_threads : omp_get_max_threads();
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        mag[i] = std::sqrt(u[i] * u[i] + v[i] * v[i] + w[i] * w[i]);
    }
}

void velocity_magnitude_simd(const double* u, const double* v, const double* w,
                             double* mag, size_t n, int num_threads) {
    const int threads = (num_threads > 0) ? num_threads : omp_get_max_threads();
    #pragma omp parallel for simd num_threads(threads) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        mag[i] = std::sqrt(u[i] * u[i] + v[i] * v[i] + w[i] * w[i]);
    }
}

void velocity_magnitude_aos_serial(const VelocityAoS* vel, double* mag, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        mag[i] = std::sqrt(vel[i].u * vel[i].u + vel[i].v * vel[i].v + vel[i].w * vel[i].w);
    }
}

void velocity_magnitude_aos_openmp(const VelocityAoS* vel, double* mag, size_t n,
                                   int num_threads) {
    const int threads = (num_threads > 0) ? num_threads : omp_get_max_threads();
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        mag[i] = std::sqrt(vel[i].u * vel[i].u + vel[i].v * vel[i].v + vel[i].w * vel[i].w);
    }
}

void velocity_magnitude_aos_simd(const VelocityAoS* vel, double* mag, size_t n,
                                 int num_threads) {
    const int threads = (num_threads > 0) ? num_threads : omp_get_max_threads();
    #pragma omp parallel for simd num_threads(threads) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        mag[i] = std::sqrt(vel[i].u * vel[i].u + vel[i].v * vel[i].v + vel[i].w * vel[i].w);
    }
}

std::vector<VelocityAoS> soa_to_aos(const double* u, const double* v, const double* w, size_t n) {
    std::vector<VelocityAoS> aos(n);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        aos[i].u = u[i];
        aos[i].v = v[i];
        aos[i].w = w[i];
    }
    return aos;
}

void aos_to_soa(const VelocityAoS* vel, double* u, double* v, double* w, size_t n) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        u[i] = vel[i].u;
        v[i] = vel[i].v;
        w[i] = vel[i].w;
    }
}

} // namespace cfd
