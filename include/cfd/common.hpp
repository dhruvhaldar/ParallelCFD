#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>

#if defined(_OPENMP)
#include <omp.h>
#else
inline int omp_get_max_threads() { return 1; }
inline int omp_get_num_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
inline double omp_get_wtime() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}
#endif

namespace cfd {

constexpr size_t CACHE_LINE_SIZE = 64; // Standard x86_64 cache line size (bytes)

// 3D Grid dimensions and uniform spacing
struct Grid3D {
    size_t nx{0};
    size_t ny{0};
    size_t nz{0};
    double dx{1.0};
    double dy{1.0};
    double dz{1.0};
    double x0{0.0};
    double y0{0.0};
    double z0{0.0};

    Grid3D() = default;
    Grid3D(size_t nx_, size_t ny_, size_t nz_,
           double dx_ = 1.0, double dy_ = 1.0, double dz_ = 1.0,
           double x0_ = 0.0, double y0_ = 0.0, double z0_ = 0.0)
        : nx(nx_), ny(ny_), nz(nz_),
          dx(dx_), dy(dy_), dz(dz_),
          x0(x0_), y0(y0_), z0(z0_) {}

    [[nodiscard]] size_t total_cells() const noexcept {
        return nx * ny * nz;
    }

    // Row-major 3D indexing: k is fastest varying (stride 1)
    [[nodiscard]] inline size_t index(size_t i, size_t j, size_t k) const noexcept {
        return (i * ny + j) * nz + k;
    }

    [[nodiscard]] inline size_t stride_x() const noexcept { return ny * nz; }
    [[nodiscard]] inline size_t stride_y() const noexcept { return nz; }
    [[nodiscard]] inline size_t stride_z() const noexcept { return 1; }

    [[nodiscard]] inline double x(size_t i) const noexcept { return x0 + static_cast<double>(i) * dx; }
    [[nodiscard]] inline double y(size_t j) const noexcept { return y0 + static_cast<double>(j) * dy; }
    [[nodiscard]] inline double z(size_t k) const noexcept { return z0 + static_cast<double>(k) * dz; }
};

// AoS representation for comparison experiments
struct VelocityAoS {
    double u;
    double v;
    double w;
};

// Reduction statistics bundle
struct FieldStats {
    double min_val{0.0};
    double max_val{0.0};
    double sum_val{0.0};
    double mean_val{0.0};
    double rms_val{0.0};
};

// High-resolution benchmark timer
class Timer {
public:
    void start() {
        start_time_ = omp_get_wtime();
    }
    double stop() {
        return omp_get_wtime() - start_time_;
    }
private:
    double start_time_{0.0};
};

} // namespace cfd
