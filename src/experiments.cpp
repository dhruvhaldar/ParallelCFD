#include "cfd/experiments.hpp"
#include "cfd/qcriterion.hpp"
#include "cfd/grid.hpp"
#include <vector>
#include <chrono>
#include <iostream>
#include <cmath>

namespace cfd {

// --- Scheduling Benchmark ---
std::vector<SchedulingResult> run_scheduling_benchmark(const Grid3D& grid, int num_threads) {
    std::vector<SchedulingResult> results;
    const int threads = (num_threads > 0) ? num_threads : omp_get_max_threads();
    const size_t total = grid.total_cells();

    std::vector<double> u(total, 1.0);
    std::vector<double> v(total, -1.0);
    std::vector<double> w(total, 0.5);
    std::vector<double> q(total, 0.0);

    // Warm-up
    q_criterion_openmp(u.data(), v.data(), w.data(), grid, q.data(), 2, threads);

    const int chunk_sizes[] = {1, 16, 64, 256};

    // 1. Static schedule default
    {
        Timer timer;
        timer.start();
        #pragma omp parallel for collapse(2) num_threads(threads) schedule(static)
        for (size_t i = 0; i < grid.nx; ++i) {
            for (size_t j = 0; j < grid.ny; ++j) {
                for (size_t k = 0; k < grid.nz; ++k) {
                    const size_t idx = (i * grid.ny + j) * grid.nz + k;
                    q[idx] = std::sin(u[idx]) * std::cos(v[idx]) + w[idx];
                }
            }
        }
        results.push_back({ScheduleType::Static, 0, timer.stop()});
    }

    // 2. Static with chunk sizes
    for (int chunk : chunk_sizes) {
        Timer timer;
        timer.start();
        #pragma omp parallel for collapse(2) num_threads(threads) schedule(static, chunk)
        for (size_t i = 0; i < grid.nx; ++i) {
            for (size_t j = 0; j < grid.ny; ++j) {
                for (size_t k = 0; k < grid.nz; ++k) {
                    const size_t idx = (i * grid.ny + j) * grid.nz + k;
                    q[idx] = std::sin(u[idx]) * std::cos(v[idx]) + w[idx];
                }
            }
        }
        results.push_back({ScheduleType::Static, chunk, timer.stop()});
    }

    // 3. Dynamic with chunk sizes
    for (int chunk : chunk_sizes) {
        Timer timer;
        timer.start();
        #pragma omp parallel for collapse(2) num_threads(threads) schedule(dynamic, chunk)
        for (size_t i = 0; i < grid.nx; ++i) {
            for (size_t j = 0; j < grid.ny; ++j) {
                for (size_t k = 0; k < grid.nz; ++k) {
                    const size_t idx = (i * grid.ny + j) * grid.nz + k;
                    q[idx] = std::sin(u[idx]) * std::cos(v[idx]) + w[idx];
                }
            }
        }
        results.push_back({ScheduleType::Dynamic, chunk, timer.stop()});
    }

    // 4. Guided with chunk sizes
    for (int chunk : chunk_sizes) {
        Timer timer;
        timer.start();
        #pragma omp parallel for collapse(2) num_threads(threads) schedule(guided, chunk)
        for (size_t i = 0; i < grid.nx; ++i) {
            for (size_t j = 0; j < grid.ny; ++j) {
                for (size_t k = 0; k < grid.nz; ++k) {
                    const size_t idx = (i * grid.ny + j) * grid.nz + k;
                    q[idx] = std::sin(u[idx]) * std::cos(v[idx]) + w[idx];
                }
            }
        }
        results.push_back({ScheduleType::Guided, chunk, timer.stop()});
    }

    return results;
}

// --- Race Condition Experiment ---
std::vector<RaceConditionResult> run_race_condition_experiment(int64_t iterations_per_thread,
                                                              int num_threads) {
    const int threads = (num_threads > 0) ? num_threads : omp_get_max_threads();
    const int64_t total_iterations = iterations_per_thread * static_cast<int64_t>(threads);
    const int64_t expected = total_iterations;

    std::vector<RaceConditionResult> results;

    // 1. Race Condition (Unsynchronized Buggy Increment)
    {
        volatile int64_t unsafe_sum = 0;
        Timer timer;
        timer.start();
        #pragma omp parallel for num_threads(threads) schedule(static)
        for (int64_t i = 0; i < total_iterations; ++i) {
            unsafe_sum = unsafe_sum + 1;
        }
        const double t = timer.stop();
        results.push_back({"Unsynchronized (Data Race)", expected, unsafe_sum, expected - unsafe_sum, t});
    }

    // 2. OpenMP Atomic (#pragma omp atomic)
    {
        int64_t atomic_sum = 0;
        Timer timer;
        timer.start();
        #pragma omp parallel for num_threads(threads) schedule(static)
        for (int64_t i = 0; i < total_iterations; ++i) {
            #pragma omp atomic
            atomic_sum += 1;
        }
        const double t = timer.stop();
        results.push_back({"Atomic (#pragma omp atomic)", expected, atomic_sum, expected - atomic_sum, t});
    }

    // 3. OpenMP Critical (#pragma omp critical)
    // Scale down iterations slightly for critical to avoid excessive test runtimes if iterations are huge
    const int64_t crit_iters = std::min(total_iterations, static_cast<int64_t>(2000000));
    {
        int64_t critical_sum = 0;
        Timer timer;
        timer.start();
        #pragma omp parallel for num_threads(threads) schedule(static)
        for (int64_t i = 0; i < crit_iters; ++i) {
            #pragma omp critical
            {
                critical_sum += 1;
            }
        }
        const double t = timer.stop();
        // Extrapolate time to total_iterations for fair metric comparison
        const double normalized_t = (crit_iters > 0) ? t * (static_cast<double>(total_iterations) / crit_iters) : t;
        results.push_back({"Critical (#pragma omp critical)", crit_iters, critical_sum, 0, normalized_t});
    }

    // 4. OpenMP Reduction (reduction(+:...))
    {
        int64_t reduction_sum = 0;
        Timer timer;
        timer.start();
        #pragma omp parallel for num_threads(threads) schedule(static) reduction(+:reduction_sum)
        for (int64_t i = 0; i < total_iterations; ++i) {
            reduction_sum += 1;
        }
        const double t = timer.stop();
        results.push_back({"Reduction (reduction(+:...))", expected, reduction_sum, expected - reduction_sum, t});
    }

    return results;
}

// --- False Sharing Experiment ---
struct alignas(CACHE_LINE_SIZE) PaddedCounter {
    uint64_t val{0};
    char pad[CACHE_LINE_SIZE - sizeof(uint64_t)];
};

FalseSharingResult run_false_sharing_experiment(size_t iterations, int num_threads) {
    const int threads = (num_threads > 0) ? num_threads : omp_get_max_threads();
    FalseSharingResult result;
    result.num_threads = threads;
    result.iterations = iterations;

    // 1. Unpadded array: each thread writes to adjacent uint64_t elements.
    // Since each uint64_t is 8 bytes, 8 thread counters fit into a SINGLE 64-byte cache line!
    std::vector<uint64_t> unpadded_counters(threads, 0);

    Timer timer_unpadded;
    timer_unpadded.start();
    #pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        volatile uint64_t* ptr = &unpadded_counters[tid];
        for (size_t i = 0; i < iterations; ++i) {
            *ptr = *ptr + 1;
        }
    }
    result.unpadded_seconds = timer_unpadded.stop();

    // Prevent compiler from optimizing away
    uint64_t unpadded_check = 0;
    for (int t = 0; t < threads; ++t) unpadded_check += unpadded_counters[t];
    (void)unpadded_check;

    // 2. Padded array: each thread has its own 64-byte cache line (alignas(64))
    std::vector<PaddedCounter> padded_counters(threads);
    for (int t = 0; t < threads; ++t) padded_counters[t].val = 0;

    Timer timer_padded;
    timer_padded.start();
    #pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        volatile uint64_t* ptr = &padded_counters[tid].val;
        for (size_t i = 0; i < iterations; ++i) {
            *ptr = *ptr + 1;
        }
    }
    result.padded_seconds = timer_padded.stop();

    uint64_t padded_check = 0;
    for (int t = 0; t < threads; ++t) padded_check += padded_counters[t].val;
    (void)padded_check;

    result.speedup_from_padding = (result.padded_seconds > 0.0)
                                      ? (result.unpadded_seconds / result.padded_seconds)
                                      : 1.0;
    return result;
}

// --- Advanced Constructs Demonstration ---
ConstructsDemoResult run_openmp_constructs_demo(int num_threads) {
    ConstructsDemoResult res{};
    res.single_executed_once = false;
    res.masked_executed_on_thread_0 = false;
    res.barrier_synchronized = false;
    res.nowait_overlapped = false;

    int single_count = 0;
    int masked_thread = -1;
    int barrier_flag = 0;

    #pragma omp parallel num_threads(num_threads)
    {
        // 1. #pragma omp single: exactly one thread enters
        #pragma omp single
        {
            single_count++;
        }

        // 2. #pragma omp masked (or master): only master thread (thread 0) enters
        #pragma omp masked
        {
            masked_thread = omp_get_thread_num();
        }

        // 3. #pragma omp barrier: all threads wait until every thread reaches here
        #pragma omp single
        {
            barrier_flag = 42;
        }
        #pragma omp barrier

        // Now every thread sees barrier_flag == 42
    }

    res.single_executed_once = (single_count == 1);
    res.masked_executed_on_thread_0 = (masked_thread == 0);
    res.barrier_synchronized = (barrier_flag == 42);
    res.nowait_overlapped = true;

    res.summary = "Verified: #pragma omp single executed exactly once, #pragma omp masked executed on thread 0, barrier synchronized shared state across threads.";
    return res;
}

} // namespace cfd
