#pragma once

#include "cfd/common.hpp"
#include <vector>
#include <string>

namespace cfd {

// --- Scheduling Experiment ---
enum class ScheduleType {
    Static,
    Dynamic,
    Guided
};

struct SchedulingResult {
    ScheduleType type;
    int chunk_size;
    double elapsed_seconds;
};

// Measures scheduling impact on CFD calculations
std::vector<SchedulingResult> run_scheduling_benchmark(const Grid3D& grid, int num_threads = 0);

// --- Race Condition & Accumulator Demonstration ---
struct RaceConditionResult {
    std::string method;
    int64_t expected_sum;
    int64_t actual_sum;
    int64_t error;
    double elapsed_seconds;
};

std::vector<RaceConditionResult> run_race_condition_experiment(int64_t iterations_per_thread = 5000000,
                                                              int num_threads = 0);

// --- False Sharing Experiment ---
struct FalseSharingResult {
    double unpadded_seconds;
    double padded_seconds;
    double speedup_from_padding;
    int num_threads;
    size_t iterations;
};

FalseSharingResult run_false_sharing_experiment(size_t iterations = 50000000, int num_threads = 0);

// --- OpenMP Advanced Constructs Demonstration ---
struct ConstructsDemoResult {
    bool single_executed_once;
    bool masked_executed_on_thread_0;
    bool barrier_synchronized;
    bool nowait_overlapped;
    std::string summary;
};

ConstructsDemoResult run_openmp_constructs_demo(int num_threads = 4);

} // namespace cfd
