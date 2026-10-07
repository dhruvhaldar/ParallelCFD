#include "cfd/common.hpp"
#include "cfd/experiments.hpp"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "========================================================\n";
    std::cout << " ParallelCFD OpenMP Educational Experiments Unit Tests\n";
    std::cout << "========================================================\n";

    // Test constructs demo
    std::cout << "[TEST] Running OpenMP constructs demo...\n";
    auto demo = cfd::run_openmp_constructs_demo(4);
    assert(demo.single_executed_once);
    assert(demo.masked_executed_on_thread_0);
    assert(demo.barrier_synchronized);
    std::cout << "  -> PASSED: Single, masked, and barrier constructs verified!\n";

    // Test race condition correctness
    std::cout << "[TEST] Running race condition experiment...\n";
    auto race = cfd::run_race_condition_experiment(100000, 4);
    for (const auto& r : race) {
        if (r.method.find("Atomic") != std::string::npos ||
            r.method.find("Critical") != std::string::npos ||
            r.method.find("Reduction") != std::string::npos) {
            assert(r.error == 0);
        }
    }
    std::cout << "  -> PASSED: Synchronized counters have zero error!\n";

    // Test false sharing experiment
    std::cout << "[TEST] Running false sharing experiment...\n";
    auto fs = cfd::run_false_sharing_experiment(1000000, 4);
    assert(fs.unpadded_seconds > 0.0);
    assert(fs.padded_seconds > 0.0);
    std::cout << "  -> PASSED: False sharing experiment completed (speedup: "
              << fs.speedup_from_padding << "x)!\n";

    std::cout << "\nALL EXPERIMENT TESTS PASSED SUCCESSFULLY!\n";
    return 0;
}
