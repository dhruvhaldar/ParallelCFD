#include "cfd/common.hpp"
#include "cfd/grid.hpp"
#include "cfd/velocity.hpp"
#include "cfd/gradients.hpp"
#include "cfd/experiments.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>

int main() {
    std::cout << "========================================================================\n";
    std::cout << " ParallelCFD: OpenMP Architecture & Performance Experiments\n";
    std::cout << "========================================================================\n\n";

    // 1. Race Condition Experiment
    std::cout << "=== 1. Race Condition & Accumulator Synchronization Experiment ===\n";
    auto race_results = cfd::run_race_condition_experiment(10000000);
    for (const auto& r : race_results) {
        std::cout << "Method: " << std::left << std::setw(32) << r.method
                  << " | Expected: " << std::setw(10) << r.expected_sum
                  << " | Actual: " << std::setw(10) << r.actual_sum
                  << " | Error: " << std::setw(10) << r.error
                  << " | Time: " << std::fixed << std::setprecision(4) << r.elapsed_seconds * 1000.0 << " ms\n";
    }

    // 2. False Sharing Experiment
    std::cout << "\n=== 2. False Sharing Cache-Line Invalidation Experiment ===\n";
    auto fs = cfd::run_false_sharing_experiment(50000000);
    std::cout << "Hardware Threads: " << fs.num_threads << ", Iterations: " << fs.iterations << "\n";
    std::cout << "Unpadded Counters (Adjacent on 64B Cache Line): "
              << std::fixed << std::setprecision(4) << fs.unpadded_seconds * 1000.0 << " ms\n";
    std::cout << "Padded Counters   (alignas(64) Dedicated Cache): "
              << std::fixed << std::setprecision(4) << fs.padded_seconds * 1000.0 << " ms\n";
    std::cout << "Speedup by eliminating False Sharing:           "
              << std::fixed << std::setprecision(2) << fs.speedup_from_padding << "x\n";

    // 3. Memory Layout: SoA vs AoS
    std::cout << "\n=== 3. Memory Layout: Structure of Arrays (SoA) vs Array of Structures (AoS) ===\n";
    const size_t n_cells = 128 * 128 * 128; // ~2.1M cells
    std::vector<double> u(n_cells, 1.2), v(n_cells, -0.8), w(n_cells, 0.5), mag(n_cells, 0.0);
    auto aos_data = cfd::soa_to_aos(u.data(), v.data(), w.data(), n_cells);

    const int max_threads = omp_get_max_threads();
    const int reps = 50;

    // SoA Serial
    cfd::Timer t_soa_ser;
    t_soa_ser.start();
    for (int r = 0; r < reps; ++r) cfd::velocity_magnitude_serial(u.data(), v.data(), w.data(), mag.data(), n_cells);
    const double time_soa_ser = t_soa_ser.stop() / reps;

    // AoS Serial
    cfd::Timer t_aos_ser;
    t_aos_ser.start();
    for (int r = 0; r < reps; ++r) cfd::velocity_magnitude_aos_serial(aos_data.data(), mag.data(), n_cells);
    const double time_aos_ser = t_aos_ser.stop() / reps;

    // SoA OpenMP SIMD
    cfd::Timer t_soa_par;
    t_soa_par.start();
    for (int r = 0; r < reps; ++r) cfd::velocity_magnitude_simd(u.data(), v.data(), w.data(), mag.data(), n_cells, max_threads);
    const double time_soa_par = t_soa_par.stop() / reps;

    // AoS OpenMP SIMD
    cfd::Timer t_aos_par;
    t_aos_par.start();
    for (int r = 0; r < reps; ++r) cfd::velocity_magnitude_aos_simd(aos_data.data(), mag.data(), n_cells, max_threads);
    const double time_aos_par = t_aos_par.stop() / reps;

    std::cout << "Serial: SoA Time = " << time_soa_ser * 1000.0 << " ms | AoS Time = " << time_aos_ser * 1000.0 << " ms\n";
    std::cout << "Parallel (" << max_threads << " threads): SoA Time = " << time_soa_par * 1000.0
              << " ms | AoS Time = " << time_aos_par * 1000.0 << " ms\n";
    std::cout << "SoA Advantage: " << (time_aos_par / time_soa_par) << "x faster due to contiguous SIMD vector loads\n";

    // 4. Loop Collapse Experiment: collapse(1) vs collapse(2) vs collapse(3)
    std::cout << "\n=== 4. Loop Collapse Experiment: collapse(1) vs collapse(2) vs collapse(3) ===\n";
    const cfd::Grid3D grid(128, 128, 128, 0.01, 0.01, 0.01);
    std::vector<double> phi(grid.total_cells(), 1.0);
    std::vector<double> gx(grid.total_cells(), 0.0), gy(grid.total_cells(), 0.0), gz(grid.total_cells(), 0.0);

    cfd::Timer t_c1, t_c2, t_c3;
    t_c1.start();
    for (int r = 0; r < 5; ++r) cfd::gradient_openmp_collapse1(phi.data(), grid, gx.data(), gy.data(), gz.data(), max_threads);
    const double time_c1 = t_c1.stop() / 5;

    t_c2.start();
    for (int r = 0; r < 5; ++r) cfd::gradient_openmp_collapse2(phi.data(), grid, gx.data(), gy.data(), gz.data(), max_threads);
    const double time_c2 = t_c2.stop() / 5;

    t_c3.start();
    for (int r = 0; r < 5; ++r) cfd::gradient_openmp_collapse3(phi.data(), grid, gx.data(), gy.data(), gz.data(), max_threads);
    const double time_c3 = t_c3.stop() / 5;

    std::cout << "collapse(1) [outer loop only]: " << time_c1 * 1000.0 << " ms\n";
    std::cout << "collapse(2) [outer 2 loops + inner SIMD]: " << time_c2 * 1000.0 << " ms\n";
    std::cout << "collapse(3) [all 3 loops]: " << time_c3 * 1000.0 << " ms\n";

    // 5. OpenMP Construct Demo
    std::cout << "\n=== 5. OpenMP Advanced Constructs Verification ===\n";
    auto demo = cfd::run_openmp_constructs_demo(max_threads);
    std::cout << demo.summary << "\n";

    // Export experiments data to CSV
    std::ofstream exp_csv("results/experiments_results.csv");
    if (exp_csv.is_open()) {
        exp_csv << "experiment,metric,value\n";
        for (const auto& r : race_results) {
            exp_csv << "race_condition," << r.method << "_time_ms," << r.elapsed_seconds * 1000.0 << "\n";
            exp_csv << "race_condition," << r.method << "_error," << r.error << "\n";
        }
        exp_csv << "false_sharing,unpadded_ms," << fs.unpadded_seconds * 1000.0 << "\n";
        exp_csv << "false_sharing,padded_ms," << fs.padded_seconds * 1000.0 << "\n";
        exp_csv << "false_sharing,speedup," << fs.speedup_from_padding << "\n";
        exp_csv << "memory_layout,soa_par_ms," << time_soa_par * 1000.0 << "\n";
        exp_csv << "memory_layout,aos_par_ms," << time_aos_par * 1000.0 << "\n";
        exp_csv << "collapse,collapse1_ms," << time_c1 * 1000.0 << "\n";
        exp_csv << "collapse,collapse2_ms," << time_c2 * 1000.0 << "\n";
        exp_csv << "collapse,collapse3_ms," << time_c3 * 1000.0 << "\n";
        std::cout << "\nSaved experiment results to results/experiments_results.csv\n";
    }

    return 0;
}
