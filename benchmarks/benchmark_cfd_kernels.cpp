#include "cfd/common.hpp"
#include "cfd/grid.hpp"
#include "cfd/velocity.hpp"
#include "cfd/reductions.hpp"
#include "cfd/gradients.hpp"
#include "cfd/divergence.hpp"
#include "cfd/vorticity.hpp"
#include "cfd/qcriterion.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>

struct KernelTiming {
    std::string kernel_name;
    size_t grid_dim;
    int threads;
    double time_sec;
    double speedup;
    double efficiency;
    double throughput_mcells_sec;
};

int main(int argc, char** argv) {
    std::cout << "========================================================================\n";
    std::cout << " ParallelCFD: Core OpenMP Scaling Benchmark\n";
    std::cout << "========================================================================\n";

    size_t grid_dim = 128;
    if (argc > 1) {
        grid_dim = std::stoul(argv[1]);
    }

    const cfd::Grid3D grid(grid_dim, grid_dim, grid_dim, 0.01, 0.01, 0.01);
    const size_t n = grid.total_cells();
    std::cout << "Grid: " << grid_dim << "x" << grid_dim << "x" << grid_dim
              << " (" << n << " cells, " << std::fixed << std::setprecision(2)
              << (n * sizeof(double) * 4) / (1024.0 * 1024.0) << " MB for 4 fields)\n\n";

    std::cout << "Generating synthetic Taylor-Green vortex field...\n";
    cfd::Field3D field = cfd::generate_taylor_green_vortex(grid);

    std::vector<double> out1(n, 0.0);
    std::vector<double> out2(n, 0.0);
    std::vector<double> out3(n, 0.0);
    std::vector<double> out4(n, 0.0);

    const int max_hw_threads = omp_get_max_threads();
    std::vector<int> thread_counts;
    for (int t = 1; t <= max_hw_threads; t *= 2) {
        thread_counts.push_back(t);
    }
    if (thread_counts.back() != max_hw_threads) {
        thread_counts.push_back(max_hw_threads);
    }

    std::vector<KernelTiming> timings;

    // Benchmark 1: Velocity Magnitude
    std::cout << "--- Benchmarking Velocity Magnitude (|u|) ---\n";
    double t1_vel = 0.0;
    for (int t : thread_counts) {
        cfd::Timer timer;
        // Warmup
        cfd::velocity_magnitude_openmp(field.u.data(), field.v.data(), field.w.data(), out1.data(), n, t);
        // Measure 3 iterations
        timer.start();
        const int reps = 3;
        for (int r = 0; r < reps; ++r) {
            cfd::velocity_magnitude_openmp(field.u.data(), field.v.data(), field.w.data(), out1.data(), n, t);
        }
        const double elapsed = timer.stop() / reps;
        if (t == 1) t1_vel = elapsed;

        const double sp = t1_vel / elapsed;
        const double eff = sp / t;
        const double mcells = (n / 1e6) / elapsed;

        timings.push_back({"VelocityMagnitude", grid_dim, t, elapsed, sp, eff, mcells});
        std::cout << "Threads: " << std::setw(2) << t
                  << " | Time: " << std::setw(8) << std::fixed << std::setprecision(4) << elapsed * 1000.0 << " ms"
                  << " | Speedup: " << std::setw(5) << std::setprecision(2) << sp << "x"
                  << " | Efficiency: " << std::setw(5) << std::setprecision(2) << eff * 100.0 << "%"
                  << " | " << std::setw(7) << std::setprecision(1) << mcells << " Mcells/s\n";
    }

    // Benchmark 2: Divergence
    std::cout << "\n--- Benchmarking Divergence (div u) ---\n";
    double t1_div = 0.0;
    for (int t : thread_counts) {
        cfd::Timer timer;
        cfd::divergence_openmp(field.u.data(), field.v.data(), field.w.data(), grid, out1.data(), 2, t);
        timer.start();
        const int reps = 3;
        for (int r = 0; r < reps; ++r) {
            cfd::divergence_openmp(field.u.data(), field.v.data(), field.w.data(), grid, out1.data(), 2, t);
        }
        const double elapsed = timer.stop() / reps;
        if (t == 1) t1_div = elapsed;

        const double sp = t1_div / elapsed;
        const double eff = sp / t;
        const double mcells = (n / 1e6) / elapsed;

        timings.push_back({"Divergence", grid_dim, t, elapsed, sp, eff, mcells});
        std::cout << "Threads: " << std::setw(2) << t
                  << " | Time: " << std::setw(8) << std::fixed << std::setprecision(4) << elapsed * 1000.0 << " ms"
                  << " | Speedup: " << std::setw(5) << std::setprecision(2) << sp << "x"
                  << " | Efficiency: " << std::setw(5) << std::setprecision(2) << eff * 100.0 << "%"
                  << " | " << std::setw(7) << std::setprecision(1) << mcells << " Mcells/s\n";
    }

    // Benchmark 3: Vorticity Vector & Magnitude
    std::cout << "\n--- Benchmarking Vorticity Vector & Mag (curl u) ---\n";
    double t1_vort = 0.0;
    for (int t : thread_counts) {
        cfd::Timer timer;
        cfd::vorticity_openmp(field.u.data(), field.v.data(), field.w.data(), grid,
                             out1.data(), out2.data(), out3.data(), out4.data(), 2, t);
        timer.start();
        const int reps = 3;
        for (int r = 0; r < reps; ++r) {
            cfd::vorticity_openmp(field.u.data(), field.v.data(), field.w.data(), grid,
                                 out1.data(), out2.data(), out3.data(), out4.data(), 2, t);
        }
        const double elapsed = timer.stop() / reps;
        if (t == 1) t1_vort = elapsed;

        const double sp = t1_vort / elapsed;
        const double eff = sp / t;
        const double mcells = (n / 1e6) / elapsed;

        timings.push_back({"Vorticity", grid_dim, t, elapsed, sp, eff, mcells});
        std::cout << "Threads: " << std::setw(2) << t
                  << " | Time: " << std::setw(8) << std::fixed << std::setprecision(4) << elapsed * 1000.0 << " ms"
                  << " | Speedup: " << std::setw(5) << std::setprecision(2) << sp << "x"
                  << " | Efficiency: " << std::setw(5) << std::setprecision(2) << eff * 100.0 << "%"
                  << " | " << std::setw(7) << std::setprecision(1) << mcells << " Mcells/s\n";
    }

    // Benchmark 4: Q-Criterion
    std::cout << "\n--- Benchmarking Q-Criterion ---\n";
    double t1_q = 0.0;
    for (int t : thread_counts) {
        cfd::Timer timer;
        cfd::q_criterion_openmp(field.u.data(), field.v.data(), field.w.data(), grid, out1.data(), 2, t);
        timer.start();
        const int reps = 3;
        for (int r = 0; r < reps; ++r) {
            cfd::q_criterion_openmp(field.u.data(), field.v.data(), field.w.data(), grid, out1.data(), 2, t);
        }
        const double elapsed = timer.stop() / reps;
        if (t == 1) t1_q = elapsed;

        const double sp = t1_q / elapsed;
        const double eff = sp / t;
        const double mcells = (n / 1e6) / elapsed;

        timings.push_back({"QCriterion", grid_dim, t, elapsed, sp, eff, mcells});
        std::cout << "Threads: " << std::setw(2) << t
                  << " | Time: " << std::setw(8) << std::fixed << std::setprecision(4) << elapsed * 1000.0 << " ms"
                  << " | Speedup: " << std::setw(5) << std::setprecision(2) << sp << "x"
                  << " | Efficiency: " << std::setw(5) << std::setprecision(2) << eff * 100.0 << "%"
                  << " | " << std::setw(7) << std::setprecision(1) << mcells << " Mcells/s\n";
    }

    // Export CSV
    std::ofstream csv("results/kernel_scaling.csv");
    if (csv.is_open()) {
        csv << "kernel,grid_dim,threads,time_sec,speedup,efficiency,throughput_mcells\n";
        for (const auto& tm : timings) {
            csv << tm.kernel_name << ","
                << tm.grid_dim << ","
                << tm.threads << ","
                << tm.time_sec << ","
                << tm.speedup << ","
                << tm.efficiency << ","
                << tm.throughput_mcells_sec << "\n";
        }
        std::cout << "\nWrote scaling data to results/kernel_scaling.csv\n";
    }

    return 0;
}
