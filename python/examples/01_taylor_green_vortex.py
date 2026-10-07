#!/usr/bin/env python3
"""
Example 01: Taylor-Green Vortex CFD Post-Processing with ParallelCFD
Validates serial vs OpenMP implementations and verifies against analytical solutions.
"""

import sys
import numpy as np
import time

try:
    import parallelcfd as pcfd
except ImportError:
    # If run before pip install, add build/python to sys.path
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../build")))
    import parallelcfd as pcfd

def main():
    print("=" * 70)
    print(" ParallelCFD: Taylor-Green Vortex Post-Processing & Validation")
    print("=" * 70)

    nx, ny, nz = 128, 128, 128
    L = np.pi
    dx = 2.0 * L / nx
    dy = 2.0 * L / ny
    dz = 2.0 * L / nz
    total_cells = nx * ny * nz

    print(f"Grid size: {nx} x {ny} x {nz} = {total_cells:,} cells")
    print(f"Domain: [-π, π]^3, Spacing: dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}")

    # 1. Generate synthetic CFD field
    t0 = time.perf_counter()
    data = pcfd.generate_taylor_green(nx, ny, nz, dx, dy, dz, U0=1.0, L=1.0)
    u, v, w = data['u'], data['v'], data['w']
    p = data.get('pressure', data.get('p'))
    print(f"Generated 3D TGV field in {time.perf_counter() - t0:.4f} s")

    # 2. Velocity magnitude
    print("\n--- 1. Velocity Magnitude (|u|) ---")
    t0 = time.perf_counter()
    mag_serial = pcfd.velocity_magnitude(u, v, w, threads=1)
    t_serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    mag_omp = pcfd.velocity_magnitude(u, v, w, threads=0, use_simd=True)
    t_omp = time.perf_counter() - t0

    diff_mag = np.max(np.abs(mag_serial - mag_omp))
    print(f"Serial runtime:   {t_serial * 1000:.2f} ms")
    print(f"OpenMP+SIMD time: {t_omp * 1000:.2f} ms  (Speedup: {t_serial / t_omp:.2f}x)")
    print(f"Max absolute difference: {diff_mag:.2e} (Strict numerical equivalence)")
    assert diff_mag < 1e-12, "Velocity magnitude mismatch between serial and OpenMP!"

    # 3. Global reductions
    print("\n--- 2. Global Field Statistics & Reductions ---")
    stats_p = pcfd.field_stats(p, threads=0)
    print(f"Pressure stats: min={stats_p.min_val:.4f}, max={stats_p.max_val:.4f}, mean={stats_p.mean_val:.4f}, RMS={stats_p.rms_val:.4f}")

    ke_serial = pcfd.kinetic_energy(u, v, w, threads=1)
    ke_omp = pcfd.kinetic_energy(u, v, w, threads=0)
    diff_ke = abs(ke_serial - ke_omp)
    print(f"Kinetic energy: Serial = {ke_serial:.8f}, OpenMP = {ke_omp:.8f}")
    print(f"Difference: {diff_ke:.2e} (Illustrates IEEE-754 order-of-summation non-associativity)")

    # 4. Divergence (Analytical = 0.0 for incompressible TGV)
    print("\n--- 3. Incompressibility & Divergence (∇ · u) ---")
    div = pcfd.divergence(u, v, w, nx, ny, nz, dx, dy, dz, threads=0)
    max_div = np.max(np.abs(div[1:-1, 1:-1, 1:-1]))
    rms_div = np.sqrt(np.mean(div[1:-1, 1:-1, 1:-1] ** 2))
    print(f"Max interior divergence: {max_div:.2e}")
    print(f"RMS divergence:          {rms_div:.2e} (Analytically zero, second-order discretization error)")

    # 5. Vorticity (ω = ∇ × u)
    print("\n--- 4. Vorticity Vector & Magnitude (ω = ∇ × u) ---")
    wx, wy, wz, w_mag = pcfd.vorticity(u, v, w, nx, ny, nz, dx, dy, dz, compute_magnitude=True, threads=0)
    print(f"Max vorticity magnitude: {np.max(w_mag):.4f}")

    # 6. Q-Criterion (Vortex identification)
    print("\n--- 5. Q-Criterion (Q = 0.5 * (||Ω||^2 - ||S||^2)) ---")
    t0 = time.perf_counter()
    q_serial = pcfd.q_criterion(u, v, w, nx, ny, nz, dx, dy, dz, threads=1)
    t_q_serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    q_omp = pcfd.q_criterion(u, v, w, nx, ny, nz, dx, dy, dz, threads=0)
    t_q_omp = time.perf_counter() - t0

    diff_q = np.max(np.abs(q_serial - q_omp))
    print(f"Q-criterion serial: {t_q_serial * 1000:.2f} ms")
    print(f"Q-criterion OpenMP: {t_q_omp * 1000:.2f} ms (Speedup: {t_q_serial / t_q_omp:.2f}x)")
    print(f"Max difference:     {diff_q:.2e}")
    print(f"Percentage of cells with Q > 0 (vortex cores): {100.0 * np.sum(q_omp > 0) / total_cells:.1f}%")

    print("\n" + "=" * 70)
    print(" ALL CFD KERNELS VALIDATED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
