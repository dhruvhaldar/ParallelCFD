import sys
import os
import pytest
import numpy as np

# Ensure parallelcfd is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../python")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))

import parallelcfd as pcfd

def test_velocity_magnitude():
    n = 50000
    u = np.full(n, 3.0)
    v = np.full(n, 4.0)
    w = np.zeros(n)

    mag_ser = pcfd.velocity_magnitude(u, v, w, threads=1)
    mag_omp = pcfd.velocity_magnitude(u, v, w, threads=4)
    mag_simd = pcfd.velocity_magnitude(u, v, w, threads=4, use_simd=True)

    np.testing.assert_allclose(mag_ser, 5.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(mag_omp, 5.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(mag_simd, 5.0, rtol=1e-12, atol=1e-12)

def test_reductions():
    n = 20000
    arr = np.linspace(1.0, 100.0, n)
    stats_ser = pcfd.field_stats(arr, threads=1)
    stats_omp = pcfd.field_stats(arr, threads=4)

    assert abs(stats_ser.min_val - 1.0) < 1e-10
    assert abs(stats_omp.min_val - 1.0) < 1e-10
    assert abs(stats_ser.max_val - 100.0) < 1e-10
    assert abs(stats_omp.max_val - 100.0) < 1e-10
    assert abs(stats_ser.mean_val - stats_omp.mean_val) < 1e-10

def test_divergence_and_vorticity_solid_body():
    nx, ny, nz = 32, 32, 32
    dx = dy = dz = 0.1
    Omega = 2.5
    data = pcfd.generate_solid_body(nx, ny, nz, dx, dy, dz, Omega=Omega)
    u, v, w = data['u'], data['v'], data['w']

    # Divergence should be identically 0
    div = pcfd.divergence(u, v, w, nx, ny, nz, dx, dy, dz, threads=4)
    np.testing.assert_allclose(div, 0.0, atol=1e-10)

    # Vorticity: curl = (0, 0, 2*Omega)
    wx, wy, wz, wmag = pcfd.vorticity(u, v, w, nx, ny, nz, dx, dy, dz, compute_magnitude=True, threads=4)
    # Check interior
    np.testing.assert_allclose(wx[2:-2, 2:-2, 2:-2], 0.0, atol=1e-10)
    np.testing.assert_allclose(wy[2:-2, 2:-2, 2:-2], 0.0, atol=1e-10)
    np.testing.assert_allclose(wz[2:-2, 2:-2, 2:-2], 2.0 * Omega, atol=1e-10)
    np.testing.assert_allclose(wmag[2:-2, 2:-2, 2:-2], 2.0 * Omega, atol=1e-10)

def test_q_criterion_solid_body():
    nx, ny, nz = 32, 32, 32
    dx = dy = dz = 0.1
    Omega = 3.0
    data = pcfd.generate_solid_body(nx, ny, nz, dx, dy, dz, Omega=Omega)
    u, v, w = data['u'], data['v'], data['w']

    q_ser = pcfd.q_criterion(u, v, w, nx, ny, nz, dx, dy, dz, threads=1)
    q_omp = pcfd.q_criterion(u, v, w, nx, ny, nz, dx, dy, dz, threads=4)

    expected_q = Omega * Omega # 9.0
    np.testing.assert_allclose(q_ser[2:-2, 2:-2, 2:-2], expected_q, atol=1e-10)
    np.testing.assert_allclose(q_omp[2:-2, 2:-2, 2:-2], expected_q, atol=1e-10)
    np.testing.assert_allclose(q_ser, q_omp, rtol=1e-11, atol=1e-11)

def test_thread_scaling_consistency():
    nx, ny, nz = 32, 32, 32
    dx = dy = dz = 0.05
    data = pcfd.generate_taylor_green(nx, ny, nz, dx, dy, dz)
    u, v, w = data['u'], data['v'], data['w']

    # Compare 1 thread vs 2, 4, 8 threads
    q_ref = pcfd.q_criterion(u, v, w, nx, ny, nz, dx, dy, dz, threads=1)
    for t in [2, 4, 8]:
        q_t = pcfd.q_criterion(u, v, w, nx, ny, nz, dx, dy, dz, threads=t)
        np.testing.assert_allclose(q_ref, q_t, rtol=1e-11, atol=1e-11)

def test_pyvista_mesh_creation():
    mesh = pcfd.create_taylor_green_mesh(nx=16, ny=16, nz=16)
    assert "velocity" in mesh.point_data
    assert "vorticity" in mesh.point_data
    assert "q_criterion" in mesh.point_data
    assert mesh.n_points == 16 * 16 * 16
