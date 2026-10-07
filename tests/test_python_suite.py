import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../python")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))

import parallelcfd as pcfd

@pytest.mark.parametrize("threads", [1, 2, 4, 8])
@pytest.mark.parametrize("use_simd", [False, True])
def test_velocity_magnitude(threads, use_simd):
    n = 20000
    u = np.full(n, 3.0, dtype=np.float64)
    v = np.full(n, 4.0, dtype=np.float64)
    w = np.zeros(n, dtype=np.float64)

    mag = pcfd.velocity_magnitude(u, v, w, threads=threads, use_simd=use_simd)
    np.testing.assert_allclose(mag, 5.0, rtol=1e-13, atol=1e-13)

def test_reductions_against_numpy():
    arr = np.linspace(-50.0, 150.0, 100000, dtype=np.float64)
    stats = pcfd.field_stats(arr, threads=4)

    assert stats.min_val == pytest.approx(np.min(arr), rel=1e-12)
    assert stats.max_val == pytest.approx(np.max(arr), rel=1e-12)
    assert stats.mean_val == pytest.approx(np.mean(arr), rel=1e-12)
    assert stats.rms_val == pytest.approx(np.sqrt(np.mean(arr ** 2)), rel=1e-12)

@pytest.mark.parametrize("collapse", [1, 2, 3])
def test_gradients_quadratic_polynomial(collapse):
    nx, ny, nz = 24, 24, 24
    dx = dy = dz = 0.1
    x = np.arange(nx) * dx - 1.0
    y = np.arange(ny) * dy - 1.0
    z = np.arange(nz) * dz - 1.0
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # phi = 2*x^2 + 3*y^2 + 4*z^2
    phi = 2.0 * X**2 + 3.0 * Y**2 + 4.0 * Z**2
    gx, gy, gz = pcfd.gradients(phi, nx, ny, nz, dx, dy, dz, collapse_level=collapse, threads=4)

    # In the interior (away from boundary stencils), central difference is exact for quadratic polynomials
    exact_gx = 4.0 * X
    exact_gy = 6.0 * Y
    exact_gz = 8.0 * Z

    np.testing.assert_allclose(gx[2:-2, 2:-2, 2:-2], exact_gx[2:-2, 2:-2, 2:-2], atol=1e-12)
    np.testing.assert_allclose(gy[2:-2, 2:-2, 2:-2], exact_gy[2:-2, 2:-2, 2:-2], atol=1e-12)
    np.testing.assert_allclose(gz[2:-2, 2:-2, 2:-2], exact_gz[2:-2, 2:-2, 2:-2], atol=1e-12)

def test_divergence_incompressible_field():
    nx, ny, nz = 32, 32, 32
    dx = dy = dz = 0.05
    data = pcfd.generate_taylor_green(nx, ny, nz, dx, dy, dz)
    u, v, w = data['u'], data['v'], data['w']

    div = pcfd.divergence(u, v, w, nx, ny, nz, dx, dy, dz, threads=4)
    # Interior divergence must be 0 to machine precision
    np.testing.assert_allclose(div[2:-2, 2:-2, 2:-2], 0.0, atol=1e-12)

def test_vorticity_and_q_criterion_solid_body():
    nx, ny, nz = 24, 24, 24
    dx = dy = dz = 0.1
    Omega = 3.5
    data = pcfd.generate_solid_body(nx, ny, nz, dx, dy, dz, Omega=Omega)
    u, v, w = data['u'], data['v'], data['w']

    wx, wy, wz, wmag = pcfd.vorticity(u, v, w, nx, ny, nz, dx, dy, dz, compute_magnitude=True, threads=4)
    q = pcfd.q_criterion(u, v, w, nx, ny, nz, dx, dy, dz, threads=4)

    # Curl u = (0, 0, 2*Omega)
    np.testing.assert_allclose(wx[2:-2, 2:-2, 2:-2], 0.0, atol=1e-12)
    np.testing.assert_allclose(wy[2:-2, 2:-2, 2:-2], 0.0, atol=1e-12)
    np.testing.assert_allclose(wz[2:-2, 2:-2, 2:-2], 2.0 * Omega, atol=1e-12)
    np.testing.assert_allclose(wmag[2:-2, 2:-2, 2:-2], 2.0 * Omega, atol=1e-12)

    # Q = Omega^2
    np.testing.assert_allclose(q[2:-2, 2:-2, 2:-2], Omega**2, atol=1e-12)

def test_couette_flow_q_criterion_zero():
    """Pure shear Couette flow has non-zero vorticity but exactly Q = 0."""
    nx, ny, nz = 16, 16, 16
    dx = dy = dz = 0.1
    y = np.linspace(0, 1, ny)
    u = np.zeros((nx, ny, nz), dtype=np.float64)
    for j in range(ny):
        u[:, j, :] = 4.0 * y[j] # u = gamma * y
    v = np.zeros_like(u)
    w = np.zeros_like(u)

    q = pcfd.q_criterion(u, v, w, nx, ny, nz, dx, dy, dz, threads=2)
    np.testing.assert_allclose(q[2:-2, 2:-2, 2:-2], 0.0, atol=1e-12)

def test_input_validation_errors():
    """Tests that mismatched array dimensions raise RuntimeError."""
    u = np.ones((10, 10, 10))
    v = np.ones((10, 10, 5)) # Mismatched size
    w = np.ones((10, 10, 10))
    with pytest.raises(RuntimeError):
        pcfd.velocity_magnitude(u, v, w)

def test_pyvista_mesh_generation():
    mesh = pcfd.create_taylor_green_mesh(nx=16, ny=16, nz=16)
    assert mesh.n_points == 16 * 16 * 16
    assert "velocity" in mesh.point_data
    assert "vorticity" in mesh.point_data
    assert "q_criterion" in mesh.point_data
    assert mesh.point_data["velocity"].shape == (16*16*16, 3)

def test_fp_nonassociativity():
    res = pcfd.test_floating_point_nonassociativity(n=500000, threads=4)
    assert "serial_sum" in res
    assert "parallel_sum" in res
    assert res["rel_diff"] < 1e-10
