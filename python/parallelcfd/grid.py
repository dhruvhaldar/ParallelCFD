import numpy as np
import pyvista as pv
from _parallelcfd import generate_taylor_green, generate_solid_body, velocity_magnitude, vorticity, q_criterion

def create_taylor_green_mesh(nx=64, ny=64, nz=64, L=np.pi, U0=1.0, threads=0):
    """
    Creates a PyVista ImageData structured grid for a 3D Taylor-Green vortex.
    Populates mesh with velocity, pressure, velocity magnitude, vorticity, and Q-criterion.
    """
    dx = 2.0 * L / nx
    dy = 2.0 * L / ny
    dz = 2.0 * L / nz

    data = generate_taylor_green(nx, ny, nz, dx, dy, dz, U0=U0, L=1.0)
    u = data['u']
    v = data['v']
    w = data['w']
    p = data.get('pressure', data.get('p'))

    # Compute derived CFD fields in C++ with OpenMP
    mag = velocity_magnitude(u, v, w, threads=threads)
    wx, wy, wz, v_mag = vorticity(u, v, w, nx, ny, nz, dx, dy, dz, compute_magnitude=True, threads=threads)
    q = q_criterion(u, v, w, nx, ny, nz, dx, dy, dz, threads=threads)

    # PyVista ImageData (uniform structured grid)
    mesh = pv.ImageData()
    mesh.dimensions = (nx, ny, nz)
    mesh.spacing = (dx, dy, dz)
    mesh.origin = (-L, -L, -L)

    # Flatten in Fortran/C order matching PyVista ImageData point ordering
    # PyVista points are ordered with X fastest, then Y, then Z.
    # In our C-contiguous array (nx, ny, nz): axis 0 is X, axis 1 is Y, axis 2 is Z.
    # We transpose to (nz, ny, nx) or flatten in C order as needed:
    # VTK ImageData expects X changing fastest (index = k*ny*nx + j*nx + i).
    # Since our array index is (i*ny + j)*nz + k, we swap axes (2, 1, 0) so flattened order matches VTK:
    def to_vtk(arr):
        return np.ascontiguousarray(np.transpose(arr, (2, 1, 0))).ravel()

    vel_vectors = np.column_stack([to_vtk(u), to_vtk(v), to_vtk(w)])
    vort_vectors = np.column_stack([to_vtk(wx), to_vtk(wy), to_vtk(wz)])

    mesh.point_data["velocity"] = vel_vectors
    mesh.point_data["velocity_magnitude"] = to_vtk(mag)
    mesh.point_data["pressure"] = to_vtk(p)
    mesh.point_data["vorticity"] = vort_vectors
    mesh.point_data["vorticity_magnitude"] = to_vtk(v_mag)
    mesh.point_data["q_criterion"] = to_vtk(q)

    return mesh

def create_solid_body_mesh(nx=64, ny=64, nz=64, R=1.0, Omega=2.0, threads=0):
    """
    Creates a PyVista ImageData structured grid for solid-body rotation.
    """
    dx = 2.0 * R / nx
    dy = 2.0 * R / ny
    dz = 2.0 * R / nz

    data = generate_solid_body(nx, ny, nz, dx, dy, dz, Omega=Omega)
    u = data['u']
    v = data['v']
    w = data['w']
    p = data.get('pressure', data.get('p'))

    mag = velocity_magnitude(u, v, w, threads=threads)
    wx, wy, wz, v_mag = vorticity(u, v, w, nx, ny, nz, dx, dy, dz, compute_magnitude=True, threads=threads)
    q = q_criterion(u, v, w, nx, ny, nz, dx, dy, dz, threads=threads)

    mesh = pv.ImageData()
    mesh.dimensions = (nx, ny, nz)
    mesh.spacing = (dx, dy, dz)
    mesh.origin = (-R, -R, -R)

    def to_vtk(arr):
        return np.ascontiguousarray(np.transpose(arr, (2, 1, 0))).ravel()

    mesh.point_data["velocity"] = np.column_stack([to_vtk(u), to_vtk(v), to_vtk(w)])
    mesh.point_data["velocity_magnitude"] = to_vtk(mag)
    mesh.point_data["pressure"] = to_vtk(p)
    mesh.point_data["vorticity"] = np.column_stack([to_vtk(wx), to_vtk(wy), to_vtk(wz)])
    mesh.point_data["vorticity_magnitude"] = to_vtk(v_mag)
    mesh.point_data["q_criterion"] = to_vtk(q)

    return mesh
