#!/usr/bin/env python3
"""
Example 02: PyVista 3D Visualization of OpenMP-Processed CFD Field
Renders 3D Q-Criterion vortex cores and orthogonal slices of vorticity.
"""

import os
import sys

try:
    import parallelcfd as pcfd
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../build")))
    import parallelcfd as pcfd

def main():
    print("=" * 70)
    print(" ParallelCFD: PyVista 3D Vortex Core & Field Visualization")
    print("=" * 70)

    nx, ny, nz = 96, 96, 96
    print(f"Creating 3D Taylor-Green vortex mesh ({nx}x{ny}x{nz})...")
    mesh = pcfd.create_taylor_green_mesh(nx=nx, ny=ny, nz=nz, threads=0)

    os.makedirs("results", exist_ok=True)

    # 1. Plot Q-criterion 3D vortex cores
    q_file = "results/pyvista_q_criterion_3d.png"
    print(f"Rendering 3D Q-criterion isosurfaces to {q_file}...")
    pcfd.plot_q_criterion_isosurfaces(
        mesh,
        q_val=0.08,
        filename=q_file,
        show=False,
        color_by="velocity_magnitude"
    )

    # 2. Plot vorticity slices
    vort_file = "results/pyvista_vorticity_slices.png"
    print(f"Rendering vorticity slices to {vort_file}...")
    pcfd.plot_vorticity_slices(
        mesh,
        filename=vort_file,
        show=False
    )

    print("\nVisualizations successfully generated and saved to results/ directory!")

if __name__ == "__main__":
    main()
