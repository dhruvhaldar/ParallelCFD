import os
import pyvista as pv

def plot_q_criterion_isosurfaces(mesh, q_val=0.05, filename="results/q_criterion_isosurfaces.png",
                                 show=False, color_by="velocity_magnitude"):
    """
    Renders 3D isosurfaces of Q-criterion indicating coherent vortex cores.
    Colors the isosurface by velocity magnitude or vorticity.
    """
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)

    # Extract isosurface of Q > q_val
    contours = mesh.contour(isosurfaces=[q_val], scalars="q_criterion")

    # Set up PyVista plotter
    plotter = pv.Plotter(off_screen=not show)
    plotter.set_background("#1e1e2e")  # High contrast modern dark theme

    # Add outline bounding box of the domain
    plotter.add_mesh(mesh.outline(), color="white", line_width=1.5, opacity=0.4)

    # Add isosurface
    if contours.n_points > 0:
        plotter.add_mesh(
            contours,
            scalars=color_by,
            cmap="turbo",
            smooth_shading=True,
            specular=0.6,
            specular_power=30,
            scalar_bar_args={
                "title": f"Isosurface (Q={q_val:.2f}) colored by {color_by}",
                "color": "white",
                "vertical": True,
            }
        )

    # Add text banner
    plotter.add_text("ParallelCFD: OpenMP 3D Q-Criterion Vortex Core Identification",
                     position="upper_left", font_size=12, color="white")

    plotter.camera_position = [(3.5, 3.2, 3.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    if filename:
        plotter.screenshot(filename)
        print(f"[PyVista] Saved Q-criterion visualization to {filename}")

    if show:
        plotter.show()
    else:
        plotter.close()

    return filename

def plot_vorticity_slices(mesh, filename="results/vorticity_slices.png", show=False):
    """
    Renders orthogonal 3D slice planes of vorticity magnitude.
    """
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)

    slices = mesh.slice_orthogonal()

    plotter = pv.Plotter(off_screen=not show)
    plotter.set_background("#1a1a24")
    plotter.add_mesh(mesh.outline(), color="gray", opacity=0.3)
    plotter.add_mesh(slices, scalars="vorticity_magnitude", cmap="inferno",
                     scalar_bar_args={"title": "Vorticity Magnitude |ω|", "color": "white"})
    plotter.add_text("ParallelCFD: Vorticity Magnitude Orthogonal Slices",
                     position="upper_left", font_size=12, color="white")

    if filename:
        plotter.screenshot(filename)
        print(f"[PyVista] Saved vorticity slices to {filename}")

    if show:
        plotter.show()
    else:
        plotter.close()

    return filename
