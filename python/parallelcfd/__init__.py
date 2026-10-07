"""
ParallelCFD: OpenMP-Accelerated CFD Field Analysis and Post-Processing Toolkit
"""

import sys
from _parallelcfd import (
    Grid3D,
    FieldStats,
    velocity_magnitude,
    kinetic_energy,
    field_stats,
    gradients,
    divergence,
    vorticity,
    q_criterion,
    generate_taylor_green,
    generate_solid_body,
    run_false_sharing_experiment,
    run_race_condition_experiment,
    test_floating_point_nonassociativity,
)

from .grid import create_taylor_green_mesh, create_solid_body_mesh
from .visualizer import plot_q_criterion_isosurfaces, plot_vorticity_slices

__version__ = "1.0.0"
__all__ = [
    "Grid3D",
    "FieldStats",
    "velocity_magnitude",
    "kinetic_energy",
    "field_stats",
    "gradients",
    "divergence",
    "vorticity",
    "q_criterion",
    "generate_taylor_green",
    "generate_solid_body",
    "run_false_sharing_experiment",
    "run_race_condition_experiment",
    "test_floating_point_nonassociativity",
    "create_taylor_green_mesh",
    "create_solid_body_mesh",
    "plot_q_criterion_isosurfaces",
    "plot_vorticity_slices",
]
