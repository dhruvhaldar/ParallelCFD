from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import subprocess
import os

ext_modules = [
    Pybind11Extension(
        "_parallelcfd",
        [
            "python/bindings.cpp",
            "src/grid.cpp",
            "src/velocity.cpp",
            "src/reductions.cpp",
            "src/gradients.cpp",
            "src/divergence.cpp",
            "src/vorticity.cpp",
            "src/qcriterion.cpp",
            "src/experiments.cpp",
        ],
        include_dirs=["include"],
        extra_compile_args=["-O3", "-march=native", "-fopenmp", "-std=c++20"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="parallelcfd",
    version="1.0.0",
    author="Dhruv Haldar",
    description="OpenMP-Accelerated CFD Field Analysis and Post-Processing Toolkit",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20",
        "pyvista>=0.38",
        "matplotlib>=3.5",
    ],
)
