cmake_minimum_required(VERSION 3.20)
project(ParallelCFD VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization Flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native -Wall -Wextra -Wno-unused-parameter)
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find MPI
find_package(MPI REQUIRED)

# Include Directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Core CFD shared library
add_library(cfd_core SHARED
    src/grid.cpp
    src/velocity.cpp
    src/reductions.cpp
    src/gradients.cpp
    src/divergence.cpp
    src/vorticity.cpp
    src/qcriterion.cpp
    src/experiments.cpp
)
target_link_libraries(cfd_core PUBLIC OpenMP::OpenMP_CXX)
target_include_directories(cfd_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Hybrid MPI library
add_library(cfd_hybrid STATIC
    src/hybrid_mpi.cpp
)
target_link_libraries(cfd_hybrid PUBLIC cfd_core MPI::MPI_CXX OpenMP::OpenMP_CXX)

# Executables: Benchmarks
add_executable(benchmark_cfd_kernels benchmarks/benchmark_cfd_kernels.cpp)
target_link_libraries(benchmark_cfd_kernels PRIVATE cfd_core OpenMP::OpenMP_CXX)

add_executable(benchmark_experiments benchmarks/benchmark_experiments.cpp)
target_link_libraries(benchmark_experiments PRIVATE cfd_core OpenMP::OpenMP_CXX)

add_executable(benchmark_hybrid benchmarks/benchmark_hybrid.cpp)
target_link_libraries(benchmark_hybrid PRIVATE cfd_hybrid MPI::MPI_CXX OpenMP::OpenMP_CXX)

# Executables: Unit Tests
add_executable(test_velocity tests/test_velocity.cpp)
target_link_libraries(test_velocity PRIVATE cfd_core OpenMP::OpenMP_CXX)

add_executable(test_reductions tests/test_reductions.cpp)
target_link_libraries(test_reductions PRIVATE cfd_core OpenMP::OpenMP_CXX)

add_executable(test_derivatives tests/test_derivatives.cpp)
target_link_libraries(test_derivatives PRIVATE cfd_core OpenMP::OpenMP_CXX)

add_executable(test_vorticity_qcriterion tests/test_vorticity_qcriterion.cpp)
target_link_libraries(test_vorticity_qcriterion PRIVATE cfd_core OpenMP::OpenMP_CXX)

add_executable(test_experiments tests/test_experiments.cpp)
target_link_libraries(test_experiments PRIVATE cfd_core OpenMP::OpenMP_CXX)

add_executable(test_hybrid_mpi tests/test_hybrid_mpi.cpp)
target_link_libraries(test_hybrid_mpi PRIVATE cfd_hybrid MPI::MPI_CXX OpenMP::OpenMP_CXX)

# Python Bindings (pybind11)
find_package(Python3 COMPONENTS Interpreter Development)
if(Python3_FOUND)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(pybind11_DIR)
            find_package(pybind11 CONFIG PATHS ${pybind11_DIR})
        endif()
    endif()

    if(pybind11_FOUND)
        pybind11_add_module(_parallelcfd python/bindings.cpp)
        target_link_libraries(_parallelcfd PRIVATE cfd_core OpenMP::OpenMP_CXX)
        set_target_properties(_parallelcfd PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python"
        )
    else()
        message(WARNING "pybind11 not found! Python bindings will not be built.")
    endif()
endif()

# CTest Integration
enable_testing()
add_test(NAME TestVelocity COMMAND test_velocity)
add_test(NAME TestReductions COMMAND test_reductions)
add_test(NAME TestDerivatives COMMAND test_derivatives)
add_test(NAME TestVorticityQCriterion COMMAND test_vorticity_qcriterion)
add_test(NAME TestExperiments COMMAND test_experiments)
add_test(NAME TestHybridMPI COMMAND mpirun -n 2 ./test_hybrid_mpi)
if(Python3_FOUND)
    add_test(NAME TestPythonSuite COMMAND "${Python3_EXECUTABLE}" -m pytest "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_python_suite.py" -v)
endif()
