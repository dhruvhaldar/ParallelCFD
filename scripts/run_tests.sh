#!/usr/bin/env bash
# ==============================================================================
# ParallelCFD: Comprehensive Test Suite Runner
# Executes C++ Unit Tests, CTest Integration, and Python Pytest Suites
# ==============================================================================

set -e

# Color helpers
GREEN="\033[0;32m"
BLUE="\033[0;34m"
CYAN="\033[0;36m"
BOLD="\033[1m"
RESET="\033[0m"

echo -e "${BLUE}${BOLD}======================================================================${RESET}"
echo -e "${BLUE}${BOLD} ParallelCFD: Running Complete Test Suite${RESET}"
echo -e "${BLUE}${BOLD}======================================================================${RESET}"

# 1. Build tests
echo -e "\n${CYAN}[1/4] Ensuring all test targets are built...${RESET}"
cmake -B build -S . -DPython3_EXECUTABLE=$(pwd)/.venv/bin/python
cmake --build build -j$(nproc)

# 2. Run Individual C++ Test Executables
echo -e "\n${CYAN}[2/4] Running Individual C++ Unit Tests...${RESET}"

echo -e "\n--- Running test_velocity ---"
./build/test_velocity

echo -e "\n--- Running test_reductions ---"
./build/test_reductions

echo -e "\n--- Running test_derivatives ---"
./build/test_derivatives

echo -e "\n--- Running test_vorticity_qcriterion ---"
./build/test_vorticity_qcriterion

echo -e "\n--- Running test_experiments ---"
./build/test_experiments

echo -e "\n--- Running test_hybrid_mpi (2 MPI Ranks) ---"
mpirun -n 2 ./build/test_hybrid_mpi

# 3. Run CTest Automation
echo -e "\n${CYAN}[3/4] Running CTest Automated Test Runner...${RESET}"
ctest --test-dir build --output-on-failure

# 4. Run Python Pytest Suite
echo -e "\n${CYAN}[4/4] Running Python Pytest Suite...${RESET}"
.venv/bin/pytest tests/test_python_suite.py tests/test_python_api.py -v

echo -e "\n${GREEN}${BOLD}======================================================================${RESET}"
echo -e "${GREEN}${BOLD} ALL TESTS PASSED! FULL TEST SUITE VERIFIED SUCCESSFULLY!${RESET}"
echo -e "${GREEN}${BOLD}======================================================================${RESET}"
