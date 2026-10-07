# ParallelCFD: System Architecture & Technical Design

**ParallelCFD** is a high-performance C++20 and OpenMP post-processing toolkit engineered for large-scale Computational Fluid Dynamics (CFD) datasets. This document details the software architecture, memory hierarchies, stencil discretization schemes, hybrid domain decomposition, and zero-copy Python interoperability.

---

## 1. High-Level System Architecture

The toolkit follows a layered architecture connecting high-level Python visualization with bare-metal C++20 multi-threaded kernels:

```mermaid
flowchart TD
    subgraph UI_Layer["Visualization & Analysis Layer (Python / PyVista)"]
        PV["PyVista 3D Engine (VTK ImageData)"]
        Script["Analysis Scripts (Taylor-Green, Vorticity Slices, Q-Cores)"]
        Matplotlib["Matplotlib Scaling & Profiling Visualizer"]
    end

    subgraph Binding_Layer["Bridge Layer (pybind11)"]
        ZeroCopy["Zero-Copy NumPy Buffer Protocol (C-Contiguous)"]
        GIL["GIL Management (py::gil_scoped_release)"]
        PyModule["Python Module: _parallelcfd"]
    end

    subgraph Core_Layer["Compute Backend (C++20 Shared Library: libcfd_core)"]
        Grid["StructuredGrid3D (Memory Strides & Coordinate Mapping)"]
        VelKernel["Velocity Kernels (Serial, OpenMP SIMD, SoA / AoS)"]
        ReductKernel["Reduction Kernels (min, max, mean, RMS, Kinetic Energy)"]
        DiffKernel["Finite Difference Stencil Kernels (2nd-Order Central & Boundary)"]
        TensorKernel["Tensor Kernels (Vorticity, Rate of Strain, Q-Criterion)"]
        ExpKernel["Hardware Probes (False Sharing, Race Condition, Schedulers)"]
    end

    subgraph Distributed_Layer["Distributed Backend (MPI + OpenMP: libcfd_hybrid)"]
        Scatterv["Domain Partitioning (MPI_Scatterv)"]
        Halo["Halo Ghost-Cell Exchange (MPI_Sendrecv)"]
        LocalOMP["Local Multi-Threaded Processing (OpenMP Threads)"]
        Gatherv["Domain Reconstruction (MPI_Gatherv)"]
    end

    Script --> ZeroCopy
    PV --> ZeroCopy
    ZeroCopy --> GIL
    GIL --> PyModule
    PyModule --> Grid
    Grid --> VelKernel
    Grid --> ReductKernel
    Grid --> DiffKernel
    Grid --> TensorKernel
    Grid --> ExpKernel

    Core_Layer -.-> LocalOMP
    Scatterv --> Halo
    Halo --> LocalOMP
    LocalOMP --> Gatherv
```

---

## 2. End-to-End Processing & Mathematical Pipeline

The post-processing pipeline transforms raw Cartesian flow fields into tensor invariants and vortex structures:

```mermaid
flowchart LR
    In["Input CFD Field (u, v, w, p)"] --> Pre["Grid Geometry (dx, dy, dz)"]
    
    Pre --> Mag["Velocity Magnitude |u| = sqrt(u² + v² + w²)"]
    Pre --> Red["Global Reductions (p_min, p_max, K_total, RMS)"]
    Pre --> Grad["Velocity Gradient Tensor J = ∇u"]

    Grad --> Symm["Rate of Strain Tensor: S = 0.5 * (J + Jᵀ)"]
    Grad --> Anti["Rate of Rotation Tensor: Ω = 0.5 * (J - Jᵀ)"]

    Symm --> NormS["||S||² = tr(S · Sᵀ)"]
    Anti --> NormO["||Ω||² = tr(Ω · Ωᵀ)"]

    NormS --> Q["Q-Criterion: Q = 0.5 * (||Ω||² - ||S||²)"]
    NormO --> Q

    Grad --> Vort["Vorticity Vector: ω = ∇ × u"]
    Grad --> Div["Divergence: ∇ · u = tr(J)"]

    Q --> Vis["PyVista 3D Contouring (Isosurface Q > 0)"]
    Vort --> Vis
```

---

## 3. Memory Layout & Cache Hierarchy Optimization

### 3.1. Row-Major Stride-1 3D Storage

All 3D fields ($N_x \times N_y \times N_z$) are stored in continuous, flat memory buffers using C-contiguous row-major order:

$$\text{Index}(i, j, k) = (i \cdot N_y + j) \cdot N_z + k$$

$$\text{Stride}_x = N_y \cdot N_z, \quad \text{Stride}_y = N_z, \quad \text{Stride}_z = 1$$

This ensures that the innermost loop index ($k$) traverses consecutive memory addresses with unit stride ($\Delta = 8$ bytes for double precision), maximizing hardware prefetcher efficiency and enabling AVX2/AVX-512 SIMD vectorization.

```mermaid
flowchart TD
    subgraph Memory_Buffer["Contiguous Memory Buffer (Row-Major)"]
        direction LR
        Cell0["(0,0,0)"] --> Cell1["(0,0,1)"] --> Cell2["(0,0,2)"] --> Ellipsis["..."] --> CellZ["(0,0,Nz-1)"] --> CellNext["(0,1,0)"]
    end

    subgraph CPU_Cache["CPU L1 Data Cache Line (64 Bytes)"]
        direction LR
        D0["Double 0 (8B)"]
        D1["Double 1 (8B)"]
        D2["Double 2 (8B)"]
        D3["Double 3 (8B)"]
        D4["Double 4 (8B)"]
        D5["Double 5 (8B)"]
        D6["Double 6 (8B)"]
        D7["Double 7 (8B)"]
    end

    Cell0 -.-> D0
    Cell1 -.-> D1
    Cell2 -.-> D2
```

---

### 3.2. Structure of Arrays (SoA) vs. Array of Structures (AoS)

```mermaid
graph TB
    subgraph AoS["Array of Structures (Interleaved Storage)"]
        direction LR
        A1["u0, v0, w0"] --- A2["u1, v1, w1"] --- A3["u2, v2, w2"] --- A4["u3, v3, w3"]
    end

    subgraph SoA["Structure of Arrays (Contiguous Component Storage)"]
        direction TB
        S_u["u: [u0, u1, u2, u3, u4, u5, u6, u7, ...]"]
        S_v["v: [v0, v1, v2, v3, v4, v5, v6, v7, ...]"]
        S_w["w: [w0, w1, w2, w3, w4, w5, w6, w7, ...]"]
    end

    subgraph SIMD_Reg["256-bit SIMD Vector Register (AVX2: 4 x double)"]
        R1["u0 | u1 | u2 | u3"]
    end

    S_u -->|"Single 256-bit aligned load (_mm256_load_pd)"| SIMD_Reg
    AoS -->|"Requires strided gather or register shuffles"| SIMD_Reg
```

- **AoS Limitation:** Accessing only $u$ requires striding over $v$ and $w$ ($24$ bytes per step). Loading into 256-bit SIMD registers requires cross-lane shuffle or gather instructions.
- **SoA Advantage:** $u$, $v$, and $w$ are completely separate contiguous vectors. Four consecutive values load directly into AVX2 registers with zero permutation overhead.

---

### 3.3. Cache-Line False Sharing & Hardware Padding

```mermaid
graph TB
    subgraph Unpadded["Unpadded Counters: False Sharing (Cache Thrashing)"]
        CL_Unpadded["64-Byte Cache Line: [Thread 0 (8B) | Thread 1 (8B) | Thread 2 (8B) | ... | Thread 7 (8B)]"]
        C0["Core 0 (Thread 0 Writes)"] -->|"Invalidates Entire Line"| CL_Unpadded
        C1["Core 1 (Thread 1 Writes)"] -->|"MESI Protocol Invalidation"| CL_Unpadded
    end

    subgraph Padded["Padded Counters: alignas(64) (Independent Cache Lines)"]
        CL0["Cache Line 0 (64B): [Thread 0 Counter (8B) + 56B Padding]"]
        CL1["Cache Line 1 (64B): [Thread 1 Counter (8B) + 56B Padding]"]
        C0_Pad["Core 0"] --> CL0
        C1_Pad["Core 1"] --> CL1
    end
```

- In the unpadded layout, writes to adjacent thread counters trigger the **MESI cache-coherence protocol**, bouncing the cache line between cores and stalling the pipeline.
- Using `struct alignas(64) PaddedCounter { uint64_t val; char pad[56]; };` ensures each counter resides on a distinct cache line, delivering an empirical **$1.8\times$ speedup**.

---

## 4. Loop Nesting & OpenMP Collapse Optimization

For a 3D finite-difference loop over $N_x \times N_y \times N_z$:

```cpp
#pragma omp parallel for collapse(2) schedule(static)
for (size_t i = 0; i < nx; ++i) {
    for (size_t j = 0; j < ny; ++j) {
        #pragma omp simd
        for (size_t k = 0; k < nz; ++k) {
            // Stencil evaluations
        }
    }
}
```

```mermaid
flowchart TD
    subgraph Loop_Hierarchy["3D Loop Optimization Strategy"]
        L_i["Outer Loop (i): X-dimension (Coarse Partitioning)"]
        L_j["Middle Loop (j): Y-dimension (Fine Partitioning)"]
        L_k["Inner Loop (k): Z-dimension (Unit-Stride Memory Access)"]

        Collapse["collapse(2): Merges (i, j) into Nx * Ny Iterations"]
        SIMD["#pragma omp simd: Vectorizes Inner Loop (k)"]
    end

    L_i --> Collapse
    L_j --> Collapse
    Collapse -->|"Distributes across OpenMP Threads"| Threads["32 Logical Cores"]
    L_k --> SIMD
    SIMD -->|"Streams into AVX2 / FMA Vector Units"| Registers["4-wide Double Vectors"]
```

- **`collapse(1)`:** Dispatches only $N_x$ chunks. If $N_x = 128$ and $p = 32$, each thread gets only 4 planes, causing potential imbalance.
- **`collapse(2)`:** Combines $(i, j)$ into $N_x \times N_y = 16,384$ iterations, distributing work evenly across all threads while keeping the innermost contiguous $k$-loop intact for SIMD vectorization.
- **`collapse(3)`:** Flattens all three loops, calculating $(i, j, k)$ via division/modulo or linear indices, which adds index calculation overhead and disrupts hardware prefetchers.

---

## 5. Hybrid MPI + OpenMP Architecture

### 5.1. Domain Decomposition & Halo Exchange

```mermaid
flowchart TD
    Global["Global 3D Domain (Nx × Ny × Nz)"] --> Decomp["1D Slab Decomposition along X-Axis"]

    subgraph Rank0["MPI Rank 0 (Core 0-3)"]
        Ghost0R["Right Halo (1 Plane)"]
        Local0["Local Domain 0 (Nx/P Planes)"]
        Threads0["OpenMP Threads (0, 1, 2, 3)"]
    end

    subgraph Rank1["MPI Rank 1 (Core 4-7)"]
        Ghost1L["Left Halo (1 Plane)"]
        Local1["Local Domain 1 (Nx/P Planes)"]
        Ghost1R["Right Halo (1 Plane)"]
        Threads1["OpenMP Threads (0, 1, 2, 3)"]
    end

    subgraph Rank2["MPI Rank 2 (Core 8-11)"]
        Ghost2L["Left Halo (1 Plane)"]
        Local2["Local Domain 2 (Nx/P Planes)"]
        Threads2["OpenMP Threads (0, 1, 2, 3)"]
    end

    Decomp --> Local0
    Decomp --> Local1
    Decomp --> Local2

    Local0 -.->|"MPI_Sendrecv"| Ghost1L
    Local1 -.->|"MPI_Sendrecv"| Ghost0R
    Local1 -.->|"MPI_Sendrecv"| Ghost2L
    Local2 -.->|"MPI_Sendrecv"| Ghost1R

    Local0 --> Threads0
    Local1 --> Threads1
    Local2 --> Threads2
```

---

### 5.2. Communication Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor Driver as Main Process
    participant R0 as MPI Rank 0 (Root)
    participant R1 as MPI Rank 1
    participant OMP as OpenMP Worker Threads

    Driver->>R0: Generate / Read Global Field
    Note over R0: Partition Nx into P Slabs
    R0->>R1: MPI_Scatterv (Local Interior Cells)
    R0->>R0: Copy Local Interior Slabs

    par Halo Boundary Exchange
        R0->>R1: MPI_Sendrecv (Send Plane Nx/P-1 -> Receive Ghost 0)
        R1->>R0: MPI_Sendrecv (Send Plane 0 -> Receive Ghost Nx/P)
    end

    Note over R0,R1: Subdomains fully populated with halos

    par Multi-Threaded Local Compute
        R0->>OMP: Execute OpenMP Stencils on Local Slab + Halos
        R1->>OMP: Execute OpenMP Stencils on Local Slab + Halos
    end

    OMP-->>R0: Local Q-Criterion Result (Interior Only)
    OMP-->>R1: Local Q-Criterion Result (Interior Only)

    R1->>R0: MPI_Gatherv (Interior Arrays)
    R0->>R0: Assemble Global Output Field
    R0->>Driver: Return Consolidated 3D Field
```

---

## 6. Zero-Copy Python / C++ Interoperability

The Python interface uses **pybind11** with zero-copy memory mapping:

```mermaid
flowchart LR
    subgraph Python_Env["Python Runtime (PyVista / NumPy)"]
        NP_Array["NumPy Array (C-Contiguous float64)"]
        PyGIL["Global Interpreter Lock (GIL)"]
    end

    subgraph PyBind["pybind11 Buffer Protocol"]
        Buffer["py::buffer_info (ptr, shape, strides)"]
        Unlock["py::gil_scoped_release"]
    end

    subgraph Cpp_Env["C++20 OpenMP Engine"]
        RawPtr["const double* data_ptr"]
        OpenMP["OpenMP Parallel Region (32 Threads)"]
    end

    NP_Array --> Buffer
    Buffer --> RawPtr
    RawPtr --> Unlock
    Unlock -->|"GIL Released: Zero Python Contention"| OpenMP
    OpenMP -->|"Writes to Pre-allocated NumPy Array"| NP_Array
```

1. **Zero-Copy Access:** `py::array_t<double>::request()` extracts the raw memory pointer (`ptr`), dimensions, and strides directly from the NumPy array without copying data.
2. **GIL Release:** Wrapping C++ compute loops in `py::gil_scoped_release` frees the Python interpreter lock. OpenMP worker threads execute with 100% CPU utilization across all cores without thread contention from Python's garbage collector.

---

## 7. Further Reading for Python Engineers

For senior Python developers seeking an in-depth breakdown comparing CPython internals, the GIL, NumPy expression evaluation, and hardware memory mechanics to C++ and OpenMP, refer to [**`PYTHON_HPC_GUIDE.md`**](PYTHON_HPC_GUIDE.md).
