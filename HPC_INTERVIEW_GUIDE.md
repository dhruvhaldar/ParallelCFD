# ParallelCFD: High-Performance Computing Technical Interview Guide

**Candidate Focus:** HPC Software Engineer / Scientific Software Developer / CFD Specialist  
**Core Technologies:** C++20, OpenMP, MPI, SIMD Vectorization, Hardware Architecture, Python/PyVista  

---

## 1. Executive Talking Track

> *"I previously developed distributed-memory scientific workflows using Python, mpi4py, Intel MPI, and OpenMPI—focusing on collective operations such as `Scatter`, `Scatterv`, and `Gather`. To build a complete parallel computing profile, I developed **ParallelCFD** to extend that experience into shared-memory multi-threading with OpenMP in C++20.*  
>  
> *Rather than a generic matrix tutorial, I built high-performance post-processing kernels for 3D CFD datasets: velocity gradients, divergence, vorticity, and the tensor-based Q-criterion. I benchmarked strong scaling up to 32 threads, investigated memory bandwidth saturation using the Roofline model, diagnosed false sharing on 64-byte cache lines, and implemented a hybrid MPI + OpenMP domain decomposition. The kernels integrate directly with PyVista for 3D vortex visualization."*

---

## 2. Core Technical Questions & Model Answers

### Question 1: When should an engineer use MPI, OpenMP, or a Hybrid MPI+OpenMP model?
**Model Answer:**
- **MPI (Message Passing Interface)** is designed for **distributed-memory** systems across separate physical nodes, cluster interconnects (InfiniBand/Omni-Path), or independent address spaces. It requires explicit data communication and halo exchange (`MPI_Sendrecv`, `MPI_Scatterv`, `MPI_Gatherv`).
- **OpenMP** is designed for **shared-memory** systems within a single node or NUMA socket. Threads operate on the same virtual address space, eliminating duplicate memory allocation and message packing overhead.
- **Hybrid MPI + OpenMP** is essential on modern supercomputers where single compute nodes contain 64 to 256 CPU cores. Running 256 pure MPI ranks per node causes:
  1. Excessive memory overhead from duplicated ghost/halo layers;
  2. MPI communication buffer exhaustion;
  3. Increased latency in collective operations.  
  In **ParallelCFD**, I demonstrated that on 16 cores, a hybrid configuration (**8 MPI ranks × 2 OpenMP threads**) ran **$24\%$ faster than pure MPI** and **$94\%$ faster than pure OpenMP** by optimizing communication endpoints and core-local cache utilization.

---

### Question 2: Why did strong scaling saturate at 8 threads on your 24-core / 32-thread CPU?
**Model Answer:**
This illustrates two key architectural realities:
1. **The Roofline Model & Memory Bandwidth Saturation:**
   CFD post-processing operations (like velocity magnitude with $I \approx 0.16$ FLOP/Byte, or divergence with $I \approx 0.36$ FLOP/Byte) have low operational intensity. On 8 P-cores executing 256-bit AVX2 SIMD instructions simultaneously, memory requests saturate the dual-channel DDR5-5600 bus at $\approx 77$ GB/s (near the theoretical $89.6$ GB/s peak). Beyond this point, cores are stalled waiting for DRAM latency; adding more threads cannot fetch data faster.
2. **Heterogeneous Core Topologies (P-cores vs. E-cores):**
   The Intel Core i9-14900HX contains 8 high-performance P-cores ($5.8$ GHz, 2 MiB dedicated L2) and 16 efficiency E-cores ($3.8$ GHz, 4 MiB shared L2 per 4-core cluster). Under OpenMP static scheduling (`schedule(static)`), loop iterations are distributed equally ($N/p$). Because P-cores process iterations much faster than E-cores, the P-cores finish early and spin-wait at the implicit barrier, creating load imbalance.

---

### Question 3: What is False Sharing, how did you demonstrate it, and how is it eliminated in C++?
**Model Answer:**
- **Definition:** False sharing occurs when multiple threads on different cores modify independent variables that reside on the same 64-byte L1 cache line.
- **Hardware Mechanism:** Even though the threads write to distinct logical variables, the hardware cache-coherence protocol (**MESI**) operates with cache-line granularity. Core 0 writing to variable A invalidates the entire 64-byte line in Core 1's L1 cache, forcing Core 1 to take an expensive cache miss when updating variable B.
- **Empirical Demonstration:** In ParallelCFD, I created a benchmark with 32 threads updating adjacent `uint64_t` counters. Updating an unpadded array required $38.05$ ms due to continuous cache line bouncing.
- **Resolution:** I aligned each thread's accumulator to a distinct 64-byte boundary using C++11/20 alignment:
  ```cpp
  struct alignas(64) PaddedCounter {
      uint64_t val;
      char pad[56]; // 64 bytes total
  };
  ```
  Padding eliminated false sharing completely, yielding an immediate **$1.81\times$ speedup**.

---

### Question 4: Compare `#pragma omp atomic`, `#pragma omp critical`, and `reduction(+:...)`. Why is `reduction` orders of magnitude faster?
**Model Answer:**
In our empirical benchmark executing $320,000,000$ accumulator increments across 32 threads:
- **`#pragma omp critical` ($22,294$ ms — ~63,500× slower):**
  Instantiates a software mutual-exclusion lock. Threads sleep or spin-wait on a serialized critical section. 31 cores are idle at any given cycle.
- **`#pragma omp atomic` ($5,748$ ms — ~16,400× slower):**
  Uses hardware-level atomic instructions (such as x86 `LOCK XADD`). While faster than an OS mutex, it still forces cache-line lock contention and serializes memory updates at the L1 cache controller.
- **`reduction(+:...)` ($0.35$ ms — 1.0× baseline):**
  Eliminates synchronization entirely during the parallel loop. The OpenMP compiler allocates a **thread-private register** for each thread's accumulator. All 32 threads increment private registers at full CPU clock frequency. When the loop terminates, OpenMP performs an $O(\log_2 p)$ tree reduction to accumulate the final result.

---

### Question 5: Why is Structure of Arrays (SoA) preferred over Array of Structures (AoS) in CFD post-processing?
**Model Answer:**
- In an **Array of Structures (AoS)**, data is interleaved: `[u0, v0, w0, u1, v1, w1, ...]`. If a kernel only needs velocity magnitude or divergence, loading $u$ requires reading 24 bytes to extract 8 bytes. Furthermore, loading into AVX2 registers requires strided gather instructions or complex in-register shuffles.
- In a **Structure of Arrays (SoA)**, data components are stored in separate contiguous buffers: `u[...]`, `v[...]`, `w[...]`. A single 256-bit vector load (`_mm256_load_pd`) streams 4 consecutive double-precision numbers directly from L1/L2 cache into registers with unit stride.
- In ParallelCFD, SoA execution was **$1.45\times$ faster** in serial and enabled clean compiler auto-vectorization.

---

### Question 6: Why did parallel reductions yield a slight numerical difference compared to serial summation? Is that a bug?
**Model Answer:**
- It is **not a bug**; it is the fundamental consequence of **IEEE-754 floating-point non-associativity**:
  $$(a + b) + c \ne a + (b + c)$$
- In a serial loop, numbers are accumulated sequentially: $S = ((a_0 + a_1) + a_2) + \dots$.
- In an OpenMP reduction, each thread accumulates a local partition, and then the $p$ partial sums are added together in tree order: $S = (S_0 + S_1) + (S_2 + S_3) + \dots$.
- Because floating-point numbers have finite precision (53 mantissa bits in 64-bit doubles), intermediate rounding differs based on summation order. In ParallelCFD, summing kinetic energy across millions of cells produced an absolute difference of $\approx 3.88 \times 10^{-8}$ (relative difference $\approx 8.7 \times 10^{-13}$), which is consistent with the machine epsilon bound $\epsilon_{\text{mach}} \sqrt{N}$.

---

### Question 7: What is the Q-Criterion and why is it preferred over vorticity for vortex identification?
**Model Answer:**
- **Mathematical Definition:**
  Given velocity gradient tensor $J = \nabla \mathbf{u}$, split it into symmetric rate-of-strain $S = \frac{1}{2}(J + J^T)$ and antisymmetric rate-of-rotation $\Omega = \frac{1}{2}(J - J^T)$. The Q-criterion is defined as:
  $$Q = \frac{1}{2}(\|\Omega\|^2 - \|S\|^2)$$
- **Physical Significance:**
  Vorticity ($\boldsymbol{\omega} = \nabla \times \mathbf{u}$) measures local fluid rotation, but it **cannot distinguish between a coherent vortex core and pure boundary-layer shear**.
  - In a **pure planar Couette shear flow** ($u = \dot{\gamma}y, v=0, w=0$), vorticity is non-zero ($\omega_z = -\dot{\gamma}$), yet there is no vortex tube.
  - In our automated unit test for pure shear, rate of strain equals rate of rotation ($\|S\|^2 = \|\Omega\|^2$), correctly yielding **$Q \equiv 0$**.
  - Where $Q > 0$, rotation dominates strain, correctly identifying coherent, tubular vortex cores.

---

### Question 8: How did you implement thread affinity and CPU binding (`OMP_PROC_BIND`, `OMP_PLACES`)?
**Model Answer:**
By default, the Linux OS scheduler can migrate threads across CPU cores or sockets during execution, causing cold cache misses and destroying NUMA locality.
In our benchmark scripts and SLURM template:
```bash
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```
- `OMP_PLACES=cores` binds each OpenMP thread to an individual physical CPU core.
- `OMP_PROC_BIND=close` packs threads close together on the CPU to maximize shared L3 cache locality for stencil operations. For memory-bandwidth bound applications, `OMP_PROC_BIND=spread` can be used to distribute threads across memory channels.

---

### Question 9: Why did you choose `collapse(2)` instead of `collapse(3)` for 3D finite-difference stencils?
**Model Answer:**
In a 3D grid with row-major indexing ($N_x \times N_y \times N_z$):
- Collapsing outer loops `collapse(2)` merges $(i, j)$ loops into $N_x \times N_y = 16,384$ iterations, providing abundant parallelism for OpenMP threads.
- Crucially, leaving the innermost loop over $k$ un-collapsed allows it to remain a clean, contiguous unit-stride loop ($\Delta z = 1$). This enables the compiler to apply `#pragma omp simd` and generate AVX2 vector instructions without calculating 3D index coordinates via modulo arithmetic.

---

### Question 10: How do you pass data between Python and C++ without memory copies?
**Model Answer:**
Using **pybind11's buffer protocol**:
```cpp
py::buffer_info u_info = u_arr.request();
const double* u_ptr = static_cast<const double*>(u_info.ptr);
```
- We extract the raw 64-bit aligned memory pointer directly from NumPy's C-contiguous array.
- We release Python's Global Interpreter Lock (`py::gil_scoped_release release;`) so that OpenMP threads run on all CPU cores at 100% capacity without interpreter interference.
- The output NumPy array is allocated once and written directly by C++ threads, achieving zero-copy transfer between Python/PyVista and C++.
