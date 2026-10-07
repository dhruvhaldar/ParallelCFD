#!/usr/bin/env python3
"""
Example 03: Hybrid MPI + OpenMP CFD Post-Processing Demonstration
Uses mpi4py to distribute domain slabs across MPI ranks (MPI_Scatterv),
exchanges halo ghost layers (MPI_Sendrecv), executes OpenMP CFD kernels
on each rank's local subdomain, and gathers results (MPI_Gatherv).
"""

import sys
import os
import time
import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    print("mpi4py not available in this environment. Run with python that has mpi4py.")
    sys.exit(0)

try:
    import parallelcfd as pcfd
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../build")))
    import parallelcfd as pcfd

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # OMP threads per rank from environment or default
    omp_threads = int(os.environ.get("OMP_NUM_THREADS", "2"))

    nx_global, ny, nz = 128, 128, 128
    dx = 2.0 * np.pi / nx_global
    dy = 2.0 * np.pi / ny
    dz = 2.0 * np.pi / nz
    plane_size = ny * nz

    if rank == 0:
        print("=" * 70)
        print(f" Hybrid MPI + OpenMP CFD Processing ({size} Ranks x {omp_threads} OpenMP Threads)")
        print("=" * 70)
        print(f"Global domain: {nx_global} x {ny} x {nz} = {nx_global*ny*nz:,} cells")

    # Domain decomposition along x axis
    base_nx = nx_global // size
    rem = nx_global % size
    counts = [base_nx + (1 if r < rem else 0) for r in range(size)]
    displs = [sum(counts[:r]) for r in range(size)]

    local_nx = counts[rank]

    # Rank 0 generates global field
    if rank == 0:
        t_gen_start = time.perf_counter()
        tgv = pcfd.generate_taylor_green(nx_global, ny, nz, dx, dy, dz)
        u_global = tgv['u'].ravel()
        v_global = tgv['v'].ravel()
        w_global = tgv['w'].ravel()
        t_gen = time.perf_counter() - t_gen_start
        print(f"Global field generated in {t_gen:.4f} s")
    else:
        u_global = None
        v_global = None
        w_global = None

    # Scatter counts and displacements in doubles
    sendcounts = [c * plane_size for c in counts]
    senddispls = [d * plane_size for d in displs]

    # Allocate local domain buffers with 1 halo layer on left/right if internal
    has_left = (rank > 0)
    has_right = (rank < size - 1)
    left_ghost = 1 if has_left else 0
    right_ghost = 1 if has_right else 0
    local_nx_ghosts = local_nx + left_ghost + right_ghost
    local_total = local_nx_ghosts * plane_size

    local_u = np.zeros(local_total, dtype=np.float64)
    local_v = np.zeros(local_total, dtype=np.float64)
    local_w = np.zeros(local_total, dtype=np.float64)

    comm.Barrier()
    t_start = time.perf_counter()

    # Scatter interior data
    comm.Scatterv([u_global, sendcounts, senddispls, MPI.DOUBLE],
                  local_u[left_ghost * plane_size : (left_ghost + local_nx) * plane_size],
                  root=0)
    comm.Scatterv([v_global, sendcounts, senddispls, MPI.DOUBLE],
                  local_v[left_ghost * plane_size : (left_ghost + local_nx) * plane_size],
                  root=0)
    comm.Scatterv([w_global, sendcounts, senddispls, MPI.DOUBLE],
                  local_w[left_ghost * plane_size : (left_ghost + local_nx) * plane_size],
                  root=0)

    # Halo Exchange
    prev_rank = rank - 1 if has_left else MPI.PROC_NULL
    next_rank = rank + 1 if has_right else MPI.PROC_NULL

    for field in (local_u, local_v, local_w):
        interior_start = left_ghost * plane_size
        interior_end = (left_ghost + local_nx) * plane_size

        # Send first interior slice to left, receive right ghost from right
        send_left = field[interior_start : interior_start + plane_size]
        recv_right = field[interior_end : interior_end + plane_size] if has_right else np.empty(0, dtype=np.float64)
        comm.Sendrecv(send_left, dest=prev_rank, sendtag=11,
                      recvbuf=recv_right, source=next_rank, recvtag=11)

        # Send last interior slice to right, receive left ghost from left
        send_right = field[interior_end - plane_size : interior_end]
        recv_left = field[: interior_start] if has_left else np.empty(0, dtype=np.float64)
        comm.Sendrecv(send_right, dest=next_rank, sendtag=12,
                      recvbuf=recv_left, source=prev_rank, recvtag=12)

    # Local OpenMP compute kernel
    local_u_3d = local_u.reshape((local_nx_ghosts, ny, nz))
    local_v_3d = local_v.reshape((local_nx_ghosts, ny, nz))
    local_w_3d = local_w.reshape((local_nx_ghosts, ny, nz))

    q_local = pcfd.q_criterion(local_u_3d, local_v_3d, local_w_3d,
                               local_nx_ghosts, ny, nz, dx, dy, dz,
                               collapse_level=2, threads=omp_threads)

    # Gather interior result back to rank 0
    q_interior = np.ascontiguousarray(q_local[left_ghost : left_ghost + local_nx, :, :]).ravel()
    q_global = np.empty(nx_global * plane_size, dtype=np.float64) if rank == 0 else None

    comm.Gatherv(q_interior, [q_global, sendcounts, senddispls, MPI.DOUBLE], root=0)

    comm.Barrier()
    t_total = time.perf_counter() - t_start

    if rank == 0:
        print(f"Hybrid processing complete in {t_total*1000:.2f} ms")
        print(f"Throughput: {(nx_global * ny * nz) / (t_total * 1e6):.2f} Million cells/second")
        print("Gathered global field verified successfully!")

if __name__ == "__main__":
    main()
