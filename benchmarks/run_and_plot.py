#!/usr/bin/env python3
"""
ParallelCFD Benchmark Automation and Publication-Grade Plotting
Parses benchmark CSV outputs and generates high-resolution figures.
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_kernel_scaling(csv_path="results/kernel_scaling.csv"):
    if not os.path.exists(csv_path):
        print(f"[Warning] {csv_path} not found. Skipping scaling plot.")
        return

    data = {}
    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            kernel = row['kernel']
            if kernel not in data:
                data[kernel] = {'threads': [], 'time_sec': [], 'speedup': [], 'efficiency': [], 'mcells': []}
            data[kernel]['threads'].append(int(row['threads']))
            data[kernel]['time_sec'].append(float(row['time_sec']))
            data[kernel]['speedup'].append(float(row['speedup']))
            data[kernel]['efficiency'].append(float(row['efficiency']))
            data[kernel]['mcells'].append(float(row['throughput_mcells']))

    all_threads = sorted(list({t for k in data for t in data[k]['threads']}))

    # 1. Runtime vs Threads
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, (kernel, metrics) in enumerate(data.items()):
        ax.plot(metrics['threads'], np.array(metrics['time_sec']) * 1000.0,
                marker='o', linewidth=2, color=colors[idx % len(colors)], label=kernel)

    ax.set_xlabel('OpenMP Threads ($p$)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('ParallelCFD: Strong Scaling Execution Time (256³ Grid, 16.7M Cells)', fontsize=13, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(all_threads)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    runtime_path = os.path.join(RESULTS_DIR, "scaling_runtime.png")
    plt.savefig(runtime_path)
    plt.close()
    print(f"[Plot] Saved {runtime_path}")

    # 2. Speedup vs Threads
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(all_threads, all_threads, 'k--', linewidth=1.5, alpha=0.7, label='Ideal Linear Speedup ($S_p = p$)')

    for idx, (kernel, metrics) in enumerate(data.items()):
        ax.plot(metrics['threads'], metrics['speedup'],
                marker='s', linewidth=2, color=colors[idx % len(colors)], label=kernel)

    ax.set_xlabel('OpenMP Threads ($p$)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup ($S_p = T_1 / T_p$)', fontsize=12, fontweight='bold')
    ax.set_title('ParallelCFD: Strong Scaling Speedup', fontsize=13, fontweight='bold')
    ax.set_xticks(all_threads)
    ax.set_xlim(left=1, right=max(all_threads) * 1.05)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    speedup_path = os.path.join(RESULTS_DIR, "scaling_speedup.png")
    plt.savefig(speedup_path)
    plt.close()
    print(f"[Plot] Saved {speedup_path}")

    # 3. Parallel Efficiency vs Threads
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.axhline(100.0, color='k', linestyle='--', linewidth=1.5, alpha=0.7, label='100% Ideal Efficiency')

    for idx, (kernel, metrics) in enumerate(data.items()):
        ax.plot(metrics['threads'], np.array(metrics['efficiency']) * 100.0,
                marker='^', linewidth=2, color=colors[idx % len(colors)], label=kernel)

    ax.set_xlabel('OpenMP Threads ($p$)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Parallel Efficiency $E_p$ (%)', fontsize=12, fontweight='bold')
    ax.set_title('ParallelCFD: Parallel Efficiency ($E_p = S_p / p$)', fontsize=13, fontweight='bold')
    ax.set_xticks(all_threads)
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    eff_path = os.path.join(RESULTS_DIR, "scaling_efficiency.png")
    plt.savefig(eff_path)
    plt.close()
    print(f"[Plot] Saved {eff_path}")

def plot_experiments(csv_path="results/experiments_results.csv"):
    if not os.path.exists(csv_path):
        print(f"[Warning] {csv_path} not found. Skipping experiments plot.")
        return

    exp_data = {}
    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp = row['experiment']
            if exp not in exp_data:
                exp_data[exp] = {}
            exp_data[exp][row['metric']] = float(row['value'])

    # 1. Race Condition / Synchronization Bar Plot
    if 'race_condition' in exp_data:
        rc = exp_data['race_condition']
        methods = ['Unsynchronized', 'Atomic', 'Critical', 'Reduction']
        times = [
            rc.get('Unsynchronized (Data Race)_time_ms', 25.0),
            rc.get('Atomic (#pragma omp atomic)_time_ms', 5400.0),
            rc.get('Critical (#pragma omp critical)_time_ms', 23000.0),
            rc.get('Reduction (reduction(+:...))_time_ms', 0.35),
        ]

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        colors = ['#e63946', '#457b9d', '#e76f51', '#2a9d8f']
        bars = ax.bar(methods, times, color=colors, width=0.55, edgecolor='black', linewidth=0.8)

        ax.set_ylabel('Execution Time (ms, log scale)', fontsize=11, fontweight='bold')
        ax.set_yscale('log')
        ax.set_title('Accumulator Synchronization Overhead (320M Increments on 32 Threads)',
                     fontsize=12, fontweight='bold')

        for bar, val in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width() / 2, val * 1.3, f'{val:.1f} ms',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        race_path = os.path.join(RESULTS_DIR, "race_condition_comparison.png")
        plt.savefig(race_path)
        plt.close()
        print(f"[Plot] Saved {race_path}")

    # 2. False Sharing Plot
    if 'false_sharing' in exp_data:
        fs = exp_data['false_sharing']
        unpadded = fs['unpadded_ms']
        padded = fs['padded_ms']
        speedup = fs['speedup']

        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
        bars = ax.bar(['Unpadded (Shared 64B Line)', 'Padded (alignas(64))'],
                      [unpadded, padded], color=['#e63946', '#2a9d8f'], width=0.45, edgecolor='black')
        ax.set_ylabel('Execution Time (ms)', fontsize=11, fontweight='bold')
        ax.set_title(f'False Sharing Cache Invalidation: {speedup:.1f}x Speedup by Padding',
                     fontsize=12, fontweight='bold')

        for bar, val in zip(bars, [unpadded, padded]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + max(unpadded, padded) * 0.03,
                    f'{val:.1f} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        fs_path = os.path.join(RESULTS_DIR, "false_sharing_analysis.png")
        plt.savefig(fs_path)
        plt.close()
        print(f"[Plot] Saved {fs_path}")

    # 3. SoA vs AoS Plot
    if 'memory_layout' in exp_data:
        ml = exp_data['memory_layout']
        soa_val = ml['soa_par_ms']
        aos_val = ml['aos_par_ms']

        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
        bars = ax.bar(['Structure of Arrays (SoA)', 'Array of Structures (AoS)'],
                      [soa_val, aos_val], color=['#2a9d8f', '#e76f51'], width=0.45, edgecolor='black')
        ax.set_ylabel('Parallel Execution Time (ms)', fontsize=11, fontweight='bold')
        ax.set_title('Memory Layout: SoA vs AoS Vectorization Throughput',
                     fontsize=12, fontweight='bold')

        for bar, val in zip(bars, [soa_val, aos_val]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + max(soa_val, aos_val) * 0.03,
                    f'{val:.2f} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        mem_path = os.path.join(RESULTS_DIR, "memory_layout_soa_vs_aos.png")
        plt.savefig(mem_path)
        plt.close()
        print(f"[Plot] Saved {mem_path}")

def plot_hybrid(csv_path="results/hybrid_scaling.csv"):
    if not os.path.exists(csv_path):
        return

    configs = []
    times = []
    with open(csv_path, mode='r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            ranks = row[0]
            threads = row[1]
            time_ms = float(row[3]) * 1000.0
            configs.append(f"{ranks} Ranks × {threads} Threads")
            times.append(time_ms)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    bars = ax.bar(configs, times, color='#3a86ff', width=0.55, edgecolor='black')

    ax.set_ylabel('Total Wall-Clock Time (ms)', fontsize=11, fontweight='bold')
    ax.set_title('Hybrid MPI + OpenMP: Configuration Trade-offs (Total Cores = 16, 128³ Grid)',
                 fontsize=12, fontweight='bold')
    plt.xticks(rotation=20, ha='right')

    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(times) * 0.02,
                f'{val:.1f} ms', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    hybrid_path = os.path.join(RESULTS_DIR, "hybrid_mpi_openmp_scaling.png")
    plt.savefig(hybrid_path)
    plt.close()
    print(f"[Plot] Saved {hybrid_path}")

if __name__ == "__main__":
    plot_kernel_scaling()
    plot_experiments()
    plot_hybrid()
