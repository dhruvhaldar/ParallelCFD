#!/usr/bin/env python3
"""
Generate publication-quality PDF technical guide and report for ParallelCFD
Compiles architecture, mathematics, benchmarks, diagrams, and interview dossier into a single PDF.
"""

import os
import sys
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))

        # Don't draw running header/footer on cover page
        if self._pageNumber > 1:
            # Header
            self.drawString(54, 755, "ParallelCFD: OpenMP-Accelerated CFD Field Analysis & Technical Guide")
            self.drawRightString(612 - 54, 755, "Dhruv Haldar")
            self.setStrokeColor(colors.HexColor("#dddddd"))
            self.setLineWidth(0.5)
            self.line(54, 747, 612 - 54, 747)

            # Footer
            self.line(54, 45, 612 - 54, 45)
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(612 - 54, 32, page_text)
            self.drawString(54, 32, "Confidential / Portfolio Dossier - NTNU & HPC Engineering")

        self.restoreState()

def build_pdf(filename="docs/ParallelCFD_Comprehensive_Guide.pdf"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1a237e'),
        alignment=1, # Center
        spaceAfter=12
    )

    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#283593'),
        alignment=1,
        spaceAfter=25
    )

    author_style = ParagraphStyle(
        'CoverAuthor',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#37474f'),
        alignment=1,
        spaceAfter=6
    )

    meta_style = ParagraphStyle(
        'CoverMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#607d8b'),
        alignment=1,
        spaceAfter=20
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.HexColor('#1a237e'),
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0d47a1'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    h3_style = ParagraphStyle(
        'SectionH3',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1565c0'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'MainBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#212121'),
        spaceAfter=8
    )

    code_style = ParagraphStyle(
        'CodeSnippet',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#263238'),
        backColor=colors.HexColor('#f5f7fa'),
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=8
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1a237e'),
        backColor=colors.HexColor('#e8eaf6'),
        borderPadding=8,
        spaceBefore=6,
        spaceAfter=10
    )

    story = []

    # =========================================================================
    # COVER PAGE
    # =========================================================================
    story.append(Spacer(1, 40))
    story.append(Paragraph("ParallelCFD", title_style))
    story.append(Paragraph("OpenMP-Accelerated 3D CFD Field Analysis & Technical Guide", subtitle_style))
    story.append(HRFlowable(width="80%", thickness=2, color=colors.HexColor('#1a237e'), spaceAfter=25))

    story.append(Paragraph("<b>Author:</b> Dhruv Haldar", author_style))
    story.append(Paragraph("HPC Engineer & Scientific Software Developer Dossier", meta_style))
    story.append(Paragraph("<b>Core Stack:</b> C++20 | OpenMP 5.2 | OpenMPI 5.0 | SIMD (AVX2) | Python | PyVista | VTK", meta_style))
    story.append(Paragraph("<b>Target Platform:</b> Intel Core i9-14900HX (24 Physical Cores / 32 Threads, 36MB L3)", meta_style))

    story.append(Spacer(1, 20))

    # Add visual highlight on cover
    if os.path.exists("results/pyvista_q_criterion_3d.png"):
        story.append(Image("results/pyvista_q_criterion_3d.png", width=6.2*inch, height=3.1*inch))
        story.append(Paragraph("<font size=7 color='#666666'><b>Figure 1:</b> 3D Q-Criterion vortex core structures in a Taylor-Green vortex computed via ParallelCFD OpenMP kernels and rendered in PyVista.</font>", ParagraphStyle('FigCap', parent=styles['Normal'], alignment=1, spaceBefore=4)))

    story.append(PageBreak())

    # =========================================================================
    # 1. EXECUTIVE SUMMARY & BACKGROUND
    # =========================================================================
    story.append(Paragraph("1. Executive Summary & Technical Motivation", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#3f51b5'), spaceAfter=10))

    summary_text = (
        "<b>ParallelCFD</b> is a high-performance C++20/OpenMP toolkit developed to accelerate "
        "computationally expensive post-processing operations on large-scale Computational Fluid Dynamics (CFD) datasets. "
        "Rather than implementing generic matrix multiplication exercises, this project establishes a direct, technically "
        "rigorous bridge connecting distributed-memory <b>MPI workflows</b> (using Scatterv, halo exchanges, and Gatherv) "
        "with <b>shared-memory OpenMP multi-threading</b>, <b>SIMD vectorization</b>, and <b>PyVista/VTK 3D visualization</b>."
    )
    story.append(Paragraph(summary_text, body_style))

    story.append(Paragraph(
        "<b>Core Competencies Demonstrated:</b><br/>"
        "• Shared-memory thread scalability and SIMD auto-vectorization using OpenMP 5.2.<br/>"
        "• Hardware-aware memory layout optimization (Structure of Arrays vs. Array of Structures).<br/>"
        "• Cache-coherence diagnostics (false sharing on 64-byte lines and alignment with alignas(64)).<br/>"
        "• Numerical behavior of parallel reductions and IEEE-754 floating-point non-associativity.<br/>"
        "• Finite-difference derivative stencils, incompressibility, and tensor-based vortex identification (Q-criterion).<br/>"
        "• Hybrid MPI + OpenMP domain decomposition with ghost cell halo exchange.<br/>"
        "• Python C-extension binding via pybind11 with zero-copy buffer protocols and GIL release.",
        callout_style
    ))

    # =========================================================================
    # 2. MATHEMATICAL FOUNDATIONS OF CFD POST-PROCESSING
    # =========================================================================
    story.append(Paragraph("2. Mathematical Formulations & CFD Kernels", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#3f51b5'), spaceAfter=10))

    story.append(Paragraph("2.1 Velocity Magnitude & Kinetic Energy", h2_style))
    story.append(Paragraph(
        "For a 3D Cartesian velocity vector <b>u</b>(x, y, z) = (u, v, w)<sup>T</sup>, the magnitude and "
        "total volume-integrated kinetic energy are computed cell-wise as:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>|u| = √(u² + v² + w²)</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>K = 0.5 · Σ (u<sub>i</sub>² + v<sub>i</sub>² + w<sub>i</sub>²) · ΔV</b><br/>"
        "With 24 bytes read and 8 bytes written per cell against 6 FLOPs, the operational intensity is "
        "I ≈ 0.1875 FLOP/byte, making this kernel strongly memory-bandwidth bound.",
        body_style
    ))

    story.append(Paragraph("2.2 Finite Difference Stencils & Boundary Discretization", h2_style))
    story.append(Paragraph(
        "On a structured grid (Nx × Ny × Nz), central differences provide 2nd-order accuracy for interior nodes:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;∂φ/∂x |<sub>i</sub> = (φ<sub>i+1</sub> - φ<sub>i-1</sub>) / (2Δx) + O(Δx²)<br/>"
        "At boundaries (i = 0 and i = Nx-1), standard 1st-order differences introduce severe O(Δx) boundary errors. "
        "ParallelCFD implements <b>2nd-order 3-point one-sided boundary stencils</b>:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;∂φ/∂x |<sub>0</sub> = (-3φ<sub>0</sub> + 4φ<sub>1</sub> - φ<sub>2</sub>) / (2Δx) + O(Δx²)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;∂φ/∂x |<sub>Nx-1</sub> = (3φ<sub>Nx-1</sub> - 4φ<sub>Nx-2</sub> + φ<sub>Nx-3</sub>) / (2Δx) + O(Δx²)<br/>"
        "This maintains uniform second-order convergence throughout the computational domain.",
        body_style
    ))

    story.append(Paragraph("2.3 Divergence & Incompressibility", h2_style))
    story.append(Paragraph(
        "For incompressible fluid flow, mass conservation requires:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>∇ · u = ∂u/∂x + ∂v/∂y + ∂w/∂z = 0</b><br/>"
        "In ParallelCFD, divergence is implemented as a <b>fused single-pass stencil</b>. Rather than allocating "
        "and storing three intermediate gradient fields (which would write and reread 24 bytes/cell to DRAM), the derivatives "
        "are accumulated directly in CPU vector registers.",
        body_style
    ))

    story.append(Paragraph("2.4 Vorticity vs. Q-Criterion: Why Vorticity Is Not Enough", h2_style))
    story.append(Paragraph(
        "The vorticity vector is defined as <b>ω = ∇ × u</b>. While vorticity measures fluid rotation, "
        "<b>vorticity cannot distinguish between a real rotating vortex tube and pure laminar shear</b>.<br/>"
        "<i>Mathematical Proof (Couette Flow):</i> In pure planar shear flow u = γ·y, v = 0, w = 0, vorticity is non-zero "
        "(ω<sub>z</sub> = -γ), yet no vortex exists.<br/><br/>"
        "The <b>Q-criterion</b> (Hunt, Wray & Moin, 1988) resolves this by decomposing the velocity gradient tensor "
        "J = ∇u into its symmetric rate-of-strain tensor S = 0.5(J + J<sup>T</sup>) and antisymmetric rate-of-rotation tensor "
        "Ω = 0.5(J - J<sup>T</sup>):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>Q = 0.5 · (||Ω||<sub>F</sub>² - ||S||<sub>F</sub>²)</b><br/>"
        "• <b>Q > 0:</b> Rotation dominates strain rate → <b>Coherent Vortex Tube</b>.<br/>"
        "• <b>Q < 0:</b> Strain dominates rotation → <b>Shear / Boundary Layer</b>.<br/>"
        "• <b>In Couette shear flow:</b> ||Ω||<sub>F</sub>² = γ²/2 and ||S||<sub>F</sub>² = γ²/2, so <b>Q ≡ 0 identically</b>.",
        body_style
    ))

    # Insert Vorticity Slices image
    if os.path.exists("results/pyvista_vorticity_slices.png"):
        story.append(Spacer(1, 8))
        story.append(Image("results/pyvista_vorticity_slices.png", width=5.8*inch, height=2.8*inch))
        story.append(Paragraph("<font size=7 color='#666666'><b>Figure 2:</b> Orthogonal 3D slice planes showing vorticity magnitude computed across a 3D Taylor-Green vortex.</font>", ParagraphStyle('FigCap2', parent=styles['Normal'], alignment=1, spaceBefore=4)))

    story.append(PageBreak())

    # =========================================================================
    # 3. MEMORY ARCHITECTURE & CACHE OPTIMIZATION
    # =========================================================================
    story.append(Paragraph("3. Memory Architecture & Hardware Optimization", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#3f51b5'), spaceAfter=10))

    story.append(Paragraph("3.1 Structure of Arrays (SoA) vs. Array of Structures (AoS)", h2_style))
    story.append(Paragraph(
        "A critical question in scientific computing is how vector fields should be laid out in memory:<br/>"
        "• <b>Array of Structures (AoS):</b> struct { double u, v, w; }; std::vector<Velocity> vel;<br/>"
        "• <b>Structure of Arrays (SoA):</b> std::vector<double> u, v, w;<br/><br/>"
        "In AoS, data components are interleaved: [u0, v0, w0, u1, v1, w1...]. An AVX2 vector load (256-bit = 4 doubles) "
        "fetches mixed components [u0, v0, w0, u1], requiring expensive register shuffles before mathematical operations.<br/>"
        "In SoA, u, v, and w are stored contiguously. A single instruction (_mm256_load_pd) loads 4 consecutive velocity "
        "values into a SIMD register with unit stride.<br/>"
        "<b>Empirical Result:</b> SoA execution was <b>1.45× faster</b> than AoS on 2.1M cells.",
        body_style
    ))

    if os.path.exists("results/memory_layout_soa_vs_aos.png"):
        story.append(Image("results/memory_layout_soa_vs_aos.png", width=4.8*inch, height=2.8*inch))
        story.append(Paragraph("<font size=7 color='#666666'><b>Figure 3:</b> Execution time comparison between Structure of Arrays (SoA) and Array of Structures (AoS).</font>", ParagraphStyle('FigCap3', parent=styles['Normal'], alignment=1, spaceBefore=4)))

    story.append(Paragraph("3.2 False Sharing & Cache Line Invalidation (MESI Protocol)", h2_style))
    story.append(Paragraph(
        "On multi-core x86 CPUs, cache coherency operates on <b>64-byte cache lines</b>. "
        "When multiple threads update independent accumulators residing within the same 64-byte segment, "
        "the hardware <b>MESI protocol</b> invalidates the entire cache line across all participating cores.<br/>"
        "We diagnosed and eliminated this overhead using C++ alignment padding:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>struct alignas(64) PaddedCounter { uint64_t val; char pad[56]; };</b><br/>"
        "<b>Empirical Result:</b> Padding accumulators to 64 bytes eliminated false sharing, delivering a <b>1.81× speedup</b> "
        "(38.05 ms vs. 21.97 ms for 50 million iterations across 32 threads).",
        body_style
    ))

    if os.path.exists("results/false_sharing_analysis.png"):
        story.append(Image("results/false_sharing_analysis.png", width=4.8*inch, height=2.8*inch))
        story.append(Paragraph("<font size=7 color='#666666'><b>Figure 4:</b> False sharing benchmark demonstrating 1.81× speedup via cache line padding.</font>", ParagraphStyle('FigCap4', parent=styles['Normal'], alignment=1, spaceBefore=4)))

    story.append(PageBreak())

    # =========================================================================
    # 4. CONCURRENCY & SYNCHRONIZATION OVERHEAD
    # =========================================================================
    story.append(Paragraph("4. OpenMP Synchronization & Concurrency Analysis", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#3f51b5'), spaceAfter=10))

    story.append(Paragraph(
        "A controlled benchmark executed <b>320,000,000 accumulator increments</b> across 32 threads to evaluate "
        "synchronization overhead:",
        body_style
    ))

    table_data = [
        ["Synchronization Primitive", "OpenMP Construct", "Measured Time", "Correctness", "Relative Cost"],
        ["Unsynchronized", "sum += 1 (Data Race)", "47.8 ms", "Corrupted (75% Lost)", "N/A"],
        ["Critical Section", "#pragma omp critical", "22,294.5 ms", "Exact (320M)", "~63,500×"],
        ["Atomic Operation", "#pragma omp atomic", "5,748.2 ms", "Exact (320M)", "~16,400×"],
        ["Parallel Reduction", "reduction(+:sum)", "0.35 ms", "Exact (320M)", "1.0× (Baseline)"]
    ]
    t = Table(table_data, colWidths=[1.4*inch, 1.4*inch, 1.0*inch, 1.3*inch, 1.1*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bbdefb')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f7fa')])
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    if os.path.exists("results/race_condition_comparison.png"):
        story.append(Image("results/race_condition_comparison.png", width=5.5*inch, height=2.8*inch))
        story.append(Paragraph("<font size=7 color='#666666'><b>Figure 5:</b> Log-scale execution time comparison across OpenMP synchronization primitives.</font>", ParagraphStyle('FigCap5', parent=styles['Normal'], alignment=1, spaceBefore=4)))

    story.append(Paragraph("4.2 IEEE-754 Floating-Point Non-Associativity in Reductions", h2_style))
    story.append(Paragraph(
        "In pure mathematics, addition is associative: (a + b) + c = a + (b + c). "
        "In IEEE-754 64-bit double precision, addition is <b>non-associative</b> due to finite 53-bit mantissa rounding.<br/>"
        "Sequential summation adds numbers linearly: S = ((a0 + a1) + a2) + ...<br/>"
        "OpenMP reduction partitions the array across p threads and performs a tree summation: S = (S0 + S1) + (S2 + S3)...<br/>"
        "In ParallelCFD, summing kinetic energy across millions of cells produces an absolute difference of "
        "Δ ≈ 3.88 × 10<sup>-8</sup> (relative difference ≈ 8.7 × 10<sup>-13</sup>). "
        "This confirms theoretical bounds and explains why parallel reduction results differ slightly from serial baselines.",
        body_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # 5. HARDWARE PERFORMANCE & THE ROOFLINE MODEL
    # =========================================================================
    story.append(Paragraph("5. Scaling Studies & Architectural Roofline Analysis", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#3f51b5'), spaceAfter=10))

    story.append(Paragraph("5.1 Strong Scaling Profile (256³ Grid, 16.78 Million Cells)", h2_style))
    story.append(Paragraph(
        "Strong scaling benchmarks on an Intel Core i9-14900HX (24 cores / 32 threads, 36MB L3 cache) "
        "reveal the execution profile of 3D CFD kernels:",
        body_style
    ))

    scaling_table = [
        ["Threads", "Velocity Mag", "Divergence", "Vorticity", "Q-Criterion", "Q Speedup", "Efficiency", "Throughput"],
        ["1", "16.27 ms", "20.00 ms", "48.09 ms", "71.07 ms", "1.00×", "100.0%", "236.1 Mcells/s"],
        ["2", "12.51 ms", "14.86 ms", "30.67 ms", "39.53 ms", "1.80×", "89.9%", "424.4 Mcells/s"],
        ["4", "10.51 ms", "11.71 ms", "27.12 ms", "20.03 ms", "3.55×", "88.7%", "837.6 Mcells/s"],
        ["8", "10.14 ms", "11.36 ms", "30.82 ms", "18.71 ms", "3.80×", "47.5%", "896.7 Mcells/s"],
        ["16", "9.68 ms", "13.33 ms", "34.59 ms", "20.38 ms", "3.49×", "21.8%", "823.3 Mcells/s"],
        ["32", "9.74 ms", "12.42 ms", "38.36 ms", "21.96 ms", "3.24×", "10.1%", "764.0 Mcells/s"]
    ]
    st = Table(scaling_table, colWidths=[0.6*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.8*inch, 0.8*inch, 1.1*inch])
    st.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bbdefb')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f7fa')])
    ]))
    story.append(st)
    story.append(Spacer(1, 10))

    if os.path.exists("results/scaling_speedup.png") and os.path.exists("results/scaling_runtime.png"):
        story.append(Table([
            [Image("results/scaling_speedup.png", width=3.1*inch, height=2.1*inch),
             Image("results/scaling_runtime.png", width=3.1*inch, height=2.1*inch)]
        ], colWidths=[3.2*inch, 3.2*inch]))
        story.append(Paragraph("<font size=7 color='#666666'><b>Figure 6:</b> Strong scaling speedup (left) and log-log execution runtime (right) across OpenMP threads.</font>", ParagraphStyle('FigCap6', parent=styles['Normal'], alignment=1, spaceBefore=4)))

    story.append(Paragraph("5.2 Why Scaling Saturates at 8 Threads (The Roofline Limit)", h2_style))
    story.append(Paragraph(
        "<b>1. Memory Bandwidth Ceiling:</b> For streaming operations (velocity magnitude, vorticity), operational intensity is "
        "low (I ≤ 0.45 FLOP/byte). At 8 P-cores executing AVX2 instructions, aggregate memory demand reaches ~77 GB/s, "
        "saturating the dual-channel DDR5-5600 bus (peak theoretical 89.6 GB/s). Cores become memory-bandwidth bound.<br/><br/>"
        "<b>2. P-Core vs. E-Core Heterogeneity:</b> Cores 0-7 are high-performance P-cores (5.8 GHz, 2MB L2/core). "
        "Cores 8-23 are efficiency E-cores (3.8 GHz, shared 4MB L2 clusters). Under static scheduling (schedule(static)), "
        "iterations are divided equally. The faster P-cores complete early and stall at the loop barrier waiting for the E-cores.<br/><br/>"
        "<b>3. Amdahl's Law Serial Fraction:</b> Solving S<sub>16</sub> = 3.49 yields an apparent serial fraction of s ≈ 23.9%, "
        "representing memory bus contention and barrier synchronization latency rather than unparallelized code.",
        body_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # 6. HYBRID MPI + OPENMP DOMAIN DECOMPOSITION
    # =========================================================================
    story.append(Paragraph("6. Hybrid MPI + OpenMP Distributed Architecture", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#3f51b5'), spaceAfter=10))

    story.append(Paragraph(
        "Modern supercomputers feature nodes with 64 to 256 CPU cores. Running 256 pure MPI ranks per node causes "
        "memory exhaustion from duplicate ghost halos and network buffer congestion. ParallelCFD implements a "
        "<b>Hybrid MPI + OpenMP</b> architecture combining distributed slab decomposition with local multi-threading:",
        body_style
    ))

    hybrid_table = [
        ["Configuration Topology", "MPI Ranks", "OpenMP Threads", "Total Cores", "Wall Time", "Throughput", "Relative Speedup"],
        ["Pure MPI", "16", "1", "16", "13.86 ms", "151.3 Mcells/s", "1.00× (Baseline)"],
        ["Hybrid (Optimal)", "8", "2", "16", "11.14 ms", "188.2 Mcells/s", "1.24× Faster"],
        ["Hybrid", "4", "4", "16", "11.54 ms", "181.7 Mcells/s", "1.20× Faster"],
        ["Hybrid", "2", "8", "16", "15.03 ms", "139.5 Mcells/s", "0.92×"],
        ["Pure OpenMP", "1", "16", "16", "21.60 ms", "97.1 Mcells/s", "0.64×"]
    ]
    ht = Table(hybrid_table, colWidths=[1.5*inch, 0.8*inch, 1.0*inch, 0.8*inch, 0.8*inch, 1.0*inch, 1.0*inch])
    ht.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bbdefb')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f7fa')])
    ]))
    story.append(ht)
    story.append(Spacer(1, 10))

    if os.path.exists("results/hybrid_mpi_openmp_scaling.png"):
        story.append(Image("results/hybrid_mpi_openmp_scaling.png", width=5.5*inch, height=2.8*inch))
        story.append(Paragraph("<font size=7 color='#666666'><b>Figure 7:</b> Total wall-clock time across hybrid process/thread configurations on 16 total CPU cores.</font>", ParagraphStyle('FigCap7', parent=styles['Normal'], alignment=1, spaceBefore=4)))

    story.append(Paragraph(
        "<b>Why Hybrid Wins:</b><br/>"
        "• Pure MPI (16×1) generates 16 sets of IPC halo exchanges and communication buffers.<br/>"
        "• Pure OpenMP (1×16) suffers from thread synchronization overhead and memory bus contention in a single process.<br/>"
        "• The hybrid configuration (8 ranks × 2 threads) minimizes communication endpoints by 50% while ensuring thread "
        "workloads reside inside private L2/L3 cache slices.",
        callout_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # 7. HPC TECHNICAL INTERVIEW DOSSIER
    # =========================================================================
    story.append(Paragraph("7. HPC Technical Interview Q&A Dossier", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#3f51b5'), spaceAfter=10))

    qa_list = [
        ("Q1: When should an engineer use MPI vs. OpenMP vs. Hybrid?",
         "MPI is designed for distributed-memory systems across physical nodes or clusters requiring explicit message passing (MPI_Sendrecv, Scatterv, Gatherv). OpenMP is designed for shared-memory multi-threading within a single node. Hybrid MPI+OpenMP is required on modern multi-core nodes (64-128 cores/socket) to prevent MPI rank explosion, reduce ghost cell memory duplication, and optimize intra-node cache locality."),

        ("Q2: Why did strong scaling plateau at 8 threads on your 24-core CPU?",
         "Streaming CFD kernels have an operational intensity below 0.45 FLOP/byte. At 8 P-cores executing AVX2 SIMD instructions, memory requests saturate the dual-channel DDR5-5600 bus at ~77 GB/s (near the theoretical 89.6 GB/s peak), hitting the slanted roofline of the Roofline Model. Furthermore, cores 8-23 are lower-frequency E-cores (3.8 GHz vs. 5.8 GHz); under static scheduling, faster P-cores finish early and wait at the loop barrier."),

        ("Q3: What is false sharing and how do you eliminate it in C++?",
         "False sharing occurs when multiple threads write to independent variables that reside within the same 64-byte L1 cache line. Under the MESI protocol, a write on core 0 invalidates the entire cache line in core 1's cache, forcing expensive pipeline stalls. We eliminated it by padding accumulators with 'alignas(64)', yielding an immediate 1.81× speedup."),

        ("Q4: Compare atomic, critical, and reduction OpenMP directives.",
         "Critical sections enforce software mutual exclusion, serializing execution (~63,500× slower). Atomic operations use hardware cache-line locking instructions like LOCK XADD (~16,400× slower). Reductions eliminate synchronization entirely during the loop by allocating thread-private registers, followed by a fast O(log2 p) logarithmic tree fold at loop exit (0.35 ms baseline)."),

        ("Q5: Why is Structure of Arrays (SoA) preferred over Array of Structures (AoS)?",
         "In AoS, vectors are interleaved (u0, v0, w0...), requiring strided reads and register shuffles for SIMD loads. In SoA, u, v, and w are stored contiguously. A single 256-bit instruction (_mm256_load_pd) streams 4 consecutive values directly into AVX2 registers with unit stride, yielding a 1.45× speedup."),

        ("Q6: Why is Q-criterion preferred over vorticity for vortex identification?",
         "Vorticity (ω = ∇ × u) measures fluid rotation but cannot differentiate between vortex cores and pure shear. In pure Couette shear (u = γ·y), vorticity is non-zero (ω_z = -γ), yet there is no vortex. Q-criterion decomposes the velocity gradient into strain S and rotation Ω: Q = 0.5(||Ω||² - ||S||²). In pure shear, ||Ω||² = ||S||², yielding Q ≡ 0 identically. Where Q > 0, rotation dominates strain, correctly isolating vortex tubes."),

        ("Q7: Why did you choose collapse(2) instead of collapse(3) for 3D stencils?",
         "collapse(2) combines the outer two loops (i, j) into Nx × Ny = 16,384 iterations, providing abundant parallelism across threads. Crucially, leaving the innermost k-loop un-collapsed preserves contiguous unit-stride memory access (Δz = 1), enabling the compiler to auto-vectorize with #pragma omp simd without integer division overhead."),

        ("Q8: How do you achieve zero-copy data transfer between Python and C++?",
         "Using pybind11's buffer protocol, we extract the raw 64-bit memory pointer directly from NumPy's C-contiguous array without memory allocation. We release Python's GIL (py::gil_scoped_release) during execution so that OpenMP threads run on all CPU cores at 100% capacity without Python interpreter lock contention.")
    ]

    for q, a in qa_list:
        story.append(Paragraph(f"<b>{q}</b>", h3_style))
        story.append(Paragraph(a, body_style))
        story.append(Spacer(1, 4))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[PDF Generator] Successfully created {filename}")
    return filename

if __name__ == "__main__":
    out = build_pdf()
    print("Done generating PDF.")
