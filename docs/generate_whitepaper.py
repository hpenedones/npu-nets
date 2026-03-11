#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate the TileFlow white paper as a PDF.

Combines explanatory text with the architecture diagrams into a
single document suitable for reading as an introduction to the project.

Usage::

    python docs/generate_whitepaper.py

Requires: weasyprint, matplotlib (both in the project .venv).
"""

from pathlib import Path
from weasyprint import HTML

DOCS_DIR = Path(__file__).parent

HTML_CONTENT = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page {
    size: A4;
    margin: 2.5cm 2cm;
    @bottom-center { content: counter(page); font-size: 9pt; color: #888; }
  }
  body {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #222;
    max-width: 100%;
  }
  h1 {
    font-size: 22pt;
    color: #1a1a2e;
    border-bottom: 3px solid #4A90D9;
    padding-bottom: 8px;
    margin-top: 0;
  }
  h2 {
    font-size: 15pt;
    color: #2c3e50;
    margin-top: 1.5em;
    border-bottom: 1px solid #ddd;
    padding-bottom: 4px;
  }
  h3 {
    font-size: 12pt;
    color: #34495e;
    margin-top: 1.2em;
  }
  .subtitle {
    font-size: 13pt;
    color: #555;
    margin-top: -0.5em;
    margin-bottom: 1.5em;
  }
  .authors {
    font-size: 10pt;
    color: #777;
    margin-bottom: 2em;
  }
  code {
    font-family: 'Courier New', monospace;
    font-size: 9.5pt;
    background: #f5f5f5;
    padding: 1px 4px;
    border-radius: 3px;
  }
  pre {
    background: #f8f8f8;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 12px;
    font-size: 9pt;
    line-height: 1.4;
    overflow-x: auto;
  }
  .figure {
    text-align: center;
    margin: 1.5em 0;
    page-break-inside: avoid;
  }
  .figure img {
    max-width: 100%;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
  }
  .figure .caption {
    font-size: 9.5pt;
    color: #555;
    margin-top: 0.5em;
    font-style: italic;
  }
  table {
    border-collapse: collapse;
    margin: 1em 0;
    width: 100%;
    font-size: 10pt;
  }
  th, td {
    border: 1px solid #ddd;
    padding: 6px 10px;
    text-align: left;
  }
  th {
    background: #4A90D9;
    color: white;
    font-weight: 600;
  }
  tr:nth-child(even) { background: #f9f9f9; }
  .highlight { background: #fff3cd; padding: 10px; border-radius: 4px;
               border-left: 4px solid #F5A623; margin: 1em 0; }
  .key-insight { background: #d4edda; padding: 10px; border-radius: 4px;
                 border-left: 4px solid #2ECC71; margin: 1em 0; }
  .warning { background: #f8d7da; padding: 10px; border-radius: 4px;
             border-left: 4px solid #E74C3C; margin: 1em 0; }
  a.gref {
    color: #2c3e50;
    text-decoration: none;
    border-bottom: 1px dotted #4A90D9;
  }
  a.gref:hover {
    color: #4A90D9;
    border-bottom: 1px solid #4A90D9;
  }
  .vocab-term {
    font-weight: 600;
    color: #2c3e50;
  }
</style>
</head>
<body>

<h1>TileFlow: Spatial Neural Networks on AMD XDNA&nbsp;2 NPU</h1>
<p class="subtitle">
  Hardware-software co-design for close-to-metal neural network inference
</p>
<p class="authors">
  Built with <a href="https://github.com/amd/IRON">IRON/MLIR-AIE</a> toolchain
  &mdash; Target: AMD Ryzen AI 9 HX 370 (Strix Point)
</p>

<h2>1. Introduction</h2>

<p>
Modern neural processing units (NPUs) are spatial dataflow computers: instead of
a single CPU with caches and a register file, they expose a 2D array of small
compute tiles, each with its own SRAM and SIMD units, connected by a
programmable interconnect. This architecture is extremely efficient for
<em>on-chip</em> computation &mdash; but only if the software maps the
algorithm directly to the physical hardware.
</p>

<p>
<strong>TileFlow</strong> takes this literally. We design a neural network whose
architecture is dictated by the physical tile layout of the AMD XDNA&nbsp;2
<a href="#g-npu" class="gref">NPU</a>. The network has learnable parameters
and non-linearities (<a href="#g-relu" class="gref">ReLU</a>), but its structure
(number of layers, block grouping, buffer sizes) matches the hardware exactly.
</p>

<p>
Our main result is a <strong>block-recurrent character language model</strong>
with 32 layers grouped into 8 blocks of 4. Each block maps to a 4-stage pipeline
on the NPU (one layer per compute row, 8 columns in parallel). The model achieves
validation loss 2.42 (perplexity 11.2) on Shakespeare text and generates
89,600 characters/second on the NPU. In a supporting throughput benchmark,
a recurrent <a href="#g-mlp" class="gref">MLP</a> with a single shared weight
achieves <strong>23.93 <a href="#g-tflops" class="gref">TFLOPS</a></strong>
sustained &mdash; 95.7% of the 25 TFLOPS theoretical peak and
<strong>26&times; speedup over CPU</strong>.
</p>

<div class="key-insight">
<strong>Key principle:</strong> We design the network to match the hardware,
not the other way around. Any architecture with learnable parameters and
non-linearities can learn &mdash; so we choose the one that maximizes
hardware utilization.
</div>

<h2>2. Background</h2>

<p>
This section provides the hardware and systems context that most ML engineers
never need to think about &mdash; until they want to understand <em>why</em>
certain hardware is fast (or slow) for their models. If you&rsquo;ve trained
models with PyTorch or JAX and think of hardware as &ldquo;a GPU with some
VRAM,&rdquo; this section fills in the gap between that abstraction and the
physical reality of a spatial processor like the XDNA&nbsp;2 NPU.
</p>

<h3>2.1 Glossary of acronyms</h3>

<p>
The table below defines every acronym used in this paper. Each entry has an
<code>id</code> so that occurrences in the text link back here. We group them
by domain so you can revisit this section as a reference while reading.
</p>

<table>
  <tr><th>Acronym</th><th>Stands for</th><th>What it means</th></tr>

  <tr><td colspan="3" style="background:#e8e8e8;font-weight:600;">
    Hardware &amp; memory</td></tr>
  <tr id="g-npu"><td>NPU</td><td>Neural Processing Unit</td>
      <td>A dedicated accelerator for neural-network inference (and
          sometimes training), built into a laptop or phone chip.</td></tr>
  <tr id="g-apu"><td>APU</td><td>Accelerated Processing Unit</td>
      <td>AMD&rsquo;s name for a single die that integrates CPU + GPU + NPU.</td></tr>
  <tr id="g-xdna"><td>XDNA</td><td>(AMD brand name)</td>
      <td>AMD&rsquo;s NPU architecture family. XDNA&nbsp;2 is the second
          generation, found in &ldquo;Strix Point&rdquo; Ryzen AI chips.</td></tr>
  <tr id="g-aie"><td>AIE</td><td>AI Engine</td>
      <td>The individual tile processor IP inside the XDNA NPU, originally
          designed by Xilinx (acquired by AMD).</td></tr>
  <tr id="g-sram"><td>SRAM</td><td>Static Random-Access Memory</td>
      <td>Fast, on-chip memory (~1&ndash;2 ns access). Each compute tile has
          ~64 KB of SRAM. Expensive per bit, but extremely fast because it
          sits right next to the compute logic.</td></tr>
  <tr id="g-dram"><td>DRAM</td><td>Dynamic Random-Access Memory</td>
      <td>The main system memory (8&ndash;64 GB). Much slower than SRAM
          (~50&ndash;100 ns access) but vastly cheaper per bit.</td></tr>
  <tr id="g-ddr"><td>DDR</td><td>Double Data Rate</td>
      <td>The interface standard for DRAM modules. &ldquo;DDR memory&rdquo; is
          the system RAM your laptop uses. In this paper, &ldquo;DDR&rdquo;
          means &ldquo;host-side main memory.&rdquo;</td></tr>
  <tr id="g-dma"><td>DMA</td><td>Direct Memory Access</td>
      <td>A hardware mechanism that copies data between memory regions
          <em>without using the CPU</em>. The NPU&rsquo;s shim tiles contain
          DMA engines that move data between DDR and tile SRAM.</td></tr>
  <tr id="g-pcie"><td>PCIe</td><td>Peripheral Component Interconnect Express</td>
      <td>The high-speed bus connecting the NPU to the rest of the system.</td></tr>

  <tr><td colspan="3" style="background:#e8e8e8;font-weight:600;">
    Compute concepts</td></tr>
  <tr id="g-simd"><td>SIMD</td><td>Single Instruction, Multiple Data</td>
      <td>A processor executes one instruction on a <em>vector</em> of values
          simultaneously. For example, multiplying 32 numbers in one clock
          cycle instead of one at a time. This is how GPUs and NPUs achieve
          massive throughput.</td></tr>
  <tr id="g-vliw"><td>VLIW</td><td>Very Long Instruction Word</td>
      <td>A processor design where each instruction encodes <em>multiple
          operations</em> to execute in parallel (e.g., a multiply, an add,
          and a load all in one cycle). Each AIE tile uses a VLIW core.</td></tr>
  <tr id="g-mmul"><td>MMUL</td><td>Matrix Multiply (unit)</td>
      <td>A hardware block dedicated to multiplying small matrices (e.g.,
          8&times;8 blocks of bfloat16). This is the workhorse of each AIE
          tile.</td></tr>
  <tr id="g-fifo"><td>FIFO</td><td>First In, First Out</td>
      <td>A queue where data is read in the same order it was written. The
          NPU uses hardware FIFOs (&ldquo;ObjectFIFOs&rdquo;) to stream data
          between tiles, like Unix pipes between processes.</td></tr>
  <tr id="g-gemm"><td>GEMM</td><td>General Matrix Multiply</td>
      <td>The standard linear algebra operation C = A &times; B + C. Neural
          network layers are essentially sequences of GEMMs with
          non-linearities in between.</td></tr>

  <tr><td colspan="3" style="background:#e8e8e8;font-weight:600;">
    Numeric formats</td></tr>
  <tr id="g-bf16"><td>bf16 / bfloat16</td><td>Brain Floating Point, 16-bit</td>
      <td>A 16-bit floating-point format with the same exponent range as
          float32 (8 bits) but less precision (7-bit mantissa vs 23-bit).
          Invented at Google Brain for ML training where range matters more
          than precision.</td></tr>
  <tr id="g-bfp16"><td>BFP16</td><td>Block Floating Point, 16-bit</td>
      <td>An emulation mode on AIE2P that groups bf16 values into blocks
          sharing a common exponent. Enables efficient SIMD matmul in the
          MMUL unit with tile factor r = s = t = 8.</td></tr>
  <tr id="g-int8"><td>INT8</td><td>8-bit Integer</td>
      <td>8-bit integer arithmetic, used for quantized inference. The
          NPU&rsquo;s peak in INT8 mode is 50 TOPS (double the bf16
          rate).</td></tr>

  <tr><td colspan="3" style="background:#e8e8e8;font-weight:600;">
    Performance metrics</td></tr>
  <tr id="g-flops"><td>FLOPS</td><td>Floating-point Operations Per Second</td>
      <td>The standard measure of compute throughput. One multiply-add on
          two numbers counts as 2 FLOPs.</td></tr>
  <tr id="g-gflops"><td>GFLOPS</td><td>Giga&nbsp;FLOPS (10<sup>9</sup>)</td>
      <td>Billions of floating-point operations per second.</td></tr>
  <tr id="g-tflops"><td>TFLOPS</td><td>Tera&nbsp;FLOPS (10<sup>12</sup>)</td>
      <td>Trillions of floating-point operations per second. The NPU&rsquo;s
          peak is 25 TFLOPS in bfloat16.</td></tr>
  <tr id="g-tops"><td>TOPS</td><td>Tera Operations Per Second</td>
      <td>Like TFLOPS but for integer operations. Used for INT8 peak specs
          (50 TOPS for this NPU).</td></tr>

  <tr><td colspan="3" style="background:#e8e8e8;font-weight:600;">
    Machine learning</td></tr>
  <tr id="g-mlp"><td>MLP</td><td>Multi-Layer Perceptron</td>
      <td>A neural network made of fully-connected (dense) layers. Each
          layer computes y = activation(x &times; W + b).</td></tr>
  <tr id="g-relu"><td>ReLU</td><td>Rectified Linear Unit</td>
      <td>The activation function max(x, 0). Simple, cheap to compute, and
          widely used.</td></tr>

  <tr><td colspan="3" style="background:#e8e8e8;font-weight:600;">
    Toolchain &amp; software</td></tr>
  <tr id="g-iron"><td>IRON</td><td>(name, not an acronym)</td>
      <td>AMD&rsquo;s Python API for programming AIE tiles at a high level:
          defining kernels, ObjectFIFOs, and tile placements.</td></tr>
  <tr id="g-mlir"><td>MLIR</td><td>Multi-Level Intermediate Representation</td>
      <td>A compiler framework (from the LLVM project) that represents
          programs at multiple abstraction levels. MLIR-AIE is the dialect
          that targets AIE hardware.</td></tr>
  <tr id="g-xrt"><td>XRT</td><td>Xilinx Runtime</td>
      <td>The userspace library and kernel driver that loads bitstreams onto
          the NPU and manages execution.</td></tr>
  <tr id="g-xclbin"><td>XCLBIN</td><td>Xilinx Container for Linux Binary</td>
      <td>The compiled binary file that contains the NPU bitstream (tile
          configuration, routing, kernel code).</td></tr>
  <tr id="g-llvm"><td>LLVM</td><td>Low Level Virtual Machine</td>
      <td>A widely-used compiler infrastructure. Peano/LLVM-AIE is a fork
          that compiles C++ to AIE tile machine code.</td></tr>
  <tr id="g-bd"><td>BD</td><td>Buffer Descriptor</td>
      <td>A hardware structure in the DMA engine that describes one data
          transfer: source address, size, stride pattern. The 10-bit size
          field (max 1024) is a constraint discussed in Section 4.3.</td></tr>
</table>

<h3>2.2 Key vocabulary (non-acronyms)</h3>

<p>
Several important terms in this paper are not acronyms but may be unfamiliar
to readers who have worked above the hardware abstraction layer:
</p>

<table>
  <tr><th style="width:120px;">Term</th><th>What it means</th></tr>
  <tr id="v-shim"><td class="vocab-term">Shim tile</td>
      <td>The row of interface tiles at the bottom of the NPU array (row 0).
          &ldquo;Shim&rdquo; means a thin adapter layer &mdash; these tiles
          <em>bridge</em> two different worlds: the system&rsquo;s DDR memory
          (accessed via PCIe/AXI bus) and the NPU&rsquo;s internal tile
          interconnect. Each shim tile contains DMA engines that translate
          between host memory addresses and on-chip tile coordinates.
          They do no computation; they are pure data movers.</td></tr>
  <tr id="v-tile"><td class="vocab-term">Tile</td>
      <td>A self-contained processing unit with its own local memory (SRAM),
          instruction memory, and compute logic. Unlike CPU cores that share
          caches, each tile is an <em>independent computer</em> that communicates
          with neighbours through explicit data channels (FIFOs), not shared
          memory.</td></tr>
  <tr id="v-bitstream"><td class="vocab-term">Bitstream</td>
      <td>The compiled binary that programs the NPU&rsquo;s hardware
          configuration: which tiles run which code, how the interconnect routes
          data, and what DMA transfers to perform. Analogous to a GPU shader
          binary, but configuring a full spatial processor rather than a single
          shader core. Packaged in an XCLBIN file.</td></tr>
  <tr id="v-kernel"><td class="vocab-term">Kernel (NPU)</td>
      <td>A C++ function compiled to run on a single tile. Not to be confused
          with an OS kernel or a CUDA kernel (which runs on thousands of GPU
          threads). An NPU kernel runs on exactly <em>one</em> tile&rsquo;s
          VLIW+SIMD core and has direct access to that tile&rsquo;s SRAM.</td></tr>
  <tr id="v-pipeline"><td class="vocab-term">Pipeline</td>
      <td>A technique where multiple stages of a task overlap in time, like
          an assembly line. While stage N processes item <em>i</em>, stage N+1
          processes item <em>i&minus;1</em>. The chess compiler pipelines the
          inner loops of our matmul kernel so that memory loads, multiplications,
          and stores overlap across iterations.</td></tr>
  <tr id="v-register"><td class="vocab-term">Register</td>
      <td>The smallest, fastest storage inside a processor core &mdash; a few
          hundred bytes at most, accessible in a single clock cycle. The MMUL
          unit accumulates partial matrix products in registers before writing
          back to SRAM. Our fused kernel zero-initialises these accumulators
          in registers, avoiding a separate SRAM clear pass.</td></tr>
  <tr id="v-memwall"><td class="vocab-term">Memory wall</td>
      <td>The growing gap between processor speed and memory speed. CPUs can
          execute arithmetic ~100&times; faster than DRAM can supply data. This
          is <em>the</em> fundamental reason NPUs (with data-local SRAM) can be
          dramatically faster than CPUs for the right workloads.</td></tr>
  <tr id="v-interconnect"><td class="vocab-term">Interconnect</td>
      <td>The on-chip wiring and switches that connect tiles to each other and
          to memory tiles. Unlike a CPU&rsquo;s shared bus, the NPU interconnect
          is <em>circuit-switched</em>: routes are configured at compile time,
          giving dedicated bandwidth with no contention.</td></tr>
  <tr id="v-throughput"><td class="vocab-term">Throughput vs Latency</td>
      <td><strong>Latency</strong> is how long one operation takes (e.g., one
          matmul = 1 &mu;s). <strong>Throughput</strong> is how many operations
          complete per second (e.g., 24 tiles &times; 1000 matmuls = 24M matmuls/s).
          The NPU excels at throughput through massive parallelism, even if
          single-tile latency is similar to a CPU.</td></tr>
  <tr id="v-doublebuf"><td class="vocab-term">Double buffering</td>
      <td>Using two buffers that swap roles each iteration: one is being read
          by the compute unit while the other is being filled by DMA.
          Also called &ldquo;ping-pong buffering.&rdquo; This hides data
          transfer latency behind computation.</td></tr>
</table>

<h3>2.3 How CPUs execute neural networks (and why it&rsquo;s slow)</h3>

<p>
When you call <code>y = torch.relu(x @ W)</code> on a CPU, here is what
actually happens at the hardware level:
</p>

<ol>
  <li>The weight matrix <code>W</code> lives in <a href="#g-dram" class="gref">DRAM</a>.
      The CPU issues a load instruction, and the data travels through the
      <strong><a href="#v-memwall" class="gref">memory hierarchy</a></strong>:
      DRAM &rarr; L3 cache &rarr; L2 cache &rarr; L1 cache &rarr;
      <a href="#v-register" class="gref">registers</a>.</li>
  <li>The CPU&rsquo;s <a href="#g-simd" class="gref">SIMD</a> units multiply a
      few rows at a time. The result goes back through the cache hierarchy to
      DRAM.</li>
  <li>For the next layer, the process repeats: read the activation from DRAM,
      read the next weight matrix from DRAM, compute, write back.</li>
</ol>

<p>
The fundamental problem is the <strong><a href="#v-memwall" class="gref">memory
wall</a></strong>: modern compute units can perform arithmetic far faster than
memory can supply data. A typical CPU can do ~1
<a href="#g-tflops" class="gref">TFLOPS</a> of
<a href="#g-bf16" class="gref">bf16</a> math, but its
<a href="#g-ddr" class="gref">DDR</a> bandwidth is ~50 GB/s. To keep the ALUs
busy doing a 128&times;128 matmul, you need 128&times;128&times;2 = 32 KB of
weight data delivered every ~1 &mu;s &mdash; which is 32 GB/s just for one
matrix. If you&rsquo;re running 4 layers, that&rsquo;s 128 GB/s, already
exceeding the memory bandwidth. The CPU stalls waiting for data.
</p>

<div class="highlight">
<strong>The memory wall in one sentence:</strong> Arithmetic is cheap; moving
data is expensive. The deeper you go in a neural network, the more time is
spent shuffling data between memory levels, not doing useful math.
</div>

<h3>2.4 The spatial dataflow alternative</h3>

<p>
A <strong>spatial architecture</strong> takes a radically different approach.
Instead of one fast processor with a deep cache hierarchy, it uses <em>many
small processors</em> (<a href="#v-tile" class="gref">tiles</a>), each with a
tiny but <em>extremely fast</em> local memory
(<a href="#g-sram" class="gref">SRAM</a>). Data moves between tiles through
dedicated hardware channels (<a href="#g-fifo" class="gref">FIFOs</a>), not
through shared caches.
</p>

<p>
Think of it like a factory assembly line versus a single master craftsman:
</p>

<ul>
  <li><strong>CPU (craftsman):</strong> One highly skilled worker goes to the
      warehouse (<a href="#g-dram" class="gref">DRAM</a>) for each part, brings
      it to the workbench (registers), processes it, takes the result back to the
      warehouse, gets the next part. Most time is spent walking, not working.</li>
  <li><strong><a href="#g-npu" class="gref">NPU</a> (assembly line):</strong>
      Many workers sit at their stations with all their parts already on their
      desk (SRAM). A conveyor belt (FIFO) moves the workpiece from one station
      to the next. Nobody walks anywhere.</li>
</ul>

<p>
The key property is <strong>data locality</strong>: once data is loaded into a
tile&rsquo;s 64 KB SRAM, it stays there for as many operations as you can do
on it. There is no cache to get evicted from, no bus to contend for. If a
128&times;128 weight matrix (32 KB) fits in SRAM, you can multiply against it
thousands of times at full speed &mdash; which is exactly what our recurrent
<a href="#g-mlp" class="gref">MLP</a> does.
</p>

<h3>2.5 Key hardware concepts used in this project</h3>

<p>
These concepts appear throughout the paper. You don&rsquo;t need to understand
every transistor, but knowing these ideas will make the design choices clear:
</p>

<h4><a href="#v-doublebuf" class="gref">Double buffering</a> (ping-pong)</h4>
<p>
If a tile needs to <em>receive</em> new data while <em>computing</em> on data
it already has, you need two buffers: one being filled by
<a href="#g-dma" class="gref">DMA</a> while the other is being read by the
compute unit. They swap roles each iteration. We use this for the activation
buffers: buffer A holds the input, the tile computes into buffer B, then B
becomes the input and A becomes the output.
</p>

<h4>Tiled matrix layout</h4>
<p>
A &ldquo;row-major&rdquo; matrix stores elements left-to-right, top-to-bottom:
<code>[[a, b], [c, d]]</code> becomes <code>[a, b, c, d]</code>. The
<a href="#g-aie" class="gref">AIE</a> matmul unit instead expects a
<strong>blocked (tiled) layout</strong>: the matrix is divided into 8&times;8
sub-matrices, and each block is stored contiguously. This matches the
<a href="#g-mmul" class="gref">MMUL</a> hardware which multiplies 8&times;8
blocks in one operation. The <code>to_tiled()</code> function handles this
conversion.
</p>

<h4><a href="#g-fifo" class="gref">ObjectFIFOs</a> and data movement</h4>
<p>
On a CPU, data movement is implicit &mdash; you <code>load</code> from an
address and the cache hierarchy handles the rest. On the NPU, you must
<em>explicitly</em> program every data transfer: &ldquo;move 4 KB from DDR
address X to tile (3, 2)&rsquo;s input buffer.&rdquo;
<a href="#g-iron" class="gref">IRON</a>&rsquo;s <code>ObjectFifo</code>
abstraction makes this manageable: you declare a typed channel between a
producer and a consumer, and the compiler generates the DMA configurations.
</p>

<h4>The invocation overhead problem</h4>
<p>
Every time the host CPU tells the NPU to run, there is a fixed overhead of
~120 &mu;s for driver calls, instruction dispatch, and DMA setup. This is
analogous to the overhead of launching a CUDA kernel on a GPU. If your actual
compute takes only 1 &mu;s (as it does for a single small matmul), you are
spending 99% of the time on overhead. The solution is to do <em>lots of work
per invocation</em> &mdash; hence the hardware loop that repeats thousands of
matmuls before returning to the host.
</p>

<h3>2.6 How to read the rest of this paper</h3>

<p>
With this background, the rest of the paper should be accessible:
</p>

<ul>
  <li><strong>Section 3</strong> describes the physical hardware: the tile
      grid, memory sizes, and interconnect.</li>
  <li><strong>Section 4</strong> presents the neural network architecture and
      explains why each design choice follows from the hardware constraints.</li>
  <li><strong>Section 5</strong> shows throughput and speedup results.</li>
  <li><strong>Section 6</strong> describes the software toolchain.</li>
  <li><strong>Section 7</strong> maps the code structure to the concepts.</li>
</ul>

<h2>3. The Hardware</h2>

<p>
The AMD <a href="#g-xdna" class="gref">XDNA</a>&nbsp;2
<a href="#g-npu" class="gref">NPU</a> in the Ryzen AI 9 HX 370 (codename
Strix Point) is a tiled spatial-dataflow processor with the following structure:
</p>

<div class="figure">
  <img src="xdna2_hardware.png" alt="XDNA 2 tile array">
  <div class="caption">
    Figure 1: Physical tile array of the AMD XDNA&nbsp;2 NPU. 32 compute
    <a href="#v-tile" class="gref">tiles</a> (rows 2&ndash;5) each contain
    ~64 KB <a href="#g-sram" class="gref">SRAM</a> and a
    <a href="#g-bf16" class="gref">bf16</a>
    <a href="#g-mmul" class="gref">MMUL</a> unit. 8 memory tiles (row 1,
    512 KB each) serve as on-chip L2 buffers and routing hubs. 8
    <a href="#v-shim" class="gref">shim tiles</a> (row 0) provide
    <a href="#g-dma" class="gref">DMA</a> access to host
    <a href="#g-ddr" class="gref">DDR</a> memory.
  </div>
</div>

<table>
  <tr><th>Property</th><th>Value</th></tr>
  <tr><td>Compute tiles</td><td>32 (8 columns &times; 4 rows)</td></tr>
  <tr><td>Memory tiles</td><td>8 (512 KB each, 4 MB total)</td></tr>
  <tr><td>Per-tile <a href="#g-sram" class="gref">SRAM</a></td><td>~64 KB data memory</td></tr>
  <tr><td>Per-tile compute</td><td><a href="#g-bf16" class="gref">bf16</a> <a href="#g-mmul" class="gref">MMUL</a> unit, <a href="#g-vliw" class="gref">VLIW</a>+<a href="#g-simd" class="gref">SIMD</a> core</td></tr>
  <tr><td>Clock frequency</td><td>~1.5 GHz</td></tr>
  <tr><td>Peak <a href="#v-throughput" class="gref">throughput</a></td><td><strong>25 <a href="#g-tflops" class="gref">TFLOPS</a></strong> (bfloat16)</td></tr>
  <tr><td><a href="#v-interconnect" class="gref">Interconnect</a></td><td>ObjectFIFOs (tile-to-tile <a href="#v-doublebuf" class="gref">double-buffered</a> streams)</td></tr>
  <tr><td>Power envelope</td><td>~6 W</td></tr>
</table>

<h3>3.1 Why spatial dataflow matters</h3>

<p>
On a CPU, a matrix multiply reads data from DRAM through multiple cache levels.
Each layer of a neural network bounces activations through L1&rarr;L2&rarr;L3&rarr;DRAM
and back. On the NPU, data stays in 64 KB tile SRAM between operations &mdash;
zero cache misses, zero bus contention. This is the source of the NPU's advantage
for deep, narrow computations.
</p>

<div class="highlight">
<strong>Lesson from Phase 1:</strong> A single large <a href="#g-gemm" class="gref">GEMM</a>
achieves only 2.49 TFLOPS on the NPU (10% of peak) because it is
<em>memory-bandwidth limited</em> &mdash; data must stream from DDR. The NPU
wins when data <strong>stays on-chip</strong>.
</div>

<h2>4. The Architecture</h2>

<h3>4.1 Throughput benchmark: Recurrent MLP</h3>

<p>
Our first architecture is a recurrent <a href="#g-mlp" class="gref">MLP</a>:
a single weight matrix <code>W</code> (128&times;128,
<a href="#g-bf16" class="gref">bfloat16</a>) is loaded once into each tile's
<a href="#g-sram" class="gref">SRAM</a> and applied repeatedly in a tight
hardware loop. A <strong>fused <a href="#v-kernel" class="gref">kernel</a></strong>
computes C&nbsp;=&nbsp;<a href="#g-relu" class="gref">ReLU</a>(A&nbsp;&times;&nbsp;B)
in a single call:
</p>

<pre>
x = input                     # loaded from DDR, 48&times;128 bf16
for i in range(num_iters):
    y = ReLU(x @ W)           # fused matmul + activation (one kernel call)
    x = ReLU(y @ W)           # ping-pong: result goes back to x
output = x                    # drained to DDR
</pre>

<p>
This is mapped to 24 compute tiles (3 rows &times; 8 columns), each running the
same loop independently on different input samples:
</p>

<div class="figure">
  <img src="recurrent_mlp.png" alt="Recurrent MLP mapped to NPU">
  <div class="caption">
    Figure 2: Recurrent MLP mapped to the XDNA&nbsp;2 tile array. 24 tiles across
    3 compute rows run the same hardware loop in parallel. Row 5 is unused due to
    MemTile routing constraints (~6 northward master ports). The detail box shows
    the per-tile computation: acquire buffers, loop, copy result.
  </div>
</div>

<h3>4.2 Main model: Block-Recurrent Character LM</h3>

<p>
The recurrent MLP demonstrates near-peak throughput but uses a single shared weight
matrix &mdash; too simple for a real ML task. The <strong>block-recurrent</strong>
architecture extends this with 32 <em>distinct</em> weight matrices, grouped into
8 blocks of 4 layers. Each block maps to one NPU pipeline call (4 rows in one column).
Between blocks, the CPU applies normalization, embedding injection, and residual
connections:
</p>

<pre>
for each character:
    for each block g = 0..7:
        CPU:  h_b = RMSNorm(h) + embed(char) + bias_g
        NPU:  h_b = ReLU(ReLU(ReLU(ReLU(h_b @ W[4g]) @ W[4g+1]) @ W[4g+2]) @ W[4g+3])
        CPU:  h = h + h_b   (residual connection)
    CPU:  h = RMSNorm(h)       (post-norm)
    CPU:  logits = h @ W_out + b_out
</pre>

<p>
This design is a deliberate trade-off: per-layer normalization and input injection
(which cannot run on the NPU pipeline) would give better quality (val loss 1.94)
but require CPU intervention between every matmul. The blocked design sacrifices
some quality (val loss 2.42) for exact NPU mapping &mdash; within each block,
the 4-stage matmul+ReLU chain executes entirely in tile SRAM with no DDR traffic.
</p>

<h3>4.3 Why these architectures</h3>

<ul>
  <li><strong>Maximizes on-chip time:</strong> Weight is loaded once (32 KB) and
      reused for thousands of matmul operations. <a href="#g-ddr" class="gref">DDR</a>
      I/O happens only at start and end.</li>
  <li><strong>Amortizes overhead:</strong> Each NPU invocation has ~120 &mu;s of
      driver/<a href="#g-dma" class="gref">DMA</a> overhead. With depth=1000,
      compute time dominates overhead by &gt;10&times;.</li>
  <li><strong>Fits <a href="#g-sram" class="gref">SRAM</a> budget:</strong>
      W (32 KB) + 2 activation buffers (12 KB each at B=48) + stack (1 KB) =
      57 KB, within the 64 KB tile limit. (B=64 would need 65 KB and fails.)</li>
  <li><strong>Linear tile scaling:</strong> Each tile is independent &mdash;
      doubling tiles doubles <a href="#v-throughput" class="gref">throughput</a>
      with no communication overhead.</li>
</ul>

<h3>4.4 Multi-row data routing</h3>

<p>
When using more than 8 tiles (i.e., more than one compute row), data must pass
through the MemTiles (row 1) which act as routing hubs:
</p>

<ul>
  <li><strong>Weights</strong> are <em>broadcast</em> via <code>forward()</code>:
      one DDR&rarr;MemTile transfer, then the MemTile fans out to all compute
      rows in the column.</li>
  <li><strong>Inputs</strong> are <em>split</em> via <code>split()</code>: the
      host buffer is partitioned so each row gets its own batch slice.</li>
  <li><strong>Outputs</strong> are <em>joined</em> via <code>join()</code>: per-row
      results are aggregated back through the MemTile to DDR.</li>
</ul>

<div class="warning">
<strong>Routing limit:</strong> Each MemTile has approximately 6 northward
master ports. Our design requires 3 data streams per row (weight + input + output).
At 3 compute rows = 9 streams, this fits; at 4 rows = 12 streams, the MLIR-AIE
router fails. This caps us at <strong>24 tiles</strong> (3 rows &times; 8 columns).
</div>

<h3>4.5 Critical implementation constraints</h3>

<p>
Several non-obvious hardware constraints shaped the design:
</p>

<ul>
  <li><strong>No <a href="#g-fifo" class="gref">FIFO</a> ops inside loops:</strong>
      Placing <code>acquire()</code> / <code>release()</code> inside
      <code>range_()</code> (which compiles to <code>scf.for</code>) causes DMA
      deadlock. All FIFO operations must happen <em>outside</em> the loop.</li>
  <li><strong><a href="#g-bd" class="gref">DMA BD</a> 10-bit size limit:</strong>
      <a href="#v-shim" class="gref">Shim</a> DMA buffer-descriptor sizes are
      10-bit (max 1024). For B=48, H=128, the product B&times;H=6144 exceeds
      this, so tensor access patterns must factor dimensions as [B, H] = [48, 128]
      instead of [B&times;H] = [6144].</li>
  <li><strong>Fused kernel requires sufficient batch size:</strong> Our
      <code>matmul_relu</code> kernel fuses zero-init + matmul +
      <a href="#g-relu" class="gref">ReLU</a> into one call, eliminating the need
      for a separate zero pass. However, the chess compiler only schedules the
      fused <a href="#v-pipeline" class="gref">pipeline</a> efficiently when the
      outer loop has &ge;3 iterations (M/(2r) &ge; 3), meaning B &ge; 48 for
      r=8. At B=16 (1 iteration), the fused kernel is actually 10% <em>slower</em>
      than separate kernels.</li>
</ul>

<h2>5. Results</h2>

<div class="figure">
  <img src="performance.png" alt="Performance optimisation journey">
  <div class="caption">
    Figure 3: Optimisation journey showing
    <a href="#g-tflops" class="gref">TFLOPS</a> throughput (left) and
    CPU speedup (right) for each configuration. The fused kernel at B=48
    achieves 23.93 TFLOPS &mdash; 95.7% of the 25 TFLOPS theoretical peak.
  </div>
</div>

<table>
  <tr><th>Configuration</th><th>TFLOPS</th><th>Peak %</th>
      <th><a href="#g-gflops" class="gref">GFLOPS</a>/tile</th>
      <th>CPU Speedup</th></tr>
  <tr><td>B=16, separate kernels (baseline)</td>
      <td>8.98</td><td>35.9%</td><td>374</td>
      <td>18.5&times;</td></tr>
  <tr><td>B=32, separate kernels</td>
      <td>13.47</td><td>53.9%</td><td>561</td>
      <td>19.5&times;</td></tr>
  <tr><td>B=48, separate kernels</td>
      <td>15.98</td><td>63.9%</td><td>666</td>
      <td>17.6&times;</td></tr>
  <tr style="background:#d4edda;font-weight:600;">
      <td>B=48, fused matmul+ReLU kernel</td>
      <td>23.93</td><td>95.7%</td><td>997</td>
      <td>25.9&times;</td></tr>
</table>

<p>
All measurements use 24 tiles (3 rows &times; 8 columns), H=128, depth=1000,
<a href="#g-bf16" class="gref">bfloat16</a>.
</p>

<h3>5.2 Character language model</h3>

<p>
The block-recurrent char LM was trained on the tiny Shakespeare dataset (1.1 MB)
for 10 epochs on a GPU, then run on the NPU:
</p>

<table>
  <tr><th>Model</th><th>Params</th><th>Val Loss</th><th>Perplexity</th><th>Device</th></tr>
  <tr style="background:#d4edda;font-weight:600;">
      <td>Block-recurrent (8&times;4 pipeline)</td>
      <td>542K</td><td>2.42</td><td>11.2</td><td>NPU (32 tiles)</td></tr>
  <tr><td>Per-layer recurrent (norm every layer)</td>
      <td>542K</td><td>1.94</td><td>6.9</td><td>CPU/GPU only</td></tr>
  <tr><td>Transformer baseline</td>
      <td>818K</td><td>1.89</td><td>6.6</td><td>GPU</td></tr>
</table>

<p>
NPU inference performance for the block-recurrent model:
</p>

<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>NPU calls per character</td><td>8 (one per 4-layer block)</td></tr>
  <tr><td>Latency per NPU call</td><td>0.14 ms</td></tr>
  <tr><td>Throughput (1 sequence)</td><td>233 chars/s</td></tr>
  <tr><td>Throughput (384 parallel sequences)</td><td>89,600 chars/s</td></tr>
</table>

<p>
The quality gap between the blocked and per-layer models (2.42 vs 1.94) reflects
the cost of grouping layers: inter-layer normalization and input injection help
gradient flow and information access, but they require CPU intervention that breaks
the NPU pipeline. This is the fundamental hardware&ndash;quality trade-off.
</p>

<h3>5.3 Optimisation analysis (throughput benchmark)</h3>

<p>
Two independent levers drove the improvement from 8.98 to 23.93 TFLOPS:
</p>

<h4>Increasing batch size (B=16 &rarr; B=48)</h4>
<p>
The <a href="#g-mmul" class="gref">MMUL</a> unit processes 8&times;8 blocks using
a 2&times;2 expansion pattern. With B=16, the outer loop runs M/(2r) = 16/16 = 1
iteration &mdash; too few for the chess compiler to
<a href="#v-pipeline" class="gref">pipeline</a> effectively. At B=48, there are
3 iterations, giving the compiler enough &ldquo;runway&rdquo; to overlap loads,
multiplications, and stores. This alone improved throughput from 8.98 to
15.98 TFLOPS (+78%).
</p>

<h4>Fused matmul+<a href="#g-relu" class="gref">ReLU</a> kernel</h4>
<p>
The original approach used three separate kernel calls per matmul: (1) zero the
output buffer, (2) accumulate C&nbsp;+=&nbsp;A&times;B, (3) apply ReLU in-place.
Each call had dispatch overhead and the zero pass wasted cycles moving data
through <a href="#g-sram" class="gref">SRAM</a>.
</p>

<p>
Our fused <code>matmul_relu</code> kernel eliminates all three overheads:
accumulators are zero-initialised in <a href="#v-register" class="gref">registers</a>
(no SRAM write), the matmul accumulates into them, and ReLU is applied during
the store phase. At B=48, this boosted throughput from 15.98 to
<strong>23.93 TFLOPS</strong> (+50%) &mdash; because the compiler can schedule
the fused zero+MAC+ReLU pipeline when there are enough loop iterations.
</p>

<div class="key-insight">
<strong>Key insight:</strong> Kernel fusion and batch size interact
<em>non-linearly</em>. The fused kernel was 10% <em>slower</em> at B=16
(8.03 vs 8.98 TFLOPS) but 50% <em>faster</em> at B=48 (23.93 vs 15.98).
The compiler needs enough outer-loop iterations to schedule the additional
fused operations into the pipeline.
</div>

<h3>5.4 Remaining gap to peak</h3>

<p>
At 95.7% of the 25 TFLOPS theoretical peak, the remaining 4.3% (~1.1 TFLOPS)
is accounted for by:
</p>

<ul>
  <li><strong><a href="#g-bfp16" class="gref">BFP16</a> emulation overhead:</strong>
      The MMUL unit uses block-floating-point emulation for bf16 with r=s=t=8
      tile factors. This requires reformatting exponents, costing a few
      cycles per block.</li>
  <li><strong>Array utilization:</strong> We use 24 of 32 tiles (75%) due
      to MemTile routing constraints. The 8 unused tiles represent
      potential headroom if routing can be improved.</li>
  <li><strong>Invocation overhead:</strong> ~120 &mu;s of fixed driver/DMA
      setup per NPU call, amortised over ~1.3 ms of compute but still
      contributing ~8% overhead at depth=1000.</li>
</ul>

<h2>6. The Toolchain</h2>

<table>
  <tr><th>Component</th><th>Role</th></tr>
  <tr><td><a href="https://github.com/amd/IRON">IRON</a></td>
      <td>Python API for tile layout, ObjectFIFOs, and dataflow</td></tr>
  <tr><td><a href="https://github.com/Xilinx/mlir-aie">MLIR-AIE</a></td>
      <td>MLIR dialect &rarr; hardware bitstream compilation</td></tr>
  <tr><td><a href="https://github.com/Xilinx/llvm-aie">Peano/LLVM-AIE</a></td>
      <td>C++ compiler for per-tile kernels</td></tr>
  <tr><td><a href="https://github.com/amd/xdna-driver">XRT</a></td>
      <td>Runtime for loading and executing on the NPU</td></tr>
</table>

<h3>6.1 Compilation pipeline</h3>

<pre>
design.py  &xrarr;  MLIR  &xrarr;  aiecc  &xrarr;  .xclbin (bitstream)
                          &xrarr;  .bin   (instruction sequence)

matmul_relu.cc &xrarr;  mlp_mm_relu.o &xrarr;  mlp_kernels.a
mlp_kernels.cc &xrarr;  mlp_copy.o    &nearr;
</pre>

<h2>7. Code Structure</h2>

<p>
The project is intentionally minimal &mdash; four Python files and two C++ files:
</p>

<table>
  <tr><th>File</th><th>Lines</th><th>Purpose</th></tr>
  <tr><td><code>char_lm/model.py</code></td><td>~210</td>
      <td>Block-recurrent character LM (RecurrentCharLM)</td></tr>
  <tr><td><code>char_lm/train.py</code></td><td>~250</td>
      <td>GPU training loop (ROCm / CUDA), spectral norm clamping</td></tr>
  <tr><td><code>char_lm/generate.py</code></td><td>~215</td>
      <td>Text generation on CPU or NPU</td></tr>
  <tr><td><code>char_lm/data.py</code></td><td>~80</td>
      <td>Shakespeare dataset and vocabulary</td></tr>
  <tr><td><code>char_lm/transformer_baseline.py</code></td><td>~240</td>
      <td>GPT-style reference model for quality comparison</td></tr>
  <tr><td><code>spatial_mlp/__init__.py</code></td><td>~55</td>
      <td>Tiling utilities (<code>to_tiled</code>, <code>from_tiled</code>)</td></tr>
  <tr><td><code>spatial_mlp/pipeline_design.py</code></td><td>~290</td>
      <td><a href="#g-iron" class="gref">IRON</a> design: 32-tile pipeline
          (8 cols &times; 4 rows)</td></tr>
  <tr><td><code>spatial_mlp/pipeline_op.py</code></td><td>~130</td>
      <td>IRON operator: compilation + runtime buffers</td></tr>
  <tr><td><code>aie_kernels/matmul_relu.cc</code></td><td>~125</td>
      <td>Fused C&nbsp;=&nbsp;ReLU(A&nbsp;&times;&nbsp;B)
          <a href="#v-kernel" class="gref">kernel</a> for
          <a href="#g-aie" class="gref">AIE2P</a></td></tr>
  <tr><td><code>aie_kernels/mlp_kernels.cc</code></td><td>~55</td>
      <td>Support kernel: <code>copy_bf16</code> for result staging</td></tr>
</table>

<p>
The character LM module (<code>char_lm/</code>) handles training and generation.
The spatial MLP module (<code>spatial_mlp/</code>) provides the NPU pipeline
infrastructure used by the generator for on-device inference.
</p>

<h2>8. Future Work</h2>

<ul>
  <li><strong>Close the quality gap:</strong> The blocked model (val loss 2.42)
      lags the per-layer model (1.94) because normalization and input injection
      happen only between blocks, not between every layer. Custom NPU kernels
      that fuse norm+matmul+ReLU could enable per-layer operations within the
      pipeline, recovering quality without sacrificing NPU mapping.</li>
  <li><strong><a href="#g-int8" class="gref">INT8</a> mode:</strong> The
      NPU&rsquo;s peak is 50 <a href="#g-tops" class="gref">TOPS</a> for
      int8 &mdash; double the bf16 rate. With quantization-aware training, this
      could push effective throughput to ~48 TOPS and allow H=256 (fitting in
      the same SRAM budget).</li>
  <li><strong>Training on NPU:</strong> Research NPU backpropagation
      (see <a href="https://arxiv.org/html/2504.03083v1">arXiv:2504.03083</a>).
      The block structure could allow per-block gradient computation on-chip,
      with CPU handling the cross-block gradient flow.</li>
  <li><strong>Larger tasks:</strong> Apply the block-recurrent architecture to
      longer text datasets, music generation, or time-series forecasting where
      deep recurrence is natural and the NPU throughput advantage compounds
      over long sequences.</li>
</ul>

<h2>References</h2>

<ol>
  <li>AMD IRON repository: <a href="https://github.com/amd/IRON">github.com/amd/IRON</a></li>
  <li>MLIR-AIE programming guide:
      <a href="https://github.com/Xilinx/mlir-aie/tree/main/programming_guide">
      github.com/Xilinx/mlir-aie</a></li>
  <li>IRON tutorial (IPDPS 2025):
      <a href="https://www.amd.com/content/dam/amd/en/documents/products/processors/ryzen/ai/iron-for-ryzen-ai-tutorial-ipdps-2025.pdf">
      AMD Technical Paper</a></li>
  <li>NPU training: <a href="https://arxiv.org/html/2504.03083v1">arXiv:2504.03083</a></li>
  <li>Linux kernel NPU docs:
      <a href="https://docs.kernel.org/accel/amdxdna/amdnpu.html">kernel.org</a></li>
</ol>

</body>
</html>
"""


def generate():
    """Render the white paper HTML to PDF."""
    output_path = DOCS_DIR / "tileflow_whitepaper.pdf"
    html = HTML(string=HTML_CONTENT, base_url=str(DOCS_DIR))
    html.write_pdf(output_path)
    print(f"  ✓ {output_path}")
    return output_path


if __name__ == "__main__":
    print("Generating TileFlow white paper...")
    generate()
    print("Done.")
