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
architecture is dictated by the physical tile layout of the AMD XDNA&nbsp;2 NPU. The
network has learnable parameters (a shared weight matrix) and non-linearities
(ReLU), making it a valid machine-learning model &mdash; but its structure
(number of parallel paths, loop depth, buffer sizes) matches the hardware
exactly.
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
The table below defines every acronym used in this paper. We group them by
domain so you can revisit this section as a reference while reading.
</p>

<table>
  <tr><th>Acronym</th><th>Stands for</th><th>What it means</th></tr>

  <tr><td colspan="3" style="background:#e8e8e8;font-weight:600;">
    Hardware &amp; memory</td></tr>
  <tr><td>NPU</td><td>Neural Processing Unit</td>
      <td>A dedicated accelerator for neural-network inference (and
          sometimes training), built into a laptop or phone chip.</td></tr>
  <tr><td>APU</td><td>Accelerated Processing Unit</td>
      <td>AMD&rsquo;s name for a single die that integrates CPU + GPU + NPU.</td></tr>
  <tr><td>XDNA</td><td>(AMD brand name)</td>
      <td>AMD&rsquo;s NPU architecture family. XDNA&nbsp;2 is the second
          generation, found in &ldquo;Strix Point&rdquo; Ryzen AI chips.</td></tr>
  <tr><td>AIE</td><td>AI Engine</td>
      <td>The individual tile processor IP inside the XDNA NPU, originally
          designed by Xilinx (acquired by AMD).</td></tr>
  <tr><td>SRAM</td><td>Static Random-Access Memory</td>
      <td>Fast, on-chip memory (~1&ndash;2 ns access). Each compute tile has
          ~64 KB of SRAM. Expensive per bit, but extremely fast because it
          sits right next to the compute logic.</td></tr>
  <tr><td>DRAM</td><td>Dynamic Random-Access Memory</td>
      <td>The main system memory (8&ndash;64 GB). Much slower than SRAM
          (~50&ndash;100 ns access) but vastly cheaper per bit.</td></tr>
  <tr><td>DDR</td><td>Double Data Rate</td>
      <td>The interface standard for DRAM modules. &ldquo;DDR memory&rdquo; is
          the system RAM your laptop uses. In this paper, &ldquo;DDR&rdquo;
          means &ldquo;host-side main memory.&rdquo;</td></tr>
  <tr><td>DMA</td><td>Direct Memory Access</td>
      <td>A hardware mechanism that copies data between memory regions
          <em>without using the CPU</em>. The NPU&rsquo;s shim tiles contain
          DMA engines that move data between DDR and tile SRAM.</td></tr>
  <tr><td>PCIe</td><td>Peripheral Component Interconnect Express</td>
      <td>The high-speed bus connecting the NPU to the rest of the system.</td></tr>

  <tr><td colspan="3" style="background:#e8e8e8;font-weight:600;">
    Compute concepts</td></tr>
  <tr><td>SIMD</td><td>Single Instruction, Multiple Data</td>
      <td>A processor executes one instruction on a <em>vector</em> of values
          simultaneously. For example, multiplying 32 numbers in one clock
          cycle instead of one at a time. This is how GPUs and NPUs achieve
          massive throughput.</td></tr>
  <tr><td>VLIW</td><td>Very Long Instruction Word</td>
      <td>A processor design where each instruction encodes <em>multiple
          operations</em> to execute in parallel (e.g., a multiply, an add,
          and a load all in one cycle). Each AIE tile uses a VLIW core.</td></tr>
  <tr><td>MMUL</td><td>Matrix Multiply (unit)</td>
      <td>A hardware block dedicated to multiplying small matrices (e.g.,
          8&times;8 blocks of bfloat16). This is the workhorse of each AIE
          tile.</td></tr>
  <tr><td>FIFO</td><td>First In, First Out</td>
      <td>A queue where data is read in the same order it was written. The
          NPU uses hardware FIFOs (&ldquo;ObjectFIFOs&rdquo;) to stream data
          between tiles, like Unix pipes between processes.</td></tr>
  <tr><td>GEMM</td><td>General Matrix Multiply</td>
      <td>The standard linear algebra operation C = A &times; B + C. Neural
          network layers are essentially sequences of GEMMs with
          non-linearities in between.</td></tr>

  <tr><td colspan="3" style="background:#e8e8e8;font-weight:600;">
    Numeric formats</td></tr>
  <tr><td>bf16 / bfloat16</td><td>Brain Floating Point, 16-bit</td>
      <td>A 16-bit floating-point format with the same exponent range as
          float32 (8 bits) but less precision (7-bit mantissa vs 23-bit).
          Invented at Google Brain for ML training where range matters more
          than precision.</td></tr>
  <tr><td>BFP16</td><td>Block Floating Point, 16-bit</td>
      <td>An emulation mode on AIE2P that groups bf16 values into blocks
          sharing a common exponent. Enables efficient SIMD matmul in the
          MMUL unit with tile factor r = s = t = 8.</td></tr>
  <tr><td>INT8</td><td>8-bit Integer</td>
      <td>8-bit integer arithmetic, used for quantized inference. The
          NPU&rsquo;s peak in INT8 mode is 50 TOPS (double the bf16
          rate).</td></tr>

  <tr><td colspan="3" style="background:#e8e8e8;font-weight:600;">
    Performance metrics</td></tr>
  <tr><td>FLOPS</td><td>Floating-point Operations Per Second</td>
      <td>The standard measure of compute throughput. One multiply-add on
          two numbers counts as 2 FLOPs.</td></tr>
  <tr><td>GFLOPS</td><td>Giga&nbsp;FLOPS (10<sup>9</sup>)</td>
      <td>Billions of floating-point operations per second.</td></tr>
  <tr><td>TFLOPS</td><td>Tera&nbsp;FLOPS (10<sup>12</sup>)</td>
      <td>Trillions of floating-point operations per second. The NPU&rsquo;s
          peak is 25 TFLOPS in bfloat16.</td></tr>
  <tr><td>TOPS</td><td>Tera Operations Per Second</td>
      <td>Like TFLOPS but for integer operations. Used for INT8 peak specs
          (50 TOPS for this NPU).</td></tr>

  <tr><td colspan="3" style="background:#e8e8e8;font-weight:600;">
    Machine learning</td></tr>
  <tr><td>MLP</td><td>Multi-Layer Perceptron</td>
      <td>A neural network made of fully-connected (dense) layers. Each
          layer computes y = activation(x &times; W + b).</td></tr>
  <tr><td>ReLU</td><td>Rectified Linear Unit</td>
      <td>The activation function max(x, 0). Simple, cheap to compute, and
          widely used.</td></tr>

  <tr><td colspan="3" style="background:#e8e8e8;font-weight:600;">
    Toolchain &amp; software</td></tr>
  <tr><td>IRON</td><td>(name, not an acronym)</td>
      <td>AMD&rsquo;s Python API for programming AIE tiles at a high level:
          defining kernels, ObjectFIFOs, and tile placements.</td></tr>
  <tr><td>MLIR</td><td>Multi-Level Intermediate Representation</td>
      <td>A compiler framework (from the LLVM project) that represents
          programs at multiple abstraction levels. MLIR-AIE is the dialect
          that targets AIE hardware.</td></tr>
  <tr><td>XRT</td><td>Xilinx Runtime</td>
      <td>The userspace library and kernel driver that loads bitstreams onto
          the NPU and manages execution.</td></tr>
  <tr><td>XCLBIN</td><td>Xilinx Container for Linux Binary</td>
      <td>The compiled binary file that contains the NPU bitstream (tile
          configuration, routing, kernel code).</td></tr>
  <tr><td>LLVM</td><td>Low Level Virtual Machine</td>
      <td>A widely-used compiler infrastructure. Peano/LLVM-AIE is a fork
          that compiles C++ to AIE tile machine code.</td></tr>
  <tr><td>BD</td><td>Buffer Descriptor</td>
      <td>A hardware structure in the DMA engine that describes one data
          transfer: source address, size, stride pattern. The 10-bit size
          field (max 1024) is a constraint discussed in Section 4.3.</td></tr>
</table>

<h3>2.2 How CPUs execute neural networks (and why it&rsquo;s slow)</h3>

<p>
When you call <code>y = torch.relu(x @ W)</code> on a CPU, here is what
actually happens at the hardware level:
</p>

<ol>
  <li>The weight matrix <code>W</code> lives in DRAM. The CPU issues a load
      instruction, and the data travels through the <strong>memory
      hierarchy</strong>: DRAM &rarr; L3 cache &rarr; L2 cache &rarr; L1 cache
      &rarr; registers.</li>
  <li>The CPU&rsquo;s SIMD units multiply a few rows at a time. The result
      goes back through the cache hierarchy to DRAM.</li>
  <li>For the next layer, the process repeats: read the activation from DRAM,
      read the next weight matrix from DRAM, compute, write back.</li>
</ol>

<p>
The fundamental problem is the <strong>memory wall</strong>: modern compute
units can perform arithmetic far faster than memory can supply data. A typical
CPU can do ~1 TFLOPS of bf16 math, but its DRAM bandwidth is ~50 GB/s. To
keep the ALUs busy doing a 128&times;128 matmul, you need 128&times;128&times;2
= 32 KB of weight data delivered every ~1 &mu;s &mdash; which is 32 GB/s just
for one matrix. If you&rsquo;re running 4 layers, that&rsquo;s 128 GB/s,
already exceeding the memory bandwidth. The CPU stalls waiting for data.
</p>

<div class="highlight">
<strong>The memory wall in one sentence:</strong> Arithmetic is cheap; moving
data is expensive. The deeper you go in a neural network, the more time is
spent shuffling data between memory levels, not doing useful math.
</div>

<h3>2.3 The spatial dataflow alternative</h3>

<p>
A <strong>spatial architecture</strong> takes a radically different approach.
Instead of one fast processor with a deep cache hierarchy, it uses <em>many
small processors</em> (tiles), each with a tiny but <em>extremely fast</em>
local memory (SRAM). Data moves between tiles through dedicated hardware
channels (FIFOs), not through shared caches.
</p>

<p>
Think of it like a factory assembly line versus a single master craftsman:
</p>

<ul>
  <li><strong>CPU (craftsman):</strong> One highly skilled worker goes to the
      warehouse (DRAM) for each part, brings it to the workbench (registers),
      processes it, takes the result back to the warehouse, gets the next part.
      Most time is spent walking, not working.</li>
  <li><strong>NPU (assembly line):</strong> Many workers sit at their
      stations with all their parts already on their desk (SRAM). A conveyor
      belt (FIFO) moves the workpiece from one station to the next. Nobody
      walks anywhere.</li>
</ul>

<p>
The key property is <strong>data locality</strong>: once data is loaded into a
tile&rsquo;s 64 KB SRAM, it stays there for as many operations as you can do
on it. There is no cache to get evicted from, no bus to contend for. If a
128&times;128 weight matrix (32 KB) fits in SRAM, you can multiply against it
thousands of times at full speed &mdash; which is exactly what our recurrent
MLP does.
</p>

<h3>2.4 Key hardware concepts used in this project</h3>

<p>
These concepts appear throughout the paper. You don&rsquo;t need to understand
every transistor, but knowing these ideas will make the design choices clear:
</p>

<h4>Double buffering (ping-pong)</h4>
<p>
If a tile needs to <em>receive</em> new data while <em>computing</em> on data
it already has, you need two buffers: one being filled by DMA while the other
is being read by the compute unit. They swap roles each iteration. We use this
for the activation buffers: buffer A holds the input, the tile computes into
buffer B, then B becomes the input and A becomes the output.
</p>

<h4>Tiled matrix layout</h4>
<p>
A &ldquo;row-major&rdquo; matrix stores elements left-to-right, top-to-bottom:
<code>[[a, b], [c, d]]</code> becomes <code>[a, b, c, d]</code>. The AIE
matmul unit instead expects a <strong>blocked (tiled) layout</strong>: the
matrix is divided into 8&times;8 sub-matrices, and each block is stored
contiguously. This matches the MMUL hardware which multiplies 8&times;8
blocks in one operation. The <code>to_tiled()</code> function handles this
conversion.
</p>

<h4>ObjectFIFOs and data movement</h4>
<p>
On a CPU, data movement is implicit &mdash; you <code>load</code> from an
address and the cache hierarchy handles the rest. On the NPU, you must
<em>explicitly</em> program every data transfer: &ldquo;move 4 KB from DDR
address X to tile (3, 2)&rsquo;s input buffer.&rdquo; IRON&rsquo;s
<code>ObjectFifo</code> abstraction makes this manageable: you declare a
typed channel between a producer and a consumer, and the compiler generates
the DMA configurations.
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

<h3>2.5 How to read the rest of this paper</h3>

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
The AMD XDNA&nbsp;2 NPU in the Ryzen AI 9 HX 370 (codename Strix Point) is a
tiled spatial-dataflow processor with the following structure:
</p>

<div class="figure">
  <img src="xdna2_hardware.png" alt="XDNA 2 tile array">
  <div class="caption">
    Figure 1: Physical tile array of the AMD XDNA&nbsp;2 NPU. 32 compute tiles
    (rows 2&ndash;5) each contain ~64 KB SRAM and a bf16 MMUL unit. 8 memory tiles
    (row 1, 512 KB each) serve as on-chip L2 buffers and routing hubs.
    8 shim tiles (row 0) provide DMA access to host DDR memory.
  </div>
</div>

<table>
  <tr><th>Property</th><th>Value</th></tr>
  <tr><td>Compute tiles</td><td>32 (8 columns &times; 4 rows)</td></tr>
  <tr><td>Memory tiles</td><td>8 (512 KB each, 4 MB total)</td></tr>
  <tr><td>Per-tile SRAM</td><td>~64 KB data memory</td></tr>
  <tr><td>Per-tile compute</td><td>bf16 MMUL unit, VLIW+SIMD core</td></tr>
  <tr><td>Clock frequency</td><td>~1.5 GHz</td></tr>
  <tr><td>Peak throughput</td><td><strong>25 TFLOPS</strong> (bfloat16)</td></tr>
  <tr><td>Interconnect</td><td>ObjectFIFOs (tile-to-tile double-buffered streams)</td></tr>
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
<strong>Lesson from Phase 1:</strong> A single large GEMM achieves only 2.49 TFLOPS
on the NPU (10% of peak) because it is <em>memory-bandwidth limited</em> &mdash;
data must stream from DDR. The NPU wins when data <strong>stays on-chip</strong>.
</div>

<h2>4. The Architecture: Recurrent MLP</h2>

<p>
We choose a recurrent MLP: a single weight matrix <code>W</code> (128&times;128,
bfloat16) is loaded once into each tile's SRAM and applied repeatedly in a tight
hardware loop:
</p>

<pre>
x = input                     # loaded from DDR, 16&times;128 bf16
for i in range(num_iters):
    y = ReLU(x @ W)           # matmul + activation
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

<h3>4.1 Why this architecture</h3>

<ul>
  <li><strong>Maximizes on-chip time:</strong> Weight is loaded once (32 KB) and
      reused for thousands of matmul operations. DDR I/O happens only at
      start and end.</li>
  <li><strong>Amortizes overhead:</strong> Each NPU invocation has ~120 &mu;s of
      driver/DMA overhead. With depth=2000, compute time (~1.3 ms) dominates
      overhead by 10&times;.</li>
  <li><strong>Fits SRAM budget:</strong> W (32 KB) + 2 activation buffers (4 KB
      each) + stack (1 KB) = 41 KB, well within the 64 KB tile limit.</li>
  <li><strong>Linear tile scaling:</strong> Each tile is independent &mdash;
      doubling tiles doubles throughput with no communication overhead.</li>
</ul>

<h3>4.2 Multi-row data routing</h3>

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

<h3>4.3 Critical implementation constraints</h3>

<p>
Several non-obvious hardware constraints shaped the design:
</p>

<ul>
  <li><strong>No FIFO ops inside loops:</strong> Placing <code>acquire()</code> /
      <code>release()</code> inside <code>range_()</code> (which compiles to
      <code>scf.for</code>) causes DMA deadlock. All FIFO operations must happen
      <em>outside</em> the loop.</li>
  <li><strong>DMA BD 10-bit size limit:</strong> Shim DMA buffer-descriptor sizes
      are 10-bit (max 1024). For B=16, H=128, the product B&times;H=2048 exceeds
      this, so tensor access patterns must factor dimensions as [B, H] = [16, 128]
      instead of [B&times;H] = [2048].</li>
  <li><strong>Accumulating matmul:</strong> IRON's <code>mm.cc</code> kernel
      computes C += A&times;B (not C = A&times;B). We must explicitly zero the
      output buffer before each matmul, which wastes ~12% of cycle time.</li>
</ul>

<h2>5. Results</h2>

<div class="figure">
  <img src="performance.png" alt="Performance scaling">
  <div class="caption">
    Figure 3: NPU throughput scaling (left) and speedup over CPU (right).
    Throughput scales near-linearly with tile count at ~360 GFLOPS/tile.
    At 24 tiles, we achieve 8.95 TFLOPS and 20&times; CPU speedup.
  </div>
</div>

<table>
  <tr><th>Tiles</th><th>Depth</th><th>NPU Latency</th><th>NPU TFLOPS</th>
      <th>CPU GFLOPS</th><th>Speedup</th></tr>
  <tr><td>8 (1 row)</td><td>1,000</td><td>1.45 ms</td><td><strong>2.89</strong></td>
      <td>237</td><td><strong>12.2&times;</strong></td></tr>
  <tr><td>16 (2 rows)</td><td>1,000</td><td>1.46 ms</td><td><strong>5.74</strong></td>
      <td>354</td><td><strong>16.2&times;</strong></td></tr>
  <tr><td>24 (3 rows)</td><td>1,000</td><td>1.46 ms</td><td><strong>8.63</strong></td>
      <td>429</td><td><strong>20.1&times;</strong></td></tr>
  <tr><td>24 (3 rows)</td><td>10,000</td><td>14.05 ms</td><td><strong>8.95</strong></td>
      <td>439</td><td><strong>20.4&times;</strong></td></tr>
</table>

<h3>5.1 Scaling analysis</h3>

<p>
Per-tile throughput is remarkably consistent at ~360 GFLOPS regardless of tile
count, confirming that the tiles operate independently with no contention:
</p>

<pre>
 8 tiles &times; 360 GFLOPS/tile =  2.9 TFLOPS  &check;
16 tiles &times; 360 GFLOPS/tile =  5.7 TFLOPS  &check;  (near-linear)
24 tiles &times; 360 GFLOPS/tile =  8.6 TFLOPS  &check;  (near-linear)
</pre>

<h3>5.2 Gap to theoretical peak</h3>

<p>
We achieve 8.95 of 25 TFLOPS (35.8%). The remaining gap is well-understood:
</p>

<table>
  <tr><th>Factor</th><th>Impact</th><th>Potential fix</th></tr>
  <tr><td>Per-tile utilization</td><td>360/768 = 47%</td>
      <td>Fused C=A&times;B kernel</td></tr>
  <tr><td>zero_bf16 overhead</td><td>~12% of step time</td>
      <td>Fused kernel eliminates this</td></tr>
  <tr><td>Array utilization</td><td>24/32 = 75%</td>
      <td>4-row routing (needs HW/compiler support)</td></tr>
  <tr><td>Combined theoretical max</td><td>~18 TFLOPS</td>
      <td>With fused kernel + 24 tiles</td></tr>
</table>

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

mm.cc          &xrarr;  mlp_mm.o    &xrarr;  mlp_kernels.a
mlp_kernels.cc &xrarr;  mlp_relu.o  &nearr;
</pre>

<h2>7. Code Structure</h2>

<p>
The project is intentionally minimal &mdash; four Python files and one C++ file:
</p>

<table>
  <tr><th>File</th><th>Lines</th><th>Purpose</th></tr>
  <tr><td><code>spatial_mlp/__init__.py</code></td><td>~55</td>
      <td>Tiling utilities (<code>to_tiled</code>, <code>from_tiled</code>)</td></tr>
  <tr><td><code>spatial_mlp/design.py</code></td><td>~300</td>
      <td>IRON design: tile topology, FIFOs, workers, DMA</td></tr>
  <tr><td><code>spatial_mlp/op.py</code></td><td>~140</td>
      <td>IRON operator: compilation artifacts, runtime buffers</td></tr>
  <tr><td><code>spatial_mlp/test.py</code></td><td>~240</td>
      <td>Benchmark: NPU vs CPU execution and reporting</td></tr>
  <tr><td><code>aie_kernels/mlp_kernels.cc</code></td><td>~55</td>
      <td>Custom AIE2P kernels: ReLU, copy (bf16, SIMD)</td></tr>
</table>

<p>
Each module has a single, well-defined responsibility. The design module is
decomposed into small functions that each handle one aspect of the hardware
mapping: validation, kernel definition, FIFO topology, worker bodies, tensor
access patterns, and DMA sequences.
</p>

<h2>8. Future Work</h2>

<ul>
  <li><strong>Fused matmul kernel:</strong> A C=A&times;B kernel (instead of
      C+=A&times;B + separate zero) would eliminate ~12% overhead and push
      per-tile throughput from 360 to ~700 GFLOPS.</li>
  <li><strong>INT8 mode:</strong> The NPU's peak is 50 TOPS for int8.
      With H=256 (fitting larger weights), this could double throughput.</li>
  <li><strong>Training:</strong> Research NPU backpropagation
      (see <a href="https://arxiv.org/html/2504.03083v1">arXiv:2504.03083</a>).</li>
  <li><strong>Real ML task:</strong> Apply the architecture to a concrete
      sequence modeling or time-series task where deep recurrence is natural.</li>
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
