#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate the experimental supplement PDF.

Contains the block-recurrent character language model architecture,
results, and code structure from earlier phases of the project.
This content was moved here when the main whitepaper was refocused
on the resmlp (32-layer residual MLP) work.

Usage::

    python docs/generate_experimental.py

Requires: weasyprint (in the project .venv).
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
  a.gref {
    color: #2c3e50;
    text-decoration: none;
    border-bottom: 1px dotted #4A90D9;
  }
</style>
</head>
<body>

<h1>Experimental: Block-Recurrent Character LM on NPU</h1>
<p class="subtitle">
  Supplementary material &mdash; earlier architecture experiments
</p>

<p>
This document describes the <strong>block-recurrent character language
model</strong> that was developed in earlier phases of the project. The model
trains on Shakespeare text and runs inference on the AMD XDNA&nbsp;2 NPU using
a 4-stage pipeline of 8 columns (32 tiles total). Each pipeline stage fuses
RMSNorm + matrix multiply + ReLU into a single on-chip kernel.
</p>

<p>
This architecture demonstrated correct hardware mapping and achieved 89,600
characters/second throughput on 384 parallel sequences. However, it operates
at only 0.4% of peak NPU utilisation due to CPU&ndash;NPU round-trip overhead
(8 NPU calls per character). The main branch has since moved to a simpler
32-layer residual MLP (<code>resmlp/</code>) that processes all 32 layers in a
single NPU call via a serpentine tile path.
</p>

<p>
For hardware background (XDNA&nbsp;2 tile architecture, IRON programming model,
design patterns), see the main whitepaper on the <code>main</code> branch.
</p>

<h2>5. The Architecture: Block-Recurrent Character LM</h2>

<p>
We build a character-level language model whose architecture is dictated by the
NPU&rsquo;s physical tile array. The model has 32 layers (one per compute tile),
grouped into 8 blocks of 4 layers (one block per column). Each block maps to
a single NPU pipeline call. Between blocks, the CPU applies the operations
that the pipeline cannot: normalisation, embedding injection, and residual
connections.
</p>

<h3>5.1 The block-recurrent computation</h3>

<p>
For each character in the input sequence, the model updates a hidden state
<code>h</code> (a 128-dimensional <a href="#g-bf16" class="gref">bfloat16</a>
vector) through 8 blocks:
</p>

<pre>
for each character:
    for each block g = 0..7:
        CPU:  h_b = h + embed(char) + bias_g              &larr; inject input
        NPU:  for stage j = 0..3:
                  h_b = ReLU(RMSNorm(h_b) @ W[4g+j])      &larr; fused on-chip
        CPU:  h = h + h_b                                  &larr; residual connection
    CPU:  h = RMSNorm(h)                                   &larr; post-norm
    CPU:  logits = h @ W_out + b_out                       &larr; predict next char
</pre>

<p>
Each of the 8 NPU calls runs a 4-stage pipeline where each stage fuses
RMSNorm&nbsp;+&nbsp;matmul&nbsp;+&nbsp;ReLU:
</p>

<ol>
  <li>The CPU prepares the block input: add the character embedding and a
      per-block learned bias to the hidden state (two vector additions).</li>
  <li>The prepared vector is sent to the NPU, where it passes through 4 tiles
      in sequence. Each tile normalises its input (RMSNorm), multiplies by its
      own weight matrix W<sub>i</sub> (128&times;128), and applies
      <a href="#g-relu" class="gref">ReLU</a> &mdash; all in a single fused
      <code>norm_matmul_relu</code>
      <a href="#v-kernel" class="gref">kernel</a>.</li>
  <li>The NPU returns the result to the CPU, which adds it back to the hidden
      state (residual connection).</li>
</ol>

<p>
After all 8 blocks, a final RMSNorm and a linear readout produce a probability
distribution over the ~65-character vocabulary.
</p>

<h3>5.2 NPU tile mapping</h3>

<p>
The model uses all 32 compute tiles in an 8-column &times; 4-row layout:
</p>

<pre>
Column 0        Column 1        ...  Column 7
(samples 0-47)  (samples 48-95)      (samples 336-383)

  Row 2: W&sub0;       Row 2: W&sub0;    ...    Row 2: W&sub0;      Stage 1
  &darr;              &darr;                  &darr;
  Row 3: W&sub1;       Row 3: W&sub1;    ...    Row 3: W&sub1;      Stage 2
  &darr;              &darr;                  &darr;
  Row 4: W&sub2;       Row 4: W&sub2;    ...    Row 4: W&sub2;      Stage 3
  &darr;              &darr;                  &darr;
  Row 5: W&sub3;       Row 5: W&sub3;    ...    Row 5: W&sub3;      Stage 4
</pre>

<ul>
  <li><strong>8 columns</strong> process 8 independent batch slices in parallel,
      48 samples each = 384 total.</li>
  <li><strong>4 rows</strong> per column form a pipeline: data flows from row 2
      to row 5 through <a href="#g-fifo" class="gref">ObjectFIFOs</a>,
      tile-to-tile, with no <a href="#g-ddr" class="gref">DDR</a> traffic
      between stages.</li>
  <li>All 8 columns share the same 4 weight matrices for a given block. The
      CPU loads the appropriate block&rsquo;s weights before each NPU call.</li>
</ul>

<h3>5.3 Why this architecture</h3>

<p>
Every architectural choice follows from a hardware constraint:
</p>

<ul>
  <li><strong>H=128:</strong> A 128&times;128 weight matrix plus a 128-element
      RMSNorm scale vector occupies 32.25 KB in
      <a href="#g-bf16" class="gref">bf16</a>. With two activation buffers
      (48&times;128 = 12 KB each, <a href="#v-doublebuf" class="gref">double-buffered</a>)
      and ~1.5 KB of stack+code, the total is ~58 KB &mdash; within the 64 KB per-tile
      <a href="#g-sram" class="gref">SRAM</a> limit.</li>
  <li><strong>4 layers per block:</strong> The NPU has 4 compute rows.
      A 4-stage pipeline uses one row per stage, fully utilizing the vertical
      dimension.</li>
  <li><strong>8 blocks:</strong> With 4 layers per block and 32 total layers,
      8&nbsp;blocks = 32&nbsp;layers = 32&nbsp;tiles. This is the maximum depth
      that maps cleanly to the hardware.</li>
  <li><strong>B=48:</strong> The fused <code>matmul_relu</code> kernel
      pipelines efficiently only when the outer loop has &ge;3 iterations:
      M/(2r) &ge; 3, so B &ge; 48 for r&nbsp;=&nbsp;8.
      (See <code>logbook.md</code> for the kernel fusion analysis.)</li>
  <li><strong>384 parallel sequences:</strong> 8 columns &times; 48 samples
      per column. Processing many sequences in parallel amortises the ~120 &mu;s
      <a href="#g-xrt" class="gref">XRT</a> driver overhead per NPU call.</li>
</ul>

<div class="highlight">
<strong>Why not B=1 with a larger hidden dimension?</strong>
<p>
A natural question: could each column process a <em>single</em> sequence
(B=1) and use the freed SRAM for a larger weight matrix?
</p>
<p>
B=1 is <em>not</em> a hardware impossibility. The MMUL instruction needs
8&times;8 input tiles, so B=1 would require padding to B=8 (7 zero rows).
This is valid &mdash; IRON&rsquo;s GEMM operator does exactly this for
arbitrary dimensions. But it is highly wasteful:
</p>
<ol>
  <li><strong>7/8 wasted compute:</strong> With B=1 padded to 8, the MMUL
      processes 8 rows but only 1 produces useful output. One matmul yields
      32K useful FLOPs instead of 1.57M with B=48 &mdash; a
      <strong>48&times; efficiency loss</strong>.</li>
  <li><strong>No loop pipelining:</strong> The outer loop has only 1 iteration
      (B/8=1), so the chess compiler cannot overlap load/compute/store phases.
      Peak throughput drops from 23.93 to &lt;4 TFLOPS.</li>
  <li><strong>SRAM still limits H:</strong> Even with B=1, the weight matrix
      alone is H&times;H&times;2 bytes. H=192 needs 72 KB &gt; 64 KB.
      The maximum reachable is H&asymp;176 &mdash; only 37% more parameters per
      layer, while losing 48&times; the batch throughput.</li>
</ol>
<p>
At B=1, a CPU is equally fast or faster &mdash; the NPU&rsquo;s advantage
comes precisely from processing many rows in parallel through its systolic
array.
</p>
<p>
An alternative approach would be to <em>shard</em> a large weight matrix
across multiple tiles (e.g., H=512 split into four 512&times;128 slices).
This requires horizontal communication between tiles for partial-sum
reduction, which the current column topology does not support. Exploring
such <em>model-parallel</em> layouts is future work.
</p>
</div>

<h3>5.4 Fusing normalisation into the pipeline</h3>

<p>
The key innovation is a custom NPU kernel that fuses three operations &mdash;
<a href="#g-rmsnorm" class="gref">RMSNorm</a>, matrix multiply, and
<a href="#g-relu" class="gref">ReLU</a> &mdash; into a single per-tile function.
Without this fusion, normalisation would require CPU intervention between every
pipeline stage, defeating the purpose of the 4-stage streaming pipeline.
</p>

<p>
The fused <code>norm_matmul_relu</code> kernel for each pipeline stage:
</p>

<ol>
  <li><strong>RMSNorm in-place</strong> (scalar float32 for stability): compute
      the root-mean-square of the input row, divide each element by it, and
      multiply by the learned scale vector. This is a reduction over 128 elements
      per row, using 8 Babylonian iterations for the inverse square root to avoid
      library dependencies.</li>
  <li><strong>Fused matmul + ReLU</strong> (vectorised bf16): the normalised
      input feeds directly into the same 2&times;2 tile-expanded matrix multiply
      used previously, with ReLU applied during the store phase.</li>
</ol>

<p>
The scale vector (128&nbsp;&times;&nbsp;2 bytes = 256 bytes) is packed at the end
of each tile&rsquo;s weight buffer: [W (32 KB), scale (256 B)].  This adds only
0.25 KB per tile, well within the SRAM budget.
</p>

<p>
Moving per-layer normalisation into the NPU pipeline closed the quality gap
from val&nbsp;loss 2.42 (pure matmul+ReLU blocks) to 2.03, approaching the
per-layer CPU-only model&rsquo;s 1.94 &mdash; while keeping the full 32-layer
inference on the NPU.
</p>

<div class="key-insight">
<strong>Design principle:</strong> We train the model with exactly the same
block structure that runs on the NPU. There is no &ldquo;compilation&rdquo; step
that approximates a richer model for hardware. What you train is what you deploy.
</div>

<h3>5.5 Making it trainable</h3>

<p>
Simply stacking 32 layers of ReLU(h&nbsp;&times;&nbsp;W&nbsp;+&nbsp;b) does not
work: the hidden state either explodes to infinity or collapses to zero. We use
three techniques (explained in
<a href="#s-nn-fundamentals">Section 2.7</a>):
</p>

<ul>
  <li><strong>Per-stage RMSNorm:</strong> Each pipeline stage normalises its
      input to unit RMS <em>before</em> the matrix multiply. Because the norm
      is fused into the NPU kernel, this happens on-chip with no CPU
      intervention.  This prevents the activations from growing exponentially
      through the 4-layer chain and across the 64-character sequence
      during <a href="#s-bptt">backpropagation through time</a>.</li>
  <li><strong>Residual connections:</strong> <code>h = h + block(h)</code>
      after each block. Gradients flow backwards through the identity shortcut,
      keeping training stable even at 32 layers.</li>
  <li><strong>Input injection:</strong> The character embedding is added at
      the start of every block, not just the first. This ensures each block
      can directly access the current input, rather than relying on information
      that has been progressively transformed by preceding blocks.</li>
  <li><strong>Spectral norm clamping:</strong> During training, each weight
      matrix is periodically rescaled so its largest singular value &le; 1.
      This prevents any single layer from amplifying the hidden state, keeping
      the recurrent dynamics stable.</li>
</ul>

<h2>6. Results</h2>

<h3>6.1 Language model quality</h3>

<p>
The block-recurrent model was trained on the tiny Shakespeare dataset (1.1 MB,
~1.1M characters) for 10 epochs on a GPU, then deployed to the NPU:
</p>

<table>
  <tr><th>Model</th><th>Parameters</th><th>Val Loss</th>
      <th>Perplexity</th><th>Device</th></tr>
  <tr style="background:#d4edda;font-weight:600;">
      <td>Block-recurrent (fused norm+mm+relu)</td>
      <td>542K</td><td>2.03</td><td>7.6</td><td>NPU (32 tiles)</td></tr>
  <tr><td>Per-layer recurrent (norm every layer)</td>
      <td>542K</td><td>1.94</td><td>6.9</td><td>CPU/GPU only</td></tr>
  <tr><td>Transformer baseline (GPT-style)</td>
      <td>818K</td><td>1.89</td><td>6.6</td><td>GPU</td></tr>
</table>

<p>
The fused norm+matmul+ReLU kernel closes the quality gap: the NPU model
(perplexity 7.6) approaches the per-layer CPU model (6.9) and the transformer
baseline (6.6), while running entirely on the 32-tile NPU pipeline.
The model produces recognisable Shakespearean English with a 542K-parameter
model trained on 1 MB of text.
</p>

<h3>6.2 Inference performance</h3>

<h4>Throughput comparison (single-sequence autoregressive generation)</h4>
<table>
  <tr><th>Model</th><th>Device</th><th>chars/s</th><th>ms/char</th></tr>
  <tr><td>Recurrent 64L (1M params)</td><td>CPU</td>
      <td><strong>1,065</strong></td><td>0.94</td></tr>
  <tr><td>Transformer 4L (818K params)</td><td>CPU</td>
      <td>775</td><td>1.29</td></tr>
  <tr><td>NPU block-recurrent 32L (542K params)</td><td>NPU &times;1 seq</td>
      <td>233</td><td>4.29</td></tr>
  <tr><td>Transformer 4L (818K params)</td><td>GPU</td>
      <td>187</td><td>5.35</td></tr>
  <tr><td>Recurrent 64L (1M params)</td><td>GPU</td>
      <td>92</td><td>10.90</td></tr>
</table>

<p>
For a single sequence, the CPU is fastest. The models are too small
(&lt;&nbsp;1M parameters) to saturate the GPU&rsquo;s compute units, so GPU
kernel launch overhead dominates. The NPU likewise loses on single-sequence
latency due to the ~120&nbsp;&mu;s XRT overhead per call.
</p>

<h4>NPU batched throughput</h4>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>NPU tiles used</td><td>32 (8 columns &times; 4 rows)</td></tr>
  <tr><td>NPU calls per character</td><td>8 (one per block)</td></tr>
  <tr><td>Latency per NPU call</td><td>~0.14 ms</td></tr>
  <tr><td>Total NPU time per character</td><td>~1.1 ms</td></tr>
  <tr><td>Total time per character (NPU + CPU)</td><td>~4.2 ms</td></tr>
  <tr><td>Throughput (1 sequence)</td><td>233 chars/s</td></tr>
  <tr><td>Throughput (384 parallel sequences)</td><td><strong>89,600 chars/s</strong></td></tr>
</table>

<p>
The 384-sequence throughput is the key figure: by processing 8&nbsp;columns
&times; 48&nbsp;samples per column in parallel, we amortise the per-call
overhead across hundreds of sequences. Compared to the CPU baseline
(1,065&nbsp;chars/s for a single sequence), the NPU achieves
<strong>84&times; higher total throughput</strong> when all 384 sequences
are served simultaneously. The CPU overhead between NPU calls
(embedding injection, bias addition, residual connection) is minimal since
normalisation is now handled on-chip by the fused kernel.
</p>

<h3>6.3 What limits throughput</h3>

<p>
Each NPU call performs only 4 matmuls of size 48&times;128&times;128 &mdash;
a tiny amount of arithmetic completed in microseconds. The ~0.14 ms measured
per call is dominated by <a href="#g-xrt" class="gref">XRT</a> driver
overhead (instruction dispatch, <a href="#g-dma" class="gref">DMA</a> setup),
not compute. With 8 calls per character, that overhead adds up.
</p>

<h4>Utilisation arithmetic</h4>
<p>
Each character step (one character for all 384 sequences) requires:
</p>
<ul>
  <li>8 NPU calls &times; 32 tiles &times; one matmul [48, 128, 128]</li>
  <li>FLOPs per matmul: 2 &times; 48 &times; 128 &times; 128 = 1,572,864</li>
  <li>Total per step: 8 &times; 32 &times; 1.57M = <strong>402.7M FLOPs</strong></li>
</ul>
<p>
At 233 steps/s this is <strong>93.8 GFLOPS &mdash; only 0.38% of the 25 TFLOPS
bf16 peak</strong>.
</p>

<table>
  <tr><th>Budget item</th><th>Time</th><th>% of step</th></tr>
  <tr><td>Pure MMUL compute (402.7M FLOPs &divide; 23.93 TFLOPS)</td>
      <td>~17 &mu;s</td><td>0.4%</td></tr>
  <tr><td>DMA + instruction dispatch (8 calls &times; ~0.14 ms)</td>
      <td>~1.1 ms</td><td>26%</td></tr>
  <tr><td>CPU overhead (embedding injection, residuals, softmax)</td>
      <td>~3.1 ms</td><td>74%</td></tr>
  <tr style="font-weight:600;"><td>Total per step</td>
      <td>~4.3 ms</td><td>100%</td></tr>
</table>

<p>
The <a href="#g-mmul" class="gref">MMUL</a> units are essentially idle.
The bottleneck is entirely in the 8 round-trips between CPU and NPU per
character.  This is fundamentally different from the throughput benchmark
(see <code>logbook.md</code>), where a single NPU call ran 1000 loop
iterations on-chip, making compute dominate overhead and achieving 95.7%
of the 25 <a href="#g-tflops" class="gref">TFLOPS</a> peak.
</p>

<div class="key-insight">
<strong>Insight:</strong> The current architecture proves the hardware mapping
is correct but achieves only 0.4% of peak compute utilisation.
The path to high utilisation is reducing the number of CPU&ndash;NPU round-trips
per character: either by running all 8 blocks on-chip in a single NPU call
(requires on-chip residual addition and embedding injection), or by switching
to INT8 mode (doubling the compute/overhead ratio), or both.
</div>
<h2>8. Code Structure</h2>

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
  <tr><td><code>aie_kernels/norm_matmul_relu.cc</code></td><td>~170</td>
      <td>Fused C&nbsp;=&nbsp;ReLU(RMSNorm(A,&nbsp;scale)&nbsp;&times;&nbsp;W)
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

<h2>9. Future Work</h2>

<ul>
  <li><strong><a href="#g-int8" class="gref">INT8</a> mode:</strong> The
      NPU&rsquo;s peak is 50 <a href="#g-tops" class="gref">TOPS</a> for
      int8 &mdash; double the bf16 rate. With quantization-aware training, this
      could push effective throughput to ~48 TOPS and allow H=256 (fitting in
      the same SRAM budget).</li>
  <li><strong>Vectorised RMSNorm:</strong> The current kernel uses scalar
      float32 for the normalisation reduction. A vectorised implementation
      using v16float SIMD could reduce the per-stage norm overhead from
      ~25&nbsp;&mu;s to ~3&nbsp;&mu;s, improving pipeline throughput
      when compute (not driver overhead) is the bottleneck.</li>
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
    """Render the experimental supplement HTML to PDF."""
    output_path = DOCS_DIR / "experimental.pdf"
    html = HTML(string=HTML_CONTENT, base_url=str(DOCS_DIR))
    html.write_pdf(output_path)
    print(f"  ✓ {output_path}")
    return output_path


if __name__ == "__main__":
    print("Generating experimental supplement...")
    generate()
    print("Done.")
