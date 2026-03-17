# TODO

The goal of this file is to keep the paper backlog focused on a simple rule:
every important claim should trace cleanly to either a citation, a figure, a
table, or a reproducible experiment.

## Measurements and baselines

- [ ] Quantify the current CPU embed overhead
  - Measure how much of the current wall-clock time is spent in the host-side
    `28 -> 32` embed.
  - Estimate how much headroom could be recovered by moving that step on-array.
  - Use that number to sharpen the motivation for the next systems step.

- [ ] Expand the latency analysis
  - Measure single-sample latency for the conveyor-belt NPU path.
  - Measure tail latency, for example p99, instead of reporting throughput
    alone.
  - Add either a latency distribution plot or a compact latency table to the
    paper.

- [ ] Document benchmark methodology and variability
  - State warm-up policy, sample count, repetition count, and whether weights
    are already resident on the device.
  - Define precisely what is included in wall time versus kernel time.
  - Repeat the key benchmarks enough to report a compact variability summary
    rather than a single point estimate.

- [ ] Add a standard HEP baseline comparison
  - Compare accuracy and throughput against a strong tabular baseline such as
    XGBoost or LightGBM on CPU.
  - Optionally include a small MLP baseline on a mobile GPU.
  - If the numbers support it, frame the comparison in terms of
    efficiency-per-dollar or efficiency-per-watt.

- [ ] Discuss scaling to wider feature sets
  - Explain how the conveyor-belt design changes as the input becomes wider.
  - State more explicitly when the 64 KB per-tile SRAM becomes the hard
    bottleneck.
  - Briefly connect this to more realistic jet-tagging or richer HEP feature
    pipelines.

- [ ] Add quantitative SRAM-budget accounting
  - Break down per-tile memory use for weights, activations, and buffers.
  - Show why the retained hidden widths fit and when they stop fitting.
  - Reuse the same accounting to support the wider-input discussion.

## Framing, figures, and literature

- [ ] Add a claim-by-claim evidence audit
  - Enumerate each quantitative, comparative, or systems claim in the paper.
  - Tag each claim with its backing source: citation, table, figure, or local
    experiment artifact.
  - Rephrase or remove any sentence that cannot be defended cleanly.

- [ ] Strengthen related-work positioning
  - Add a compact comparison against classic HIGGS baselines, HEP trigger or
    low-latency inference work, and relevant accelerator or spatial-computing
    literature.
  - Make the novelty statement explicit and well-bounded.
  - Clarify what is new here versus what is inherited from the benchmark or the
    toolchain.

- [ ] Plot the accuracy-throughput frontier
  - Put the retained operating points on one figure so the trade-off is visible
    at a glance.
  - If the baseline comparison is strong, place those points on the same plot.
  - Use the figure to support the ``best throughput point'' versus ``best
    accuracy point'' narrative.

- [ ] Add an in-paper model-to-hardware figure
  - Show the CPU-side embed, the NPU conveyor-belt residual body, and the final
    head placement options.
  - Make the tile-to-block correspondence visible.
  - Use the figure to reduce the amount of explanation needed in the text.

- [ ] Tighten claim boundaries and limitations
  - Add a short explicit limitations or scope subsection to the paper.
  - State clearly what is not being claimed: production trigger readiness,
    state-of-the-art HIGGS accuracy, fully on-array execution, or
    energy-efficiency leadership.
  - Tie the trigger-motivation discussion to cited rates and measured numbers
    without implying deployment equivalence.

## Suggested order

1. Quantify the CPU embed overhead.
2. Expand the latency analysis.
3. Document benchmark methodology and variability.
4. Add the HEP baseline comparison.
5. Add the wider-input discussion together with SRAM-budget accounting.
6. Plot the accuracy-throughput frontier.
7. Add the in-paper model-to-hardware figure.
8. Strengthen related-work positioning.
9. Run the claim-by-claim evidence audit.
10. Finish with the limitations / claim-boundary pass.
