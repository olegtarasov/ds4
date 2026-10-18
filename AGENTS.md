# Maintained DwarfStar fork

Read `AGENT.md`, `CONTRIBUTING.md`, then `docs/FORK.md`. This repository holds
DwarfStar engine source only. The sibling `../local-llm-serving` is the broader
research hub and owns cross-engine harnesses, measurements and host operations.

## Owner constraints

- Target: M2 Max, 96 GB unified memory, Qwen3.8-Flash-Next Q4. Keep native
  262144 context, one session, prefill 2048, mmap, native Metal residency, f16
  K/V and MTP off for the retained profile unless running a deliberate experiment.
- The owner explicitly excludes SSD-heavy optimization leads. Do not enable
  routed-expert SSD streaming, disk KV caching, additional n-gram I/O prefetch,
  or download alternate weights. Existing BF16 n-gram row reads remain required
  by this model; do not preload/wire the 95 GiB table. This instruction overrides
  inherited AGENT.md requests to run SSD-streaming integration tests here.
- Do not alter GPU wired limits, sleep settings, existing installed llama.cpp /
  Pi configuration, or validated runtimes in the sibling hub's artifacts.
- Use authenticated loopback serving through the hub's guarded harness. Check
  model/router state before each run; never run two large models concurrently.
- The experimental raw-indexer ring supports one session only. Future prompt
  anchors/replay work must preserve the tail across wrap; no arbitrary rewind.

## Changes and delivery

Keep upstream ancestry and a conceptual divergence/provenance map in
`docs/FORK.md`. Import narrow changes with their exact source SHA/PR and explain
any port, omission or device restriction. Do not infer speed from commit titles.
Use appropriate model-free and guarded API correctness/performance tests. Record
unsupported CUDA/ROCm/TP checks as unexecuted; no external hardware is available.

Keep weights, keys, installed packages, generated binaries and raw traffic out
of Git. Put scratch output in ignored `artifacts/`. Commit intended source/docs
with the configured identity; do not rewrite upstream history. Review diffs,
secrets and generated-output disposition; leave a clean working tree.
