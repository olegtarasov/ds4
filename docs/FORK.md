# Oleg's DwarfStar fork

Upstream: https://github.com/antirez/ds4
Fork: https://github.com/olegtarasov/ds4
Initial upstream base: `fc80bd695da76ee14bcb86820bf900a5cdfda806`.
Local checkout: `~/Projects/ds4`. The sibling `~/Projects/local-llm-serving` stays
the research hub for all serving engines. It owns benchmark orchestration and
reviewed cross-engine evidence; do not copy that project into this repository.

## Retained baseline

The first fork changes transfer the previously tested Qwen patches from the hub:

- Target-only Metal model spans and allocations: omit unused MTP state when MTP
  is off; keep large n-grams out of the GPU views.
- M2 tuning: bounded grouped views (4 GiB maximum), optional FP32 raw-indexer
  ring, checkpoint layout handling and single-session guard. The native context
  and complete block-key/K/V history remain available.
- Greedy CLI prefill fix: propagate the requested chunk into the Qwen argmax
  graph instead of silently allocating the default 8192.

Sources: `../local-llm-serving/patches/dwarfstar-qwen-target-only-metal.patch`,
`dwarfstar-qwen-m2-tuning.patch`, and `dwarfstar-qwen-greedy-prefill.patch`.
The optional allocation diagnostic patch is not part of the serving baseline.
Previous capacity and API results are in the sibling hub's October 17 reports;
they do not automatically qualify a newly compiled binary.

The host profile is `profiles/m2-max-qwen-q4.json`. It is consumed by the hub's
explicit `scripts/api-setup-compare.py` harness; it is not a native ds4 flag file.
Model payloads and authentication stay outside this repository.

## Optimization boundary

Keep the same Q4 payload, native context, chunk, sampling and memory guard when
comparing kernels. Test vector index scoring/gathers and selected attention
scheduling; evaluate API keepalive/correctness independently. No new SSD-heavy
leads: no expert streaming, disk KV writes or additional n-gram prefetch. The
existing BF16 table still requires SSD reads; full residency of this 165 GiB
file is impossible on the 96 GB host. No new quant, ANE, YaRN or lossy KV change.

## Validation contract

Build Metal and the CPU reference target (do not run this huge model on CPU).
Run model-free server and focused Qwen kernel/ngram/ring checks. Run one guarded
API model at a time using the hub, full 262144 context, chunk 2048, MTP off,
temperature 0 and seed 42. Compare identical short/8K prompts and final near-full
retrieval, plus bounded executable coding tasks when accepting a candidate.
Report pressure, wired/compressor memory and swap-out deltas, not only RSS.
No claims about CUDA/ROCm/TP, general coding readiness or live disk replay without
those tests. Raw outputs belong in ignored artifacts; reviewed results and final
acceptance decisions remain in the research hub.
