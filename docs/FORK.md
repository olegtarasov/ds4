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

## Selective imports under evaluation (October 18)

API repairs preserve original contributor authorship through cherry-picks:

| Upstream PR | Original commits | Scope |
| --- | --- | --- |
| antirez/ds4#1006 | `4610665e` | SSE comments while tool/reasoning output is withheld |
| antirez/ds4#1134 | `1074193f` | Preserve mid-list system messages and live tails |
| antirez/ds4#1155 | `8a8b2a23`, `8922c813`, `8f61eced` | Actual backend-prefix validity and rebuild progress |
| antirez/ds4#1169 | `995fba52`, `395a3edd` | Exact tool spans, EOS and closed-turn frontiers |
| antirez/ds4#1175 | `9d471c45`, `4e8706de` | Parent-history validation and user follow-ups |

The #1175/#1134 conflict was resolved by retaining both canonical parent-key
construction and DeepSeek's fallback when a late system message cannot be
appended to live KV. These changes do not enable disk KV caching.

Compute candidates from `paperniuk/ds4`:

- `c4045f71`: shared four-head block scorer and vector indexer. Retain upstream
  device defaults in code; the explicit M2 profile selects the candidate.
- `ded46d5f`, `9b7bf1a3`: coalesced full and prefiltered radix gathers. The former's
  unrelated cold-matvec benchmark edits require foreign Q2_0 support and were
  omitted; existing indexer correctness fixtures remain in use.

These are compute changes, with no additional SSD reads or disk-cache policy.
A textual merge or unit pass alone is not acceptance. The research hub records
runtime hashes, API comparisons, memory and the final retention decision.

Build pitfall: upstream `make cpu` overwrites the frontend binary names, including
`ds4-server`. CPU portability checks must use an isolated build tree or be
followed by removing/relinking the generated frontend binaries for Metal. Merely
running `make ds4-server` can consider the CPU binary up to date. Verify the
Metal framework link before copying a runtime snapshot. The first candidate API
startup caught this mismatch before loading the model; that failed diagnostic
is retained in the hub and excluded from timing results.
