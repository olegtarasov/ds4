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
The existing `DS4_QWEN4_MOE_DOWN_PREFETCH` profile flag schedules MXFP4 GPU
loads; it is not n-gram I/O prefetch or routed-expert disk streaming.
A textual merge or unit pass alone is not acceptance. The research hub records
runtime hashes, API comparisons, memory and the final retention decision.

Build pitfall: upstream `make cpu` overwrites the frontend binary names, including
`ds4-server`. CPU portability checks must use an isolated build tree or be
followed by removing/relinking the generated frontend binaries for Metal. Merely
running `make ds4-server` can consider the CPU binary up to date. Verify the
Metal framework link before copying a runtime snapshot. The first candidate API
startup caught this mismatch before loading the model; that failed diagnostic
is retained in the hub and excluded from timing results.

The four-key attention experiment is preserved on
`codex/qwen-attention-experiment` and reverted from the working line: it passed
numeric checks but did not show a consistent 8K API gain. Its combined 64-key /
wide-merge trial had lower, variable generation rates; a 32-key follow-up was
stopped by the host memory-pressure guard without new swap-outs. That safety
stop does not establish a kernel-specific memory regression. Keep the
indexer/gather candidate and original split/merge settings for further checks.

When running the complete kernel suite, leave `DS4_QWEN4_IK_RING` unset because
that suite also intentionally exercises batched-row paths that our graph guard
rejects. `DS4_TEST_IK_RING=1` independently wraps the synthetic raw-key tensor;
the dedicated payload-ring test validates the graph serialization contract.
Imported live test launchers were written for other hardware/chunks: use the
hub's memory-guarded owner on this Mac, not their defaults without review.

## Working on this Mac

After edits, build the Metal frontends (`make -j8 all`) and relevant focused
checks. Keep CPU portability builds isolated or explicitly restore/relink Metal
frontends afterward, as described above. Do not use inherited live launchers
with different chunk/residency defaults for this large model.

With the installed router's original model unloaded and sufficient host memory
free, run a guarded API check from the sibling hub. Use a fresh output directory
for every run; this command is host-specific and assumes the existing model and
private API key file are installed:

```sh
cd ~/Projects/local-llm-serving
python3 scripts/api-setup-compare.py --engine dwarf \
  --runtime "$HOME/Projects/ds4" \
  --profile "$HOME/Projects/ds4/profiles/m2-max-qwen-q4.json" \
  --cases short 8k --repetitions 3 \
  --output artifacts/dwarfstar-fork-manual-check
```

The harness owns one temporary backend and authenticated gateway, checks memory
pressure/swap, and stops them on completion. It preserves the installed service
and Pi configuration. For a reproducible comparison, snapshot both the binary
and `metal/` sources under ignored artifacts before changing source again.

## October 18 qualification

The retained runtime snapshot was built from `9f382b8` (later fork changes in
this task are documentation only). Server SHA-256:
`3c86583f08dbd4e504b037e434690ec6487e629655999f988909d26f0c980502`.
Qwen shader SHA-256:
`cfdbf01b02f0856d2a111f1451d1382854fcf8e94effb384699966f2d33ba3ff`.
Full runtime/shader-tree/profile identities and raw-log hashes are recorded by
the sibling hub in `evidence/dwarfstar-fork-2026-10-18/summary.json`.

Locally measured on M2 Max / 96 GB, existing Q4, full 262144 capacity, chunk
2048, one session, f16 K/V, MTP off, temperature zero and seed 42:

- Fresh short/8K three-repeat comparisons after Docker exited pass with matching
  prompts, output text and token counts. Median 8K generation is 27.86 versus
  27.87 tokens/s: no established gain at that length. An earlier apparent 10%
  screen gain did not repeat and is not the accepted speed claim.
- One near-full pair (261617 input, 256 output) passes retrieval with identical
  text/usage. Generation rises from 18.59 to 24.67 tokens/s (32.7% in this pair),
  while TTFT stays about 764 versus 766 seconds and total about 778 versus 777.
  Repeat before treating this as a distribution or general serving advantage.
- Live Chat/tool calls and result consumption pass. For a withheld long tool
  call, the baseline's 32.9-second SSE gap becomes 5.1 seconds with six keepalive
  comments. Other imported branches have model-free coverage; this does not
  qualify Pi or live Responses/Anthropic clients or history/cache replay.
- All accepted runs above have normal pressure and zero new swap-outs. Near-full
  system wired peaks are about 80.97 versus 81.76 GiB. A prior host-memory guard
  stop with an approximately 11.3 GiB VM is excluded; the owner freed memory.
  These results require headroom and are not a guarantee alongside arbitrary
  other workloads. The guard remains enabled.

Metal `make all`, model-free server/agent tests, focused numeric Qwen kernels
(normal and wrapped synthetic raw keys), n-gram fixtures, ring roundtrips and a
CPU portability build pass. CPU inference, CUDA/ROCm/TP and SSD-heavy integration
checks were not executed. The attention experiment is reverted. The retained
API/indexer line is a research baseline, not a broad coding-readiness claim.

The frozen four-task Python coding suite, two no-feedback attempts each, scores
4/8 strict passes: SQLite and DAG planning pass twice, SSE parsing and download
resume fail twice. All eight prompt and visible output hashes match the prior
working DwarfStar run, including failures. Pooled generation median in this run
is 24.67 tokens/s; it is not a fresh controlled coding-speed comparison. Pressure
stays normal and no new swap-outs occur. Detailed findings and acceptance limits
are in the hub's `docs/dwarfstar-fork-2026-10-18.md` and reviewed evidence.
