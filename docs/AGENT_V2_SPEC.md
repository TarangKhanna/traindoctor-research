# TrainDoctor v2: bounded implementation gates

2026-10-30. Owner-authorized local implementation; no policy training or publishing.
Research base 676f784; environment c5a8a9b (runtime inherited from 438b573).
Public heads verified c5a8a9b / 4eea735; working branch demo/agent-v2.

1. Gate 1: actual cached Gemma full agent first, using the existing checkpoint
   migration fixture. Native tools plus strict whole-object JSON fallback;
   twelve response turns, two relaunches, 512 tokens/turn, 8192 context,
   temperature zero, seed zero, thinking off. Malformed outputs are retained.
   Add hash-bound line replacement to reduce duplicated source serialization.
   Exact resulting-file review remains mandatory before sandbox execution;
   review supplies no semantic hints. Failed executable repairs count as loop
   evidence, not successful repair. Original literal patch interface retained.
2. Gate 2: only after actual intervention execution, investigate one gradient-
   normalization integration defect using the same launcher/data/NumPy update
   oracle. Execute no-op, completing-invalid and correct controls; test source-
   rule and full-template replacement baselines. If these solve everything,
   report the shortcut and NO-GO for a challenging learning benchmark.
3. Gates 3/4: terminal verified semantics only; diagnosis/protocol/cost separate,
   infrastructure null. Preserve live/cached provenance. Compare no-tool,
   read-only and full conditions on paired tasks only if task validity passes;
   never present one greedy episode as a repair-rate estimate.
4. Gate 5: readiness/configuration only, no full training. Reuse MLX/LoRA/RL,
   require held-out structures, variation diagnostics, matched SFT origin/budgets
   and numerical parity before optimization. No multi-agent execution now.

Keep >=8 GiB available, stop on >128 MiB swap growth, serialize models. Do not
stop other research. Historical v1.2 and receipts unchanged. No dependencies,
frameworks, broad refactors or cloud computation. A scheduled earlier probe was
removed to prevent overlap. Record raw results, exact commits and limitations.
