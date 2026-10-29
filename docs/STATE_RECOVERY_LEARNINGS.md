# Local research learning log

Append-only project observations and corrections. "New here" does not mean novel
in the literature. Evidence takes precedence over the explanation; unresolved
questions remain unresolved. No historical result is relabeled by this log.

## 2026-10-29 — Scientific catches

1. **The historical gap was mostly execution failure.** The local historical audit
   found 38 V1-pass/V2-fail small-model answers: 11 completing-invalid repairs and
   27 noncompletions. Fixed bad identities explained 10 of the 11 semantic errors.
   The published summary was reproduced, not newly sampled. Do not interpret the
   entire gap as semantic reasoning headroom. Audit provenance remains on the
   separate `audit/state-twins-20261029` branch/worktrees.
2. **Scratch acceptance was contract-consistent.** Historical fresh re-execution
   allows one scratch validation run in lost-work accounting. This is different
   from demonstrating efficient continuation of the original failed incident.
   Preserve v1.2; new continuation/cost objectives need separate versioning.
3. **State dependence exists but does not imply a useful learning task.** The worked
   state-only pair reverses the two selective repairs' validity under actual
   execution. Full restore nevertheless attains the best score in both cases.
   Removing universal validity is unnecessary; establishing a meaningful optimal
   decision would be necessary for the proposed selection experiment.
4. **Correct state rules are desirable.** Matching the oracle with trusted evidence
   confirms the task is observable. It does not show an RL algorithm is needed.
   An identity-only common optimum is the stronger limitation here. Do not add
   arbitrary penalties or obscure evidence merely to defeat a simple baseline.
5. **Independent means conditional independence of the calculation, not magical
   independence of every input.** The NumPy oracle reimplements the continuation
   update/history calculation but shares the specified data and genuine anchors.
   All twelve continuations agree; the maximum accepted tensor error is about
   `5.22e-8`. This is not a separate end-to-end implementation of the whole system.
6. **Control claims must match their intervention.** Evidence ablation prevents
   choosing the selective fix from identical initial observations; robust full
   restoration still works. Metadata perturbations establish structural
   noninterference, not population robustness. A scripted multi-turn transcript
   demonstrates a tool interface, not model investigation skill.
7. **A graded reward does not prove GRPO signal.** No stochastic G=4/G=8 groups were
   measured for the new task. Historical reward-variance logs cannot substitute.
   A greedy common-optimum policy can still generate zero reward variance.

Evidence for items 2–7: [worked report](../experiments/state_recovery_v1/REPORT.md),
[executed action table](../results/state_recovery_v1_loopback_verified/TABLE.md),
[independent recount](../results/state_recovery_v1_loopback_verified/recount.json).

## 2026-10-29 — Runtime, performance and security catches

1. **Template defaults can masquerade as capability failure.** Gemma's initial
   bounded JSON probe spent its token budget in thinking text. Inspection isolated
   the installed MLX wrapper's thinking default. Explicit thinking-off produced
   valid full-restore choices. Preserve both controls; do not blame the model for
   a format failure before checking its actual template and output budget.
2. **Measured inference improvement is narrow but real.** Gemma's native prefix
   cache preserved all five greedy outputs and reduced repeated-prompt total time
   from 35.76 to 24.30 seconds including setup (1.47x). This does not establish
   stochastic rollout, multi-turn episode or training-gradient acceleration.
   Native-token throughput across different tokenizers is not a controlled
   architecture comparison. See [profile recount](../results/local_profile_recount_20261029/summary.json).
3. **Free-form code repair is possible to test without a new execution service.**
   Qwen's single reviewed B_DUP edit applied and completed but preserved the bug;
   verifier parameters and sample-history checks failed. Gemma's bounded edit was
   incomplete, so its functional correctness is unknown. Neither single probe is
   a population accuracy estimate. See [execution receipt](../results/reviewed_qwen_code_20261029/receipt.json).
4. **Avoid a homegrown security boundary.** Review found callable-rebinding and
   contextual-parsing bypasses in an exploratory generic edit validator. The
   validator was discarded; only the exact manually reviewed proposal is replayed
   in the existing sandbox. Do not infer arbitrary generated-code safety.
5. **Check live sockets, not just configuration.** `MASTER_ADDR=127.0.0.1` still
   left a wildcard TCPStore listener. Existing PyTorch socket-ownership APIs fixed
   it without a new runtime. The live `lsof` test, 241 focused tests and repeated
   twelve-continuation execution passed. See the report's transport findings.
6. **A stale manifest was a real preliminary bug.** Review caught full restoration
   without a matching digest update. The implementation now updates the manifest
   and validates every intervention. Preliminary evidence remains separate.
7. **Memory savings and speed are different hypotheses.** MLX quantized LoRA can
   address fit; original CUDA QLoRA mechanisms do not prove Apple throughput gains.
   Saved 0.6B update microbenchmarks were recounted, not rerun or extrapolated to
   4B RL. No new training occurred. Liquid's model-specific speculative decoding
   is not a demonstrated Qwen/Gemma speedup on this Air. Primary sources and
   limitations are recorded in the report.

## Current decision and unresolved work

GO for the reproducible engineering demonstration; NO-GO for training on this
pair as a challenging conditional-repair benchmark. The highest-value next action
is independent reproduction and technical critique of the existing artifact.
External reproduction, Linux behavior, larger-model quality, stochastic group
variation and fresh training speed remain unverified. No new experiment is implied
by adding this log. New findings should be appended with receipts and limitations.

## 2026-10-29 — Reference-free gate and bounded validation

- **Prior art defeats the proposed method novelty.** Proof-of-Learning (IEEE S&P
  2021) already replays recorded batches between checkpoints; actual author code
  `verify_all` implements it. TorchTitan and Megatron also have runtime replay
  diagnostics. The prior-art gate finished before new experimental implementation.
  [Inspected sources and novelty boundary](REFERENCE_FREE_PRIOR_ART.md).
- **The old NumPy continuation calculation was already reference-free in a narrow
  sense.** The new assay makes input isolation explicit: twelve saved-artifact
  checks read no uninterrupted reference or correctness flags; eight accept and
  four reject. Agreement with the same reused NumPy kernel is not a new independent
  validator. All four invalid examples are also caught by archived-component
  comparison; replay adds no detection coverage on this pair.
- **Correct the numerical shorthand.** Accepted final momentum discrepancy is
  exactly `5.2193850685888776e-08`, but the original 4→8 transition discrepancy is
  `7.561030429537041e-08`. The latter is the largest accepted replay error. Preserve
  both and the unchanged absolute `1e-5` tolerance; distinguish tensors from
  discrete-state errors and relative diagnostics near zero.
- **A loaded-byte provenance field was weaker than its comment.** Legacy
  `resume_sha256` looks up an older archived capture by ID. It names the clean
  checkpoint even when the delivered bytes were mixed. The assay does not use it
  as actual per-rank load attestation. Observed continuation captures, correct-source
  and single-writer assumptions remain narrower evidence; v1.2 was not changed.
- **Inputs are reconstructible, not independently byte-attested.** Saved IDs and
  a fixed public demo key/recipe suffice here. Input-value bytes/salt are not
  independently committed by the ledger. Stochastic transforms, mutable datasets,
  scheduler/RNG state and production sharding need stronger instrumentation.
- **Trust cannot be manufactured by replay.** A constructed poisoned anchor with
  coherent later transitions passes relative checks although it differs from the
  original final state by `0.09537724040869114`; pinning original anchor state
  rejects it. Missing input history is inconclusive, conflicting history rejects.
  These controls are not empirical security guarantees or new training incidents.
- **Local feasibility is not scale evidence.** Twelve artifact checks took a total
  `0.21975845692213625` seconds excluding imports/extraction. Two sandboxed CPU/Gloo
  8→12 continuations matched saved tensors bitwise. No inference about NCCL/FSDP/GPU
  tolerances or large-model replay cost follows.

Evidence, exact reproduction commands and remaining unknowns:
[reference-free report](REFERENCE_FREE_REPORT.md). Next action remains independent
human reproduction and assessment of a concrete unmet resume regression. No novel
method, paper readiness, independent recognition or RL benefit is claimed.

### Same milestone — review catches and final evidence supersession

Static review found that removing scratch's step-4 capture could leave only a
final-state check, empty/missing ledgers were not classified, and historical
disagreement could suppress the primary matrix. The bounded fixes now require
scratch checkpoint 4, classify missing required ledger evidence as inconclusive,
and persist primary results before reading old labels. Regression tests cover these
paths; final research gate is 28 passed. Follow-up review found no remaining
false-acceptance blocker within the frozen-artifact contract. Arbitrary malformed
schemas and full load-time attestation remain outside scope.

The earlier `0.21975845692213625`-second assay above is retained locally as a
preliminary packet. The reviewed packet totals `0.21627999999327585` seconds, with
unchanged twelve classifications and numerical discrepancies. This timing variation
is not a performance improvement claim. Both repeatability continuations were reused,
not rerun after the evidence-classification fixes. See the final report and its
linked raw matrix, source manifest, verification output and two static reviews.

## 2026-10-29 — Release clarity and reproducible installation

The historical environment requirement pin predates the state-recovery module.
Installing that pin and then promising all expanded tests would be misleading.
Keep the historical pin intact; document and check editable sibling-source
installation for the new saved-artifact example. The saved recount passes without
new distributed jobs, model inference or a cloud account. README should distinguish
historical Modal isolation from the local backend and completed artifacts from
unfinished agent work. A read-only critic caught both installation and backend
wording issues before release. No result or historical receipt was changed.

2026-10-29 README compression: lead with checksum-valid inconsistent recovery; move the previous detailed landing page intact (relative links rebased) to docs/PROJECT_DETAIL.md. Retain historical evidence and numerical checks. Critic caught that structural well-formedness does not imply checksum validation; wording now explicitly describes recomputed manifest checksums. Documentation changes only.
