# Local research learning log

## 2026-10-30 — Bounded masked-loss follow-up

The owner reopened one source-repair incident on `demo/masked-mean-v1`. Existing
external checkpoint copies plus an independent NumPy transition calculation can
check per-step rank0 state when checkpoint frequency is increased through existing
configuration. This remains conditional on exact-source review: serialized state
is not a live-memory attestation, and input delivery is not direct consumption.

Both correctly implemented loss-side and gradient-side global normalization pass
the fixed regressions. A source rule solving all cases is legitimate correctness,
not evidence of a failed task. Conversely, a hardcoded control must actually fit
the visible case before hidden failures establish overfitting; the initial
fixed-count control did not, so a corrected control was executed separately.

Actual model traces still do not establish autonomous repair. Qwen investigates
and emits a clipped prose/code proposal; Gemma emits a tool-role delimiter after
inspection. The frozen controller prematurely accepts both as final answers.
Regression tests correct future classification, but the cause of Gemma's delimiter
emission remains unresolved. Do not call these runs proof of a model reasoning
limit or infer that inspection helps. Qwen's verbatim fragment, separately reviewed
and executed, fails on a Python integer/tensor API error. No semantic correction,
trajectory credit, model retry or training follows. See the raw final packet.

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

## 2026-10-29 — Actual local native-tool agent, resource-blocked repair

- The owner authorized one code-agent milestone after the verifier audit; this is
  an explicit scope change, not reversal of the failed novelty/RL conclusions.
  Reuse existing Tier B reads, MLX-LM native templates/parsers, isolated launcher,
  trusted captures and NumPy math; no new framework/dependencies.
- One constructed checkpoint migration bug really completes with wrong momentum.
  An authored fix passes current and legacy restarts; skipping work fails trusted
  validation. Scratch remains semantically correct but does not repair the loader.
  Scripted positive controls are not model-agent achievements.
- Cached Gemma performed three actual native calls. Schema-valid tool use is not
  automatically useful inspection: it used a file path with a directory-list tool.
  No patch was proposed or executed before the resource stop.
- The first run encountered available RAM below 8 GiB and swap growth above the
  guard. Subsequent low-memory observations persisted without a matching model
  process. Root cause is unresolved; do not infer a model memory bug, blame an
  uninspected process, lower guards or convert infrastructure failure to reward zero.
  The initial runner failed to persist that terminal status automatically; its
  raw transcript was preserved and explicitly closed as unscored. Later exception
  accounting and mid-generation checks address this implementation gap.
- Review caught a causal reward bug: an earlier successful patch/run could otherwise
  be credited to a different final patch. Bind the final file hash to actual
  execution and both hidden restarts. Preserve evidence IDs across security-review
  pauses, reject extra native call blocks, preserve pending state on CLI validation
  failure, and record failure/verification timing without fabricating missing data.
- Source-only repair may be enough for this small defect; four planned greedy
  conditions cannot establish GRPO group variation or held-out generalization.
  No model training or multi-agent execution is justified by the stopped run.

Evidence and remaining work: [local-agent report](LOCAL_AGENT_V1_REPORT.md).

2026-10-29 resource catch: an adequate external preflight did not guarantee capacity at the runner pre-load guard a few minutes later. Cached Qwen invocation stopped before loading (zero loads/turns); post-stop available RAM 5.1192169189453125 GiB and swap 8.299072265625 GiB are later observations, not the failing-guard measurement. Root cause unresolved. Preserve the unscored result and stop; do not reduce guards or attribute model ability to this infrastructure outcome.

2026-10-29 Qwen protocol finding: adequate startup RAM allowed genuine tool-use, but did not guarantee episode capacity. One generated apply_patch block repeated escaped newlines and reached 512-token length termination; parser correctly rejected it. This is an observed protocol failure, not a completed semantic-repair failure. The next generation was resource-interrupted and unscored. Partial/complete token and timing counts are separate in the raw summary; trigger-specific memory counters remain missing. Training permission does not resolve task-validity, data, held-out mechanism or matched-comparison gaps.

2026-10-29 retry catch: failing-sample telemetry isolates the guard predicate (7.7312469482421875 GiB available, swap -40 MiB): low RAM, not swap growth. It does not isolate the process causing RAM decline. Raw outputs match all eight comparable previous greedy responses, including the malformed patch, so identical deterministic retries should not be counted as independent evidence or expected to fix protocol behavior. Save and probe one decoding/model change against the already saved context before another full episode. No guard changes; 12 focused tests pass after correcting a test-only constructor mock.

## 2026-10-30 — Agent feasibility catches

- Actual native-tool traces are not proof of autonomous repair. Gemma's four edit
  attempts were inapplicable. Qwen's applicable last-turn edit was applied after
  exact security review, but the model did not choose relaunch before exhaustion.
- A real line-editor boundary bug glued a following statement to replacement
  content. The original proposal failed isolated execution. Only a separator was
  normalized by the corrected editor; that control completed but still violated
  model/momentum semantics with valid hashes and sample history. The tool defect
  and wrong repair logic therefore have distinct executed evidence.
- No valid G4/G8 groups were sampled. Undefined fractions remain null, not zero;
  authored positives and dependent retries cannot be repackaged as policy groups.
- Fixed source logic repairs this constructed format defect; unconditional full
  restoration remains optimal for both state cases. These cases do not establish
  investigation benefit or held-out structural generalization. STOP expansion;
  do not manufacture difficulty or substitute more infrastructure for a result.
- The fresh retest stopped at the RAM policy floor with no swap growth. This was
  not OOM and does not prove the model cannot fit. Sustained rollout throughput
  and a complete model-chosen relaunch/verification sequence remain unmeasured.
- Harbor hidden test delivery is not scoring isolation. A separate verifier,
  source-only transfer and real privilege/capability/private-file/network proofs
  are necessary for this adapter's declared boundary. Static/unit review is not
  a Docker runtime proof. Setup approval is pending; no fake compatibility claim.

Exact receipts and provenance: AGENT_V2_REPORT.md and results/agent_v2_*.

2026-10-30 — Harbor runtime gate passed on one existing task: correct1/no-op0/completing-invalid0/tamper0, with matching native source/verdict controls. Completing an optimizer-reset repair with valid hashes and sample history still fails semantic transitions. Shared-verifier cross-backend agreement is not independent mathematical validation. Integration fixes were limited to reproduced provider-policy, read-only transport, tmpfs-copy, dormant-interface and stale reward.txt issues. Default lint keeps its root-capability finding; the narrow exception and actual Docker isolation are explicit. See HARBOR_VALIDATION.md and the scripted raw packet. No autonomous agent or learning result was added; STOP training remains.

Closure learning: static tests cannot establish SDK transport semantics, and a successful CLI exit can wrap an errored trial. Inspect actual trial/verifier receipts before claiming a result; keep failed setup receipts and exact image/source identities. Reusing Harbor's own transfer and separate-verifier contracts was sufficient—no generic adapter or scheduler was needed.

2026-10-30 release catch: CI does not install the optional Mac profiler. The agent module can now be imported without psutil; actual inference still refuses without the memory guard. Review caught that this refusal must follow completed-state and bad-approval checks, while preceding new inference/approved-patch work. Completed, exhausted and invalid-approval paths are tested without psutil; all 22 focused tests pass against the historical CI environment source as well. No dependency addition or new model execution.

2026-10-30 CI root cause: a report's relative Harbor link resolved through a sibling worktree, so local filesystem checks passed while a standalone GitHub checkout failed. The URL is now public and the existing release checker validates every tracked Markdown link against tracked files/directories. A deliberate regression with the old link is rejected despite the sibling existing; the restored public link passes. This strengthens release validation without changing experimental evidence.

## 2026-10-31 — Keep the install pin aligned with the featured example

The old requirements pin predates `state_recovery_v1`; the README previously
needed a sibling source checkout to provide it. Installing environment 221096d
from the pinned Git URL supplies the module directly. The current setup now uses
that revision, and CPU CI recounts the fresh archive. This verifies saved evidence,
not fresh distributed execution or an independent external reproduction.
