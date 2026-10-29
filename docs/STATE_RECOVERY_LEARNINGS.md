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
