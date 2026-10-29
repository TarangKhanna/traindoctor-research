# TrainDoctor research: current local work

## Scientific goal

Make the recovery mechanism and negative learning results independently assessable.
Current authorized milestone: [TrainDoctor v2](docs/AGENT_V2_SPEC.md), superseding
older one-incident/no-expansion limits only for the explicitly gated upgrade.
Reuse the local loop, fix the smallest blocker and move sequentially through gates.
No policy-model training, cloud computation, publishing or generic orchestration.
Preserve the completed reference-free audit and its novelty NO-GO.
Read [current status](docs/STATUS.md), [the worked report](experiments/state_recovery_v1/REPORT.md)
and [the learning log](docs/STATE_RECOVERY_LEARNINGS.md) before proposing more work.
One state-only incident pair is complete. Do not expand it into a benchmark,
orchestration system or RL run without a new, concrete scientific justification.

- Preserve historical receipts, protocols and preregistration contradictions.
  A new objective needs a new version, rationale and separate results.
- Cross-tabulate before fitting explanatory models. Distinguish semantic-invalid,
  noncompletion and infrastructure failure; V1-pass/V2-fail is not one failure type.
- Keep correctness separate from cost. Robust restoration and scratch remain valid.
  Do not force reward variation by making sound engineering actions incorrect.
- Treat paired twins, repeated rollouts and timing repeats as dependent. Seed splits
  do not establish generalization to a new causal mechanism.
- A state rule matching the oracle is evidence of observability, not an automatic
  failure. A common optimal action defeats the proposed conditional-selection claim.
- Describe oracle independence precisely, including shared data/committed anchors.
  Never use verifier flags as independent ground truth or script transcripts as
  model evidence. Make no novelty/significance claim without supporting evidence.

## Lean, safe experiments

- Reuse existing scripts, raw receipts, stdlib and supported library APIs. Before
  adding infrastructure, document the missing capability and why reuse fails.
  No dependency additions/bumps without authorization; no general task generator,
  dashboard, new code-security framework or broad refactor.
- Work only on isolated branches/worktrees; preserve main worktrees. No cloud
  experimental computation, paid actions, pushes, merges or publication.
- Local inference/training is authorized for a justified experiment in this session.
  Authorization is not a reason to train on a task with no informative comparison.
- Serialize model processes, keep at least 8 GiB available, record swap growth and
  use loopback-only services. Do not execute unchecked generated code.
- Record model/snapshot, quantization, tokenizer/template/thinking settings, runtime,
  context/output budgets, warmups, repeats and cache conditions. Preserve raw outputs.
- Measure useful episode latency as well as token throughput. Do not transfer vendor
  hardware claims or model-specific acceleration to this Mac without measurement.
- QLoRA memory fit is not a throughput result. Require loss/gradient/optimizer parity
  and a measured bottleneck before changing training kernels or group computation.

## Evidence and learning discipline

Use documented preflight and narrow checks; execute the smoke path for behavior
changes. Generated results must come from scripts with saved configuration. Recount
raw evidence before reporting numbers; label historical recounts separately from
new executions. Documentation-only edits need link/diff checks, not fresh training.

Append verified findings, catches and unresolved questions to
[the learning log](docs/STATE_RECOVERY_LEARNINGS.md) at meaningful milestones.
Include evidence, implication and limits; explicitly supersede mistaken claims.
Update status at closure. Keep this lightweight: no new memory/logging framework.
