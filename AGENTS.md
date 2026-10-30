# TrainDoctor research: current local work

## Owner-authorized protocol-corrected trials — 2026-10-30

On `demo/protocol-corrected-20261030`, run one Qwen and one Gemma episode on
the existing masked-loss task with the corrected controller and historical
budgets. Only if Qwen hits the output cap while attempting an edit, permit one
separately labelled higher-output-budget diagnostic. Preserve task, model
revisions, hidden regressions, semantic grading and execution isolation. No
training, new tasks/frameworks, automatic retries, publication or pushes.
This current authorization supersedes older inference holds for these trials only.

Paused episodes must retain their recorded fixture and controller/task code
identity before reconstruction or approval. Do not migrate legacy unbound paused
trajectories in place; preserve completed receipts. The bounded trials are closed:
further inference or training needs a new owner-authorized experiment.

## Owner-authorized hiring release — 2026-10-30

Prepare the completed masked-loss work and sanitized receipts for public review,
use a native subagent critic (no Claude Code this session), and push only after
release checks pass. This supersedes the earlier publication hold for this release.
It authorizes no new model episode, training, dependency change or task family.

## Owner-authorized follow-up — 2026-10-30

On `demo/masked-mean-v1` only, implement the one-task plan in
`docs/MASKED_MEAN_V1.md`, execute its gates and at most four local model episodes.
This supersedes the older no-new-task/no-inference hold for this bounded work.
No training, publishing, dependency changes or new infrastructure follows.

## Scientific goal

Make the recovery mechanism and negative learning results independently assessable.
Completed evidence: [reference-free validation](docs/REFERENCE_FREE_REPORT.md),
after the [prior-art gate](docs/REFERENCE_FREE_PRIOR_ART.md). Reference-free means
conditional on an explicit trusted anchor, input recipe and update specification;
replay itself is established prior art. Current owner-authorized work is release
preparation and clear documentation, not more model runs or training. Keep unfinished
local-agent outcomes explicit; do not claim successful multi-turn code repair.
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
  experimental computation or paid actions. Publication requires explicit owner approval.
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

## Feasibility closure (2026-10-30)

Read docs/AGENT_V2_REPORT.md before any further work. STOP training and task/benchmark
expansion on current cases. The already authorized one-task Harbor interoperability
proof has owner-approved isolated setup and local execution; it is not a learning task redesign.
The earlier validation-only publication hold is superseded by the release
authorization below; no new model runs are authorized.
Do not restart model inference automatically or pool retries as RL reward groups.

Harbor runtime gate is complete: read docs/HARBOR_VALIDATION.md and recount the
saved final packet before repeating local trials. This is one interoperability
smoke task; preserve STOP on training/benchmark expansion. No automatic model
restart, publishing or Harbor Hub upload. Docker Desktop/images were retained;
only our recorded trial containers/volumes were cleaned.

## Authorized release — 2026-10-30

The owner now authorizes publishing the reviewed local-agent/Harbor work and
concise README updates to both existing GitHub repositories. This supersedes
earlier publication holds for this release only. Preserve historical receipts,
review the full outgoing diff and use a normal fast-forward push. No training,
new inference, Hub upload or task expansion follows from this approval.
