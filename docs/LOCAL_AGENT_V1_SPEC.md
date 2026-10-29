# One real local agent: bounded specification

2026-10-29. Research base `3deafaa`; environment source `438b573`, unchanged.
Latest owner request supersedes the previous session's no-inference constraint.
No policy-model training, cloud execution, paid APIs, dependency changes or publication.

## Incident and scientific limits

One application checkpoint-format migration regression: the writer stores momentum
under parameter names while the reader still filters indexed legacy keys. Model
weights and sampler restore; momentum silently cold-starts. Training completes on
both ranks but the continuation deviates from the intended SGD trajectory.
This is a constructed integration defect in an actual executed MLP, not a claim
to reproduce an external production incident. PyTorch documents that optimizer
parameter identifiers/names and restoration mapping require care. No mutation diff,
fault label, expected patch, oracle trajectory or truth verdict is shown initially.

The task is to repair reusable checkpoint loading while retaining old-format
compatibility. Reading source alone may suffice; the experiment must disclose that.
An older indexed checkpoint and current named checkpoint are hidden restart tests.
A scratch rerun is semantically legitimate but does not by itself repair loading;
report trajectory correctness separately from durable code-repair success.
No claim of a challenging benchmark follows from one incident or a fixed patch.

## Reuse and execution boundary

Reuse Tier B Episode's inspection functions/schema checks and public records,
existing `resubmit`, manifest, trusted captures, sandbox and NumPy update oracle.
Keep the task/runner inside this research experiment; do not refactor core APIs.
Historical `finish` reexecutes a fresh incident, so this task's terminal grading
must instead inspect the agent's actual executed continuation plus hidden restart
regressions. No change to v1.2.

Expose a bounded subset of existing inspection tools, checkpoint summaries and
literal exact-match `apply_patch` to `job/ckpt.py` only. This is an application
constraint, not a security proof. No shell or arbitrary Python tool. Model edits
are staged as text. The supervising assistant reviews the exact resulting file
for unsafe behavior and authorizes its content hash before execution; no semantic
coaching or patch correction. Record this assistance and exclude it from model skill.
Do not revive the discarded generic AST validator. All edited workers use existing
fail-closed macOS sandbox-exec; residual loopback access remains documented.

## Agent baseline and controls

Use cached Gemma E2B and Qwen3-4B with native chat templates and installed native
tool parsers, thinking off. One model process at a time, >=8 GiB available, stop
on swap growth >128 MiB, no downloads or trust_remote_code. Default seed 0,
temperature 0, <=12 calls, <=2 relaunches, <=512 output tokens/turn, <=8192 input
tokens. Persist actual raw tokens/text, prompts, calls, observations, diffs, timings,
model snapshot/runtime, memory, execution result and terminal verdict.

Prespecify one full-evidence and one reduced-evidence episode per model (four
dependent greedy controls, not an accuracy estimate). Reduced evidence withholds
checkpoint tensor summaries; source/log/config access, task, budgets and seed stay
the same. This may show no difference; no investigation-benefit claim is guaranteed.
Use one neutral policy prompt; no answer-specific repair hints or post-failure tuning.

Before model runs: execute unchanged negative and reviewed correct positive
continuations and compatibility checks. Test patch path/shape denial and an
infrastructure-error outcome; infrastructure is unscored, never a negative reward.
Terminal reward is binary verified reusable repair success; diagnosis, protocol,
patch application, actual completion/semantics and costs are separate fields.
Hidden tests must not call agent-editable checkpoint functions in the host process.

## Gate and stop

No RL recommendation from four greedy episodes. Future stochastic G=4/G=8 signal
and matched SFT controls remain prerequisites, along with nontrivial baselines and
held-out mechanisms. Design an equal-budget investigator/reviewer comparison only;
do not implement multi-agent execution. Stop after these controls, trajectories and
an honest report, including unsuccessful proposals and security-review limitations.
