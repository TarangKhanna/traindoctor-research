# TrainDoctor v2 feasibility and Harbor handoff

**Decision: STOP model-training and benchmark expansion on the current task set.**
The bounded study produced real model tools/edits, executed negative repairs and
positive semantic controls. It did not demonstrate an autonomous model choosing
relaunch and observing its result, inspection benefit or a valid RL comparison.
Complete only the already scoped Harbor interoperability proof after setup approval;
that packaging is not evidence of a stronger learning task.

## Provenance and implementation

Public heads verified at entry: environment `c5a8a9b`, research `4eea735`, newer
than the supplied starting heads. Existing agent branch `676f784` was reused via
isolated `.wt/agent-v2-research`, branch `demo/agent-v2`. Source milestones:
`f7e7b89` (tools/protocol), `d0509b4` (one sampled retest), `6e0045a`
(line-boundary fix, budgets and audit). Environment `.wt/harbor-v1-env`, branch
`demo/harbor-v1`, commit `cc69e14`, branches from the current public environment.
Research agent history predates the public README compression; it has not been
merged into public main. Earlier state/replay worktrees and uncommitted historical
audit files were inspected and preserved.

Reused `RepairEpisode`, Tier B schemas/reads, launcher, trusted captures, sandbox,
checkpoint manifests, NumPy update oracle and installed MLX-LM native templates.
Changed research files: `experiments/local_agent_v1/{run,task,audit_v2}.py`,
`tests/test_local_agent_v1.py`, scoped instructions/spec/status and this report.
Added diagnosis submission, clear file-tool contracts, complete JSON fallback
(no partial repair, duplicate/extra keys rejected), hash-bound inclusive line edits,
malformed-response history, frozen decoding/budget configuration, budget feedback,
preload-free exhausted-episode finalization and separate inference/tool timings.
No policy training, new model downloads, cloud experiments or publishing occurred.

## Actual model evidence

| Episode | Responses | Schema-valid calls | Applicable edits | Agent relaunches | Terminal reward | Inference s | Prefill / decode s | MLX peak GB |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| Gemma full, 12 calls | 12 | 12 | 0 | 0 | 0 | 147.92 | 82.69 / 52.87 | 3.446 |
| Qwen sampled full, 12 calls | 12 | 12 | 1 | 0 | 0 | 292.37 | 241.90 / 47.50 | 3.479 |
| Qwen retest, 16 calls | 9 | 8 | 0 | 0 | unscored | 307.50 | 215.69 / 89.09 | 3.453 |

These are individual debugging episodes, not repair-rate estimates or a matched
model comparison. Gemma used greedy decoding. Qwen used temperature 0.2 and seed
0 plus turn index; all used thinking off, 512 output tokens/turn and 8192 context.
Snapshots: Gemma `238767527555cb75a05732a84dff5d6ba0dd6809`; Qwen
`50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`. Runtime versions, prompts, raw tokens,
observations, diffs, memory and partial outputs are in each trajectory.

- Gemma exhausted its budget proposing identical/non-applicable replacements.
- Qwen generated an applicable line edit at its last response. Exact security review
  approved the staged file, and the tool applied it; the agent had no calls left
  to relaunch. Diagnosis was not submitted. Its terminal reward is zero.
- That edit exposed a real editor defect: missing trailing newline joined the next
  Python statement. The original file was preserved and independently executed in
  the macOS sandbox; execution failed (`PYTHON_ERROR`), not semantic corruption.
- The editor was fixed and tested for interior replacement, deletion and EOF. A
  separately versioned fresh retest received sixteen calls and budget observations.
  It produced a malformed capped block, then hit the RAM guard. Reward is null.
  The captured available RAM was 8492023808 bytes; swap growth was zero. This is a
  policy guard interruption, not an OOM or proof that the model cannot fit.

[Gemma transcript](../results/agent_v2_gemma_full_20261030/TRANSCRIPT.md),
[Qwen transcript/diff](../results/agent_v2_qwen_full_20261030/TRANSCRIPT.md),
[retest transcript](../results/agent_v2_qwen_retest_20261030/TRANSCRIPT.md).
Exact-file review was supervising AI security review, not human validation or
semantic coaching. Workers ran under macOS sandbox-exec with its documented
loopback residual. This is not gVisor-equivalent or a general security guarantee.

## Applied intervention and task-quality controls

A diagnostic control interpreted the SAME generated arguments using the corrected
line editor. It added only the missing line separator; it supplied no semantic
repair and is not a fresh autonomous trajectory. The resulting source was reviewed
and executed through the same isolated launcher. It completed, preserved sample
history and validated checkpoint hashes, but the independent NumPy comparison
rejected model and momentum state. Maximum absolute difference: `0.24375582367959417`
under the existing absolute `1e-5` tolerance. Whole execution/grading control took
`4.928122500015888` seconds. [Raw receipt](../results/agent_v2_qwen_editor_control_20261030/receipt.json).
Thus fixing editor syntax alone did not make the proposed repair correct.

| Existing incident | Executed baseline evidence | Investigation/shortcut finding |
|---|---|---|
| Checkpoint-format migration | New unchanged restart completes incorrectly; fixed source patch passes current and legacy restarts | Writer/reader mismatch is visible in source; a fixed source rule solves this constructed defect. Dynamic inspection necessity is not demonstrated. |
| State-only recovery pair | Prior twelve executed continuations, eight valid/four invalid, independently recounted | Selective validity reverses, but full restoration is correct and optimal in both states. No evidence is needed to choose that safe action. |

[New live baseline results and model metrics](../results/agent_v2_feasibility_20261030/summary.json).
The fixed source baseline is authored, not model inference, and is specific to this
format regression. The second row reuses cached executed evidence, not new runs.
These are dependent paired states, not independent tasks. No new task family was
built to evade the shortcuts. Historical seed leakage remains documented; there is
no new statistical leakage/generalization result. Irrelevant-metadata controls for
the original pair remain structural, not newly executed population robustness.
A no-inspection fixed-patch baseline is supported; matched no-tool/read-only/full
model evaluations were not run after the task-quality gate failed. No claim that
tools improve repair probability follows.

## Rewards, bottlenecks and training readiness

Terminal semantic repair is separate from protocol, diagnosis, applicability,
completion, history, cost and infrastructure. Real controls include accepted
correct restarts, checksum-valid completing-invalid restarts, a rejected skip-work
exploit, legitimate scratch computation and an original syntax failure. Scratch
can be semantically valid while failing the separately stated durable-loader-fix
objective; it is never called semantic corruption. Unit tests cover invalid edits,
stale hashes, multiple/truncated calls, infrastructure-null rewards, premature and
repeated finish, stale executed-patch credit and exhausted-budget finalization.

G=4 and G=8: **zero prespecified groups sampled; useful-variation fractions and
zero-advantage fractions are unavailable (null), not zero.** Pooling different
models/configurations or dependent retries would fabricate a group estimate.
The implemented diagnostic excludes infrastructure groups and is tested with
explicit synthetic groups. Positive authored controls are not policy rollouts.
No diagnosis-accuracy estimate or unseen-mechanism evaluation was established.

Prefill dominates the sampled Qwen episode. Existing prefix-cache controls remain
narrow prior evidence; multi-turn cache parity and speedup were not assumed or
implemented. The measured model failures include serialization, no-op/applicability,
wrong repair logic, an editor defect, budget exhaustion and RAM guard interruption.
There is no new gradient/learning-dynamics measurement. Existing `rl/first_update.py`
and `rl/lp_patch.py` provide the prior local update/reload and completion-logprob
work; they do not yet provide validated multi-turn assistant-only loss masking.

A future matched comparison would start SFT/continued-SFT/GRPO from the same cached
base and SFT adapter, use identical quantized LoRA parameters, hold out code/task
structures, mask tool observations, report tokens/executions/wall time separately,
and verify logprob/gradient parity before optimized kernels. Current barriers:
no successful model repair dataset, no nontrivial task/evaluation split, no sampled
semantic reward groups and no resource-feasible model-requested repair/relaunch/verification loop demonstrated. No training
configuration is approved by these findings; do not start an RL run.

## Harbor interoperability: implemented, runtime unverified

One existing B_DUP task is packaged in the environment branch. Standard task files
use Harbor schema1.4/native separate verification, source-only transfer and existing
v1.2 checks. The task-specific Linux bridge requires actual uid/capability,
network-denial, private-file and worker-termination proofs before accepting execution.
It protects trusted records and exports bounded evidence; infrastructure emits no
reward. Static review caught image-copy, capabilities, hardlink and shared-log
issues; those were fixed. No core verifier behavior was changed.

Harbor is not installed and Docker Desktop was stopped at preflight. Approval for
an isolated CLI install/backend startup is pending under repository dependency rules.
Therefore Harbor lint/oracle/nop/completing-invalid trials, image digests and startup
or verification overhead are NOT established. Static Compose parsing and 18 adapter
unit tests are not runtime isolation proof. See the environment's
[Harbor guide](../../harbor-v1-env/docs/HARBOR.md) for exact conditional commands,
primary sources, dotenv safety and the external-host-MLX agent path. No generic
adapter framework, hosted rubric, Hub upload or agent swarm was added.

## Reproduction and role mapping

From this research worktree with the prepared pinned Python3.12 environment:

```bash
export TD_KEY=state-recovery-demo-key
export PYTHONPATH=/absolute/path/to/traindoctor-env:.
uv run --no-project --offline /path/to/prepared/python -m pytest -q tests/test_local_agent_v1.py tests/test_state_recovery_oracle.py tests/test_menu_v3_analysis.py
python3 -m zipfile -e results/local_agent_v1_fixture_20261029/raw_evidence.zip /tmp/new-td-fixture
uv run --no-project --offline /path/to/prepared/python -m experiments.local_agent_v1.audit_v2 --fixture /tmp/new-td-fixture/fixture --replay-controls --out results/new-baseline-recount
```

Native model command (using the same prepared interpreter):

```bash
python -m experiments.local_agent_v1.run \
  --fixture /tmp/new-td-fixture/fixture --model /absolute/cached/snapshot \
  --out results/new-episode --temperature 0.2 --seed 0 --turns 16
```
 Use offline flags, one model and the resource
guards. A staged patch pauses for exact security review; only its approved hash may
resume. This command is documented, not authorization for another model search.
All new packets include checksum-protected raw evidence archives. Replaying them
is distinct from fresh execution and independent human reproduction.

[VMAX Applied RL responsibilities](https://job-boards.greenhouse.io/vmax/jobs/4056502009):

| Capability | Demonstrated evidence / limit |
|---|---|
| RL environment engineering | Reused executable jobs, tools, state and source-based outcomes; task difficulty remains insufficient. |
| Rollouts | Real native-tool traces, bounded parsing, patch staging/review, resume and failure accounting; autonomous relaunch remains unproven. |
| Rewards/verifiers | Completing-invalid model candidate rejected despite valid hashes/history; positive and exploit controls. |
| Experimental methodology | Frozen versions, raw traces, fixed baselines, independent numerical checks, explicit unavailable groups. |
| Training/optimization | Existing MLX/LoRA and logprob work inspected; no new training or acceleration claim. |
| Learning-dynamics debugging | Separated infrastructure, tool serialization, editor, policy and reward/task failures; no RL improvement inferred. |
| Efficiency | Bounded reuse, measured prefill/decode/tool costs, no larger downloads or repeated task generation. |

**Single highest-value next action:** execute the prepared Harbor oracle/nop/known-
invalid compatibility controls once the isolated install/backend is approved.
Keep TrainDoctor as a reproducible engineering artifact; do not expand its learning
program on these findings. AI critiques are not independent human validation.

Verification: 39 focused research tests and 18 Harbor adapter unit tests passed; XML output is saved in results/agent_v2_verification_20261030. Native Compose configuration and source/shell syntax checks passed. Harbor CLI lint and actual Docker trials remain unrun.

## Dated addendum — 2026-10-30 Harbor validation

The separately approved interoperability milestone is now executed: correct1,
no-op0, completing-invalid0 and tamper0 through native local Harbor/Docker, with
matching native TrainDoctor controls and 20 adapter tests. See
[the validation report](HARBOR_VALIDATION.md) for exact source/image identities,
runtime defects, narrow lint exception and measured costs. This supersedes only
the earlier pending-approval/runtime status; it changes no model trajectory or
scientific learning conclusion above. No push or publication occurred.
