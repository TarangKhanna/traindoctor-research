# TrainDoctor learning-signal audit

source: executed (historical receipts re-scored and two new source-only Qwen requests on 2026-11-02; no training).

**Finding:** execution-verified reward had little additional repair-selection signal
in the saved original-menu development groups. This is evidence of a task/reward
limitation, not proof that GRPO cannot learn repairs or a causal explanation of the
historical training result. A V1 ablation is not justified by these cases.

## What the 2.3% means

The [portable recount](results/learning_signal_20261102_verified/REPORT.md) reproduces
70 V1-positive/V2-negative responses among 3,072 identical completions. It reuses
the original cached episodes and scorer, including its strict format/safety gates.
The export contains grading fixtures, raw completion text, outcome tables and
per-group vectors; original input hashes are recorded. These are **eight saved
DEV evaluation samples per prompt**, from SFT, continued SFT and the first three
paired GRPO seeds. Sealed TEST evaluation used G=4; these older exploratory DEV files explicitly store `g=8`. Training-time sampled completions, token masks, logprobs and
optimizer states were not recovered. This is an approximation to training's
opportunity for reward discrimination, not a reconstruction of its gradients.

| Reward | Positive completions | Zero-variance groups | Fraction |
|---|---:|---:|---:|
| V0: diagnosis | 1,973 | 163 / 384 | 42.45% |
| V1: diagnosis + evidence | 767 | 174 / 384 | 45.31% |
| V2: V1 + repair/safety | 697 | 179 / 384 | 46.61% |

V1 and V2 change both centered and normalized advantages in **20/384 groups
(5.21%)**; the other **364 groups are identical**. Groups stay within the actual
prompt and policy; no adjacent prompts or policies are pooled. Conventional group
normalization `(r - mean)/(sample_std + 1e-4)` and population-std sensitivity give
the same changed-group count. No completion reaches the 400-token cap; no
truncation-based exclusion is needed. Exact historical EOS/padding masks and the
pinned trainer's runtime tensors remain unavailable. Historical configuration
uses TRL 1.16.0, eight generations, DAPO and beta=0; normalized values are formula
reconstructions, not an authenticated replay of that trainer.

All 70 disagreements are **non-completing repairs**, not completing-invalid
semantic traps: A1 wrong-shard 30, wrong-sync 30, wrong-checkpoint 6, no-op 2;
A2 wrong-checkpoint 1; B_SKIP wrong-per-rank edit 1. Thus 68 occur on A1. Conditional
on V1 credit, repair acceptance is **697/767 (90.87%)**. V1-positive failure is
concentrated rather than absent; unconditional disagreement alone hid this fact.
Each original menu has one recovering candidate. Private role names explain the
re-score but are not model-visible features or a deployable policy. This audit
does not establish that the model used those identities or superficial cues.

V2 here transiently marks cached `dev_container` outcomes sandboxed via the
historical `dev_episode` path. That reproduces exploratory training reward; it is
not fresh trusted execution. Held-out headline evaluation instead used sandbox
replays. No data in the older worktrees was overwritten.

## Task designs already attempted

| Design / intended capability | Remaining shortcut or limit | Verifier / real-agent evidence / learning signal |
|---|---|---|
| Menu-v1: diagnosis, evidence, repair selection | One recovering choice; repeatable repair templates | Strong agents saturated small development cases. Original reward gaps and grouped analysis above; tests alone say nothing about competence. |
| Menu-v2: select among plausible repairs | Extra timeout/scratch/alignment edits often valid on Task A | Historical re-score: about 2.6% disagreement; not live verification. Original raw `v2` fields are zero because DEV was unsandboxed; treating them as operative reward would be wrong. No retraining. [Finding](docs/FINDINGS.md). |
| Menu-v3: reject completing-but-wrong alternatives | Four fixed never-valid items explain most signal; strong-agent saturation | 720 executed candidates: v1.2 accepts 175 valid, rejects 271 completing-invalid and 274 non-completing. Labels derive from verifier fields, not independent truth. Codex V1/V2 gap 0/120; trained-checkpoint gaps 2.9–7.1%. [Report](experiments/menu_v3_dev/REPORT.md). |
| A2v2: reason about a real producer stall | Producer bypass remains semantically valid under the actual data semantics | A bypass must not be mislabeled a cheat. The historical executable audit distinguishes outcomes, but no useful same-policy reward groups or successful autonomous learning were established. [Executed S02 audit](docs/A2V2_STATUS.md). |
| State-recovery twins: choose sampler vs optimizer restoration | Full committed restoration works in both states; scratch remains valid | Twelve real continuations agree with NumPy; selective actions differ. Scripted controls prove recovery verification, not autonomous choice or RL difficulty. [Report](experiments/state_recovery_v1/REPORT.md). |
| Tier B checkpoint format: durable source migration | A fixed source patch handles current/legacy formats; full restoration handles state twins | Correct and completing-invalid controls distinguished. Qwen applied an edit but no agent relaunch in agent-v2; supervised original failed Python, normalized diagnostic completed but failed semantics. Retest is unscored resource interruption. [Report](docs/AGENT_V2_REPORT.md). |
| Tier B masked loss: edit, relaunch, use semantic feedback | One global-normalization patch solves every hidden schedule; source-only reasoning suffices | Correct loss/gradient patches pass; unchanged and visible-hardcoded controls complete and fail semantic cases. Later Qwen edits/relaunches fail Python; Gemma edits without relaunch. No autonomous semantic repair or sampled reward groups. [Recount](docs/CODING_AGENT_RECOUNT.md). |

## Capability floor: masked loss

Two minimal source-only questions now separate reasoning from the multi-turn
controller. The [new packet](results/capability_floor_20261102/) preserves full
prompts, raw outputs, token IDs, frozen source, configuration and runner status.
The same cached Qwen Instruct snapshot used in the coding studies ran two fresh
requests, greedy/seed 0, thinking off, 1024 output tokens, 8192 context. This is a
new condition, not a matched repeat of temperature-0.2 episodes. Both stopped
normally, neither reached its cap, and sampled swap growth was zero. Generated
code was text only and never executed. No tools or bug-location hint were supplied.

| Floor | Evidence | Diagnosis |
|---|---|---|
| Identify bug and correct formula from source | New math response identifies unequal local/global weighting, then presents another rank-local denominator and incorrectly recommends replacing gradient synchronization | Qualitative recognition, **correct formula not demonstrated**; reasoning failure rather than output exhaustion |
| Produce a valid repair without tools | New patch response supplies no unified-diff headers/hunks and adds a zero-count branch without removing the original loss line or introducing global normalization | **Incorrect/inapplicable patch**; 107-token normal stop, no context or infrastructure interruption |
| Apply the correct repair with bug location given | Historical Qwen/Gemma applied native edits; no frozen location-given correct-repair probe exists | Mechanical editing demonstrated, **guided correct repair unknown**. Stop after the failed source-only prerequisite rather than expand model searches |
| Independently diagnose, edit, relaunch and interpret feedback | Later Qwen episodes apply edits, request failed execution and act afterward; 512 claims fixed despite IndentationError; 1024 acknowledges failure but reruns unchanged code after a rejected stale-hash edit | Partial loop competence; **successful repair and reliable feedback interpretation not demonstrated** |

Earlier Qwen proposals also omitted the world-size compensation for gradient
averaging and called `.clamp_min` on an integer. That reasoning/patch defect is
separate from 512's indentation error, native-call truncation, rejected stale-hash
edits, Gemma's context exhaustion and the checkpoint retest's unscored RAM guard.
Those attempts are **not controlled G=4/G=8 groups**. No Tier B zero-variance or
zero-advantage rate is estimated from them.

An isolated strong coding-agent source-only baseline, given only the same source
fixture, derives `world * total / global_valid_scalar_count` and proposes a
configuration-count patch. Its [analytic proposal](results/capability_floor_20261102/strong_source_only.json)
uses information already in the source/configuration and no runtime inspection.
It is not executed agent success or a controlled comparison with Qwen. Together
with the already executed global-count controls, it confirms that the task is
straightforward for a competent source-level solver, while these Qwen responses
are below the necessary formula/patch floor.

**Too easy, too hard, or wrong signal?** Original Tier A is too easy in repair
selection conditional on evidence, and its reward rarely changes group advantages.
Tier B's single masked-loss mechanism is straightforward for a strong source-only
solver yet too hard for this small policy's demonstrated reliable patch competence.
The semantic oracle rejects real errors correctly; there is no evidence that its
correctness target is wrong. The learning signal is poorly informative for the
intended repair-reasoning claim: Tier A mostly separates evidence compliance,
while Tier B terminal zeros currently combine reasoning, patch and interaction
failures. This is a diagnosis of the observed evidence, not a causal experiment
showing that task design prevented GRPO from learning.

## Strong simple policies and decision

The [recomputed menu-v3 tables](experiments/menu_v3_dev/results/dev1/analysis_verified/TABLES.md) preserve
existing baselines: canonical checkpoint repair with majority checkpoint diagnosis,
no inspection and perfect evidence when the label is right yields 40.0% V2;
config-only yields 48.3%, status-only 38.3%. These evidence-granted baselines are
upper bounds, not complete autonomous policies. A leave-one-seed-out fixed avoid-list
reduces a perfect-V1 uniform-selection reference's disagreement from 30.1% to 2.5%.
This is an analytical reference policy, not learned improvement. It explains
6/7, 9/14 and 14/17 V1-positive/V2-negative choices of the three tested checkpoints.
Strong-agent successes and these shortcuts jointly undermine a repair-reasoning
training claim; they do not imply that the executable verifier is unnecessary.

For masked loss, the fixed `all_reduce(count)` / `loss = total * world / count`
repair is mathematically correct for every declared positive-total schedule: averaging rank gradients cancels its world-size factor, leaving global summed gradients divided by global valid count.
Existing executed controls validate it on visible and hidden schedules, including
a zero-valid rank. Tools expose useful source/count/log information and Qwen used
execution feedback, but their necessity or benefit is unmeasured. That universal
repair fails the proposed task-discrimination gate. Keep the task as an engineering
demonstration; after the two source-only diagnostics, run neither more model
feasibility episodes nor V1 GRPO here.

Historical SFT/GRPO: 5.0% base, 17.2% SFT, 23.0% V0 and 33.4% V2; repair selection
71.4%/71.5%. Continued SFT is 29.7%, one run without proper compute matching.
Improved evidence accompanies the V2 score gain; the missing V1 arm prevents causal
attribution. Whether a structurally varied, shortcut-resistant task supplies useful
repair gradients, and whether RL beats compute-matched SFT on it, remain unresolved.
No new task distribution or framework was built to obtain a positive result.

## Verification and provenance

`audit_learning_signal.py --check` re-scores exported grading fixtures and recounts
all group vectors; `recount_coding_agents.py --check` validates the later model
rows, identities and derivative hashes. Existing portable control, menu accounting,
README numerical and focused unit checks are recorded in
[the verification receipt](results/overnight_verification_20261102/verification.json).
Local bases were environment `2569a2a` and research `f9933a2`, beyond the older
release worktrees. Supplied `46eb031`/`a8b26a2` objects were absent locally; remote
heads were not fetched or verified. No push or publication occurred. Additional
Modal spend: $0; no model process was started. Process enumeration was denied by
the sandbox, so existing heavy-job status was unknown and heavy work was avoided.
