# TrainDoctor: reference-free recovery verification

2026-10-29. **Local feasibility GO; replay-method novelty NO-GO.** The checker can
detect the saved checksum-valid invalid recoveries without an uninterrupted run.
It establishes conditional transition consistency, not correctness of an entire
training history. No differentiated ML systems paper contribution is established.

## Provenance and prior-art gate

Research worktree: `.wt/reference-free-v1-research`, branch
`audit/reference-free-v1`, based on `0419515`. Environment source is unchanged at
`438b573` in `.wt/state-recovery-v1-env` (launcher hardening `65e4bea`). Earlier
`demo/state-recovery-v1` and `audit/state-twins-20261029` worktrees were inspected
and preserved. The public heads were verified in the earlier audit; they were not
re-queried in this session. No main worktree was changed.

Frozen prototype, tests and raw receipts: research commit
`aea970db30f8e43b63285f9e03ee644ccaf4333a`. This report/status closure is a separate
documentation commit; the tested source hashes are stored with the receipts.

The [prior-art matrix and one-page boundary](REFERENCE_FREE_PRIOR_ART.md) were
completed before experimental implementation. They distinguish documentation,
inspected source/tests, papers and contributor-reported failures. Nineteen public
source files were inspected without execution; exact revisions, URLs and content
hashes are in [the source manifest](../results/reference_free_v1_prior_art_20261029/source_manifest.json).

| Prior art | Established capability relevant here |
|---|---|
| [PyTorch DCP](https://docs.pytorch.org/docs/main/distributed/checkpoint.html) | Parallel checkpointing/resharding and model/optimizer structural consistency; caller-supplied state remains important. |
| [TorchTitan checkpoint test](https://github.com/pytorch/torchtitan/blob/2db55136735f35ea77d629115890c7fa40decb02/scripts/checkpoint_compat_test.py) | Reference-versus-resumed loss regression checking. |
| [TorchTitan SDC replay](https://github.com/pytorch/torchtitan/blob/2db55136735f35ea77d629115890c7fa40decb02/torchtitan/observability/sdc_replayer.py) | Optional in-training deterministic forward/backward replay checks. |
| [StatefulDataLoader](https://github.com/meta-pytorch/data/blob/ee79a399d3a53676544e0d3f0e826ffa58d1fec9/torchdata/stateful_dataloader/README.md) | Mid-epoch sampler/dataset/worker state restoration; not automatic coordination with optimizer state. |
| [DeepSpeed tests](https://github.com/deepspeedai/DeepSpeed/blob/12ffb990341f021ea9847d86a98ba91c81f77fdb/tests/unit/checkpoint/common.py) | Tensor/optimizer/counter save-load checks, beyond checksums alone. |
| [Megatron reruns](https://github.com/NVIDIA/Megatron-LM/blob/5de526f232464e1b1ff1d6645fe2c4aaf01bc0d8/megatron/core/rerun_state_machine.py) | Runtime result replay and checkpoint-based diagnostic restarts. |
| [OLMo-core loader](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/olmo_core/data/data_loader.py) | Dataset fingerprint/seed checks and consumed-token tracking. |
| [Proof-of-Learning](https://arxiv.org/html/2103.05633), [author verifier](https://github.com/cleverhans-lab/Proof-of-Learning/blob/main/verify.py) | **Already replays logged training inputs between checkpoints and compares endpoints without an uninterrupted run.** |

Published source and tests do not establish what every production team enables.
Open OLMo-core reports [#929](https://github.com/allenai/OLMo-core/issues/929)
(cursor rounding repeats data after a batch-size change) and
[#908](https://github.com/allenai/OLMo-core/issues/908) (step-0 optimizer mutation)
provide concrete reported failures. Their reproducers were read, not executed here;
they are not treated as independently confirmed production incidents.

## Contract and trust boundary

[Specification](REFERENCE_FREE_SPEC.md) was reviewed before implementation.
**Method A** compares continuation-run captures at overlapping steps 4/8 with
genuine archived committed state, separately examining model, momentum, sampler
and step. It does not infer a momentum tensor's historical step from its values.
No original checkpoint 12 exists; its expectations belong only to Method B.

**Method B** trusts checkpoint 4, checks 4→8, then computes expected 8→12 with
the existing NumPy manual gradients/SGD implementation. Trusted served history is
checked against intended ordered samples; records are not accepted merely because
replaying their own contents is self-consistent. Applicable continuation captures
and their hashes/manifest are checked. Neither checker reads a fault-free run,
receipt, old oracle flag or historical verifier boolean.

| Claim | What this assay can establish |
|---|---|
| Byte integrity | Captured bytes match ledger hashes and current files match manifests, conditional on trusted records. |
| Component consistency | Observed step-4/8 components match the authoritative captured state. No intrinsic tensor-origin proof. |
| Transition consistency | Observed endpoints agree with deterministic replay from trusted anchors and the specified inputs/update rule. |
| Intended algorithm correctness | The MLP/SGD specification and input recipe are assumed; the NumPy calculation is independent of PyTorch autograd. It does not prove the specification is the right scientific algorithm. |
| Uninterrupted-run equivalence | Not established from initialization. Scratch is checked at checkpoint 4 and afterward; its initialization→4 segment is not replayed. Endpoint agreement does not attest every unobserved historical operation. |

The fixed public demo key plus versioned recipe regenerates input values; the
ledger does not independently commit batch-value bytes or salt. Source identity
compares two captured source trees, not an independently authenticated pristine
source commitment. Post-load observed checkpoints and pristine-source/single-writer
assumptions are used; per-rank load-time byte attestation is absent. In particular,
legacy `resume_sha256` names an earlier archived capture by ID, and remains the
clean hash for mixed delivered checkpoints. It is excluded as loaded-byte proof.
No historical behavior was changed to repair that evidence gap.

The job has fixed LR and no stochastic model operations after initialization:
scheduler and resumed model RNG verification are **not applicable**, not passed
tests. Malformed/missing ledger schemas may raise rather than yield an explanatory
verdict. This is a fixed-artifact assay, not a general robust/security validator.

## Detection, numerical results and cost

The original archive SHA256 is
`0579811fa306765a6a5a5b1c279efaca1acdbf4e123235dbd336af4f80bc1671`.
Only failed-prefix and continuation inputs were extracted; reference trees,
observations and correctness receipts were excluded. The original twelve jobs
were **not rerun**. [Full detection matrix](../results/reference_free_v1_20261029_reviewed/TABLE.md)
and [raw diagnostics](../results/reference_free_v1_20261029_reviewed/matrix.json):

| Recovery | Sampler mixture | Optimizer mixture |
|---|---|---|
| Resume delivered checkpoint | reject: component, transition, input history | reject: component, transition |
| Restore sampler only | accept | reject: component, transition |
| Restore optimizer only | reject: component, transition, input history | accept |
| Full committed restore | accept | accept |
| Earlier coherent checkpoint | accept | accept |
| Scratch | accept, initial segment unverified | accept, initial segment unverified |

All twelve executions completed; all checked hashes validate. Eight acceptances
and four rejections agree with the existing NumPy continuation oracle. That
agreement reuses the same update kernel; it is not a second independent mathematical
implementation. The independent calculation boundary is executed PyTorch versus
manual NumPy. There are no false decisions against these existing labels; the
paired/repeated observations do not support an independent-trial accuracy estimate.
Method A already detects all four invalid examples, so Method B adds **no detection
coverage on this pair**. Keeping a genuine full step-8 archive already makes full
restoration and component comparison possible.

Exact stored historical accepted errors: final parameters
`3.73132204023463e-08`; final momentum `5.2193850685888776e-08`;
original 4→8 transition `7.561030429537041e-08`. The latter is the largest accepted
replay discrepancy. Acceptance keeps the existing absolute `1e-5` threshold, with
exact step/sampler equality and finite/schema checks. No new threshold was fitted.
Each interval restarts from its recorded anchor; this does not establish a single
accumulated-error bound for replay from initialization or even a chained 4→12 run.

Across the full checked intervals, invalid sampler cases have maximum tensor
error `0.2348765521049382` and discrete sampler-coordinate difference `8`;
invalid optimizer cases have maximum tensor error `0.7232248443929389`.
These include observed start/overlap states; they are not the old final-only errors.
Relative diagnostics divide each coordinate error by `max(abs(expected),1e-12)`:
accepted maximum `2.1859106281341663e-05`; invalid optimizer maximum
`227.82069745494792`; sampler-coordinate ratios reach `8e12` near an expected zero.
Normalized infinity errors are also saved. Relative values do not change acceptance
and are not comparable across unlike tensor and discrete-state units.

Measured per-outcome checker wall time: minimum `0.014452167029958218` s,
median `0.017933166498551145` s, maximum `0.026139499968849123` s; twelve total
`0.21627999999327585` s. Includes local file/hash checks and replay, excludes imports
and archive extraction, uses warm/uncontrolled filesystem caches and a fixed order.
This tiny MLP timing does not establish affordable large-model verification.

Two **new**, sandboxed CPU/Gloo continuations of steps 8→12 completed in
`4.055986875027884` and `3.6939087079954334` seconds including process launch.
Both were bitwise identical to the existing correct continuation and had NumPy
maximum discrepancy `5.2193850685888776e-08`. Live parent TCPStore listeners were
loopback. [Repeatability receipt](../results/reference_free_v1_repeat_20261029/repeatability.json)
and its archived captures are preserved. This tests local repeatability, not
NCCL/FSDP/GPU arithmetic, scheduler restoration or nondeterministic optimizers.

## Bounded falsification

[Constructed counterexample receipts](../results/reference_free_v1_20261029_reviewed/falsifications.json)
are labeled simulated inputs with an executed checker, not new training incidents.

1. Perturb checkpoint-4 weights, derive coherent checkpoints 8/12 with the specified
   rule. Both relative transition checks accept although final state differs from
   the original trajectory by `0.09537724040869114`. This is one actual false
   acceptance **of the stronger original-trajectory claim**, not of the conditional
   transition contract. Comparing with the pinned original anchor rejects it.
   Consistently incorrect algorithms/data can likewise be self-consistent; they
   were considered but not separately executed.
2. Remove required rank-0 window 8 without changing checkpoint bytes: insufficient
   evidence. Substitute one sample ID: input-history rejection. Top-level status
   tests confirm both behaviors. Falsifying the entire trusted capture/input authority
   remains outside the contract. A correct but more expensive earlier/scratch
   recovery still accepts; cost is not semantic corruption.

No empirical security guarantee or population false-positive/negative rate follows
from these author-constructed controls. Numerically different valid GPU updates
could be rejected by this CPU tolerance; this platform-bound risk remains untested.

## Adoption, paper and conditional model-research path

A real job would need an authenticated earlier anchor, coherent model/optimizer/
scheduler/RNG/dataloader states, versioned code/config/data, ordered input or
reconstructible transformation records, rank/topology/precision information and a
trusted commitment tying components to a checkpoint transaction. IDs alone cannot
reconstruct stochastic augmentations or mutable remote data. Existing DCP and
StatefulDataLoader should supply state where possible; TrainDoctor should not
replace their checkpoint infrastructure. An independent replay kernel for each
optimizer/model/backend can be costly to implement and maintain. Replaying every
interval can approach duplicating training compute; the MLP timing says otherwise
only for this very small job. Production utility and acceptable overhead are unknown.

The strongest current direction is a reproducible resume-correctness case study
and negative research result. Original technical contribution: modest integration
and explicit trust boundaries. Scientific novelty: not established. Rigor: local
execution plus an independent arithmetic implementation, with narrow attacks.
Generalizability: unverified. Engineering utility: plausible regression fixture,
not demonstrated team benefit. Reproducibility: local saved-artifact and execution
checks; external human reproduction absent. Fit: research-engineering artifact;
a full ML systems paper needs a differentiated real-workload finding and useful
cost/coverage result, not another synthetic task.

Retain multi-turn executable repair as a **design-only** extension. First use a real
resume regression to test whether inspection changes justified decisions, code
edits execute, actual trajectories are graded, and fixed-identity/full-restore/
scratch/state-rule baselines explain performance. The present pair fails that gate.
If a distinct problem passes, freeze disjoint bug/mechanism evaluation groups and
measure base-model format, inspection behavior, code completion/semantics, reward
variation in dependent G=4/G=8 groups, local latency/memory and oracle cost before
training. No mechanism-transfer claim from seed splits.

Only then compare base, quantized-LoRA SFT, continued SFT and optionally preference
optimization/GRPO. Continued SFT and GRPO must start from the same SFT adapter with
matched data access, verifier, parameterization and held-out evaluation. Report
tokens, examples, optimizer steps, wall time and memory separately; equal steps
are not equal compute. Retain robust repairs and charge measured costs separately.
Use existing MLX/MLX-LM/mlxrl work; no implementation or promise of RL improvement.

## Verification and reproduction

Changed implementation: three `experiments/state_recovery_v1/reference_free*.py`
modules and `tests/test_reference_free.py`. Documents: this report, prior-art gate,
specification, scoped AGENTS, learning/status updates and ignore rules. New generated
receipts are confined to `results/reference_free_v1_*`. Environment/v1.2 and all old
receipts remain unchanged. Preliminary new assay output is retained separately.

Baseline: 18 focused research tests passed. Final focused research gate: 28 passed.
The earlier environment claim was rerun: **241 passed, 1 deselected**. The manual
smoke path was filtered saved captures → reference-free decision matrix → old
oracle comparison, followed by two real sandboxed continuations and live loopback
inspection. Native read-only critic reviewed specification and implementation;
this is AI critique, not independent human recognition. Linux/external reproduction
and new dependency installation were not performed. No experimental model
inference/training ran; AI research/review assistance is identified separately.

The [terminal test output](../results/reference_free_v1_20261029_reviewed/tests.txt),
[tested source hashes](../results/reference_free_v1_20261029_reviewed/verification.json),
[initial static review](../results/reference_free_v1_20261029_reviewed/review_initial.txt)
and [follow-up review](../results/reference_free_v1_20261029_reviewed/review_followup.txt)
are retained. Review caught missing scratch overlap evidence, empty/missing ledger
classification and suppression of primary outputs on historical disagreement.
These were fixed with regressions; the follow-up found no remaining blocker within
the frozen-artifact contract. It did not execute the tests. The repeatability archive
checksum is `4276bb189e367e5ca017f51b749214a1cb4153d9b357ba8bd7c68f7f1c168fe4`.

From this research worktree, with the existing prepared Python and environment
source (replace paths as needed):

```bash
export TD_KEY=state-recovery-demo-key
export PYTHONPATH=/path/to/state-recovery-v1-env:.
export UV_CACHE_DIR=/tmp/td-audit-uv
uv run --no-project --offline /path/to/prepared/python \
  -m experiments.state_recovery_v1.reference_free_audit \
  --archive results/state_recovery_v1_loopback_verified/raw_evidence.zip \
  --baseline results/state_recovery_v1_loopback_verified/recount.json \
  --out /tmp/a-new-reference-free-audit
TD_REFERENCE_FREE_INPUT=/tmp/a-new-reference-free-audit/inputs \
  uv run --no-project --offline /path/to/prepared/python \
  -m pytest -q tests/test_reference_free.py tests/test_state_recovery_oracle.py tests/test_menu_v3_analysis.py
caffeinate -s uv run --no-project --offline /path/to/prepared/python \
  -m experiments.state_recovery_v1.reference_free_repeat \
  --inputs /tmp/a-new-reference-free-audit/inputs --out /tmp/a-new-repeatability-run
```

The replay/math tests need existing CPU PyTorch/NumPy and matching environment
source, not Metal or cached LLMs. The last command requires macOS sandbox-exec and
permission to launch local sandboxed workers. Output paths must not exist.

## Decision and single next action

| Decision | Verdict |
|---|---|
| Conditional reference-free checker | GO as a bounded engineering validation |
| Novel verifier contribution | NO-GO: equivalent interval replay exists |
| Full research paper now | NO-GO: differentiation, scale and external value unproved |
| External release now | NO-GO pending human reproduction/review and owner approval |
| Local RL training | NO-GO: common optimal repair, no demonstrated investigation benefit |

Choose **B: independent human reproduction**, not an integration PR or manuscript
yet. Potential scientific reviewers include Proof-of-Learning authors Hengrui Jia
or Nicolas Papernot for method overlap/trust assumptions; PyTorch DCP/TorchTitan
checkpoint contributors are an upstream channel only after a concrete unmet
regression is identified. These are potential channels, not confirmed availability,
endorsement or outreach. No messages/issues/PRs were sent. Any eventual O-1A use
should rest on documented external adoption/recognition, not AI review or star counts.

**Strongest argument for no novelty:** this is a custom-instrumented MLP using a
known replay method, and its component archive already detects every tested fault.
**Evidence:** source inspection of Proof-of-Learning supports that objection;
the twelve-row assay supplies no detection advantage that defeats it.
**Single highest-information next action:** have an independent checkpoint/resume
engineer reproduce this exact packet and identify one real failure whose diagnosis
their current framework/tests cannot already supply. A negative assessment ends
the paper claim; do not respond by generating more tasks.

Publication status update, 2026-10-29: the owner subsequently authorized sharing
this bounded engineering artifact and its reproduction evidence. That supersedes
the publication hold above, not the novelty/RL conclusions. Independent human
reproduction remains outstanding; this release is not a validated benchmark or
evidence of external recognition.
