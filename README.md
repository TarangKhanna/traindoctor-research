# TrainDoctor Research

**Does a repaired training job preserve the intended computation—or merely finish?**

This repository contains TrainDoctor’s experiments, analysis and raw execution evidence. The [environment repository](https://github.com/TarangKhanna/traindoctor-env) supplies the distributed runtime, investigation tools and verifier.

The strongest current result is an executable recovery demonstration: checksum-valid checkpoints can resume successfully while silently changing sample consumption or optimizer state. We test competing recoveries, check their actual trajectories and preserve the evidence for reproduction.

## The worked recovery experiment

A small two-rank PyTorch/Gloo job is preempted. The training code stays unchanged. Recovery artifacts mix components from different genuine checkpoints:

| Delivered checkpoint | Sampler-only restoration | Optimizer-only restoration | Full restoration |
| --- | --- | --- | --- |
| Current model/momentum, stale sampler | Correct | Incorrect | Correct |
| Current model/sampler, stale momentum | Incorrect | Correct | Correct |

Every tested intervention completes, including the incorrect ones. Manifest checksums are recomputed for the delivered artifacts, so byte integrity does not establish component consistency. Semantic checks distinguish them using trusted committed state, delivered samples and training updates.

The experiment includes older-checkpoint and scratch recovery. Both remain legitimate fallbacks; repeated computation is measured separately. Full restoration is optimal in both states, so this pair demonstrates recovery verification rather than a difficult conditional-choice benchmark.

An independently implemented NumPy continuation oracle agrees with the executed trajectories. Its inputs include genuine committed anchors and the same data-service specification; this is mathematical cross-checking, not independent human validation.

[Worked report and action outcomes](experiments/state_recovery_v1/REPORT.md) · [Recovery contract](https://github.com/TarangKhanna/traindoctor-env/blob/main/docs/STATE_RECOVERY_V1.md)

## Reproduce the saved local example

Use Python 3.12. Clone both repositories as siblings, install the environment source and recount the archived executions. Use a new, empty extraction directory.

```bash
git clone https://github.com/TarangKhanna/traindoctor-env
git clone https://github.com/TarangKhanna/traindoctor-research
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e './traindoctor-env[dev]'
cd traindoctor-research
export TD_KEY=state-recovery-demo-key PYTHONPATH=.
python -m zipfile -e results/state_recovery_v1_loopback_verified/raw_evidence.zip /tmp/traindoctor-evidence
python -m experiments.state_recovery_v1.recount --run /tmp/traindoctor-evidence
```

Expected output:

```text
PASS: 12 raw continuation recounts; 60 capture hashes; common optimal actions: ['restore_full']
```

This checks saved execution evidence; it launches no training or model inference and needs no cloud account. For a fresh distributed run, follow the [full reproduction commands](experiments/state_recovery_v1/REPORT.md). The local execution backend is macOS `sandbox-exec`, with loopback permissions. Historical evaluations used Modal/gVisor; those isolation boundaries are different.

## Harbor and local coding agents

The existing checkpoint code-repair task now runs through Harbor on local Docker. The correct patch passes; no-op, completing-but-incorrect and forged-evidence controls fail. All four verdicts agree with native TrainDoctor execution. [Validation report, raw receipts and costs](docs/HARBOR_VALIDATION.md) · [Task and setup](https://github.com/TarangKhanna/traindoctor-env/blob/main/docs/HARBOR.md).

Actual local Qwen/Gemma trajectories include inspection tools and code-edit attempts. They have not demonstrated an autonomous repair/relaunch/result-inspection loop. The [agent feasibility report](docs/AGENT_V2_REPORT.md) identifies the failures and task shortcuts; these cases do not justify further RL training.

The subsequent [masked-loss coding-agent study](evidence/masked_mean_v1/REPORT.md) adds an executable source-debugging task, hidden semantic regressions and a working multi-turn harness. Its [sanitized model traces and captured states](evidence/masked_mean_v1/) support an inference-free numerical recount. It demonstrates the infrastructure to attempt autonomous repair; neither recorded model episode completed that repair loop. Protocol fixes were tested afterward without model reruns. This study is separate from the historical menu-based SFT/GRPO experiment below.

## Can verification work without an uninterrupted reference run?

The [reference-free audit](docs/REFERENCE_FREE_REPORT.md) checks transitions from a trusted earlier checkpoint using recorded inputs and the specified optimizer rule. It reproduces the recovery decisions without reading an uninterrupted reference trajectory.

The guarantee is conditional: it verifies consistency with the anchor and update specification. A poisoned anchor can support a consistently wrong trajectory. The audit includes that counterexample, missing-history checks and a CPU numerical repeatability test. It also measures replay cost and reports the additional trusted records required.

Our [prior-art review](docs/REFERENCE_FREE_PRIOR_ART.md) found that checkpoint-transition replay is established, including Proof-of-Learning and existing training-system checks. We make no new verification-method claim. On this pair, component comparisons already detect the tested inconsistencies; replay adds no detection coverage. The useful artifact is the inspectable execution, checks and documented failure boundaries.

## Results (historical v0.1)

Qwen3-4B was adapted with LoRA SFT, then GRPO, and scored on a held-out evaluation set of 100 incidents from the same generator. Training used cached recovery outcomes; evaluation used live sandbox replay. These were single-turn diagnosis/evidence/repair-menu tasks, not interactive code repair.

| Policy | Full-task success |
| --- | ---: |
| Base model | 5.0% |
| SFT | 17.2% |
| Continued SFT | 29.7% |
| GRPO, diagnosis-only reward | 23.0% |
| GRPO, recovery-aware reward | 33.4% |

The recovery-aware arm gained 10.36 percentage points over diagnosis-only reward (95% bootstrap interval +5.83 to +15.26; six paired training seeds). Repair-choice rates were similar: 71.4% versus 71.5%. Without an evidence-only reward arm, the gain cannot be attributed to better recovery selection.

Continued SFT was not distinguishable from recovery-aware GRPO (+3.7 points; interval −1.7 to +9.1); compute comparability was not established. The base-to-SFT difference also includes a large format-validity improvement. Seed splits do not test unseen failure mechanisms.

**Conclusion: these experiments do not establish that RL improves repair selection or beats continued SFT.** The [detailed results](docs/PROJECT_DETAIL.md#results-historical-v01), [evidence map](docs/EVIDENCE_MAP.md) and [protocol erratum](docs/ERRATUM_mmd.md) retain uncertainty, shortcuts and unresolved preregistration contradictions.

## What the failed pilots taught us

The [menu v3 analysis](experiments/menu_v3_dev/REPORT.md) and [code-repair pilot](docs/PROJECT_DETAIL.md) narrowed the learning problem: failure to finish differs from completing with wrong semantics, and fixed bad-repair identities can supply shortcuts. Adding more menu options does not establish state-dependent reasoning.

Tier B supplies interactive inspection and source edits. Further SFT/RL comparisons remain conditional on a task that needs investigation, informative rewards and matched budgets; the current feasibility study recommends stopping training on these cases.

## Navigate the evidence

| Artifact | What to inspect |
| --- | --- |
| [State-recovery report](experiments/state_recovery_v1/REPORT.md) | Executed interventions, NumPy checks and raw receipts |
| [Reference-free report](docs/REFERENCE_FREE_REPORT.md) | Detection matrix, trust assumptions, counterexamples and timing |
| [Evidence directory](evidence/README.md) | Historical evaluation outputs and checksums |
| [Detailed project record](docs/PROJECT_DETAIL.md) | Full historical analysis, model setup, audits and limitations |
| [Reproduction guide](docs/RERUN_EXAMPLE.md) | Historical recount and rerun commands |
| [Current status](docs/STATUS.md) | Completed work and next steps |

Saved evidence, local reruns and independent human reproduction are distinct; human reproduction remains outstanding. The demonstrated workloads are small CPU/Gloo MLP jobs, with no established production NCCL/FSDP coverage or generalization to arbitrary training programs.

Apache-2.0: [LICENSE](LICENSE), [NOTICE](NOTICE). [Citation](CITATION.cff) · [Security](SECURITY.md).
