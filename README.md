# TrainDoctor research artifacts

**Question.** Does training a small model against a reward that checks whether a repaired training job actually recovers teach it to debug distributed training better than a reward that only checks the diagnosis label?

TrainDoctor is an environment where agents diagnose and repair failed multi-process PyTorch (Gloo) training jobs: four executed fault kinds in two synthetic families (A1 mismatched collectives, A2 a rank stalled before communication, B_SKIP and B_DUP checkpoint and sampler state that disagree after a worker failure) plus a healthy control (B_OK). A verifier re-executes each repair in a sandbox. The environment, verifier and quickstart live in the library, [traindoctor-env](https://github.com/TarangKhanna/traindoctor-env) ([task suite](https://github.com/TarangKhanna/traindoctor-env/blob/main/docs/TASKS.md)); this repository holds the training and evaluation code, sealed-test evidence, experiments and analysis scripts.

**Findings (exploratory, partly negative).**
- **SFT and GRPO.** On a sealed test of 100 incidents, GRPO with the recovery-aware reward outscored the label-only reward (33.4% vs 23.0%). The gain came from evidence grounding. Repair selection did not improve, and continued SFT with similar compute is not distinguishable from the recovery-aware arm.
- **Verifier audit.** None of 86 scripted fake repairs was accepted, but 38 of them only crashed, which shows they do not run, not that an invariant covers them. I wrote the attacks; there was no external review.
- **Strong agents.** Claude and Codex solved all six development tasks. The tasks are too easy to rank agents, and a mutation pilot confirmed it (Codex 24 of 24).

Navigate: [evidence map](docs/EVIDENCE_MAP.md) | [analysis commands](#recomputing-the-results) | [limitations](#limitations) | [library](https://github.com/TarangKhanna/traindoctor-env) | [ERRATUM](docs/ERRATUM_mmd.md) | [provenance](PROVENANCE.md) (split from the library at `6a8f565`, fresh history).

![Four ranks: rank 2 calls broadcast while ranks 0, 1 and 3 call all_reduce, so every rank times out with the same error](docs/figures/fig_failure.svg)

The figure shows the A1 kind. All results are exploratory. Raw results and checksums are in [`evidence/`](evidence/README.md).

## Install

Python 3.12 or 3.13. The research code imports the library (`traindoctor_env`, `jobs`), which is pinned by commit in `requirements.txt`. Run everything from the repository root.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-rl.txt   # only for rl/ (Apple Silicon, MLX)
export PYTHONPATH=. TD_KEY=$(python -c "import secrets;print(secrets.token_hex(16))")
python -m pytest -q --ignore=tests/test_first_update.py   # that file needs MLX (Apple Silicon)
```

## Results

**Training.** Qwen3-4B was fine-tuned with SFT and then GRPO, with a label-only reward (V0) or the recovery-aware reward (V2), six paired seeds each, and scored once on a sealed test set of 100 incidents. Success means V2 on the faulted task kinds: correct diagnosis, supporting evidence and a repair that passes live sandbox replay.

| Policy | Success |
| --- | ---: |
| Base model | 5.0% |
| SFT | 17.2% |
| Continued SFT, similar compute | 29.7% |
| GRPO, label-only reward (mean of 6 seeds) | 23.0% |
| GRPO, recovery-aware reward (mean of 6 seeds) | 33.4% |

The recovery-aware arm gained 10.36 percentage points over the label-only arm (95% bootstrap interval +5.83 to +15.26; positive in all six seeds). The gain came from evidence grounding, not better repair selection: both arms chose the recovering repair at the same rate (71.4% vs 71.5%) and label accuracy did not change. Continued SFT with similar compute (a single run) is not distinguishable from the recovery-aware arm (+3.7 points; interval −1.7 to +9.1), so the table does not show that the recovery-aware reward beats spending the same compute on more SFT. On the development split the same V2-minus-V0 comparison was +4.64 points (interval −0.69 to +9.98), inconclusive. The frozen protocol states a minimum meaningful difference of 7.2 points; its power-analysis amendment states 20. The 10.36-point gain is above the first and below the second, and the two were not reconciled ([ERRATUM_mmd.md](docs/ERRATUM_mmd.md)). Training used precomputed outcomes from a development container; test scores come from 369 live sandbox replays. Report: [evidence/results/TEST_REPORT.md](evidence/results/TEST_REPORT.md).

![Full-task success and working-repair selection for the two reward arms](docs/figures/fig_reward_outcome.svg)

**Attacking the verifier.** None of 86 fake repairs (swallow the exception, exit 0, shrink the world, raise timeouts, forge logs) was accepted: 48 were rejected by a correctness invariant and 38 failed during execution, which shows those cheats do not run, not that an invariant covers them. The reference repair was accepted 5/5 and alternative valid repairs 9/10 (one false rejection, fixed in a prepared v1.1 verifier). One scripted cheat, `world_size: 1` on the rank-divergence task, completes the job but leaves 72 missing and 8 duplicated samples. I wrote the attacks; there was no external review. See [REDTEAM_TABLE.md](evidence/results/REDTEAM_TABLE.md) and [the audit write-up](experiments/verifier_audit/VERIFIER_AUDIT.md).

![Outcomes of the fake repairs: rejected by a correctness invariant or failed during execution](docs/figures/fig_redteam_funnel.svg)

**Agent baselines.** On six development tasks, Claude Sonnet and Codex both produced patches that recovered all six; one Claude answer omitted a required evidence record (strict score 5/6 vs 6/6). The tasks are easy for strong agents and do not rank them. Per-task results are published; prompts and transcripts are not ([report](experiments/cross_agent_dev6/BENCHMARK_REPORT.md)).

**Mutation pilot.** Eight procedurally mutated tasks passed trusted re-execution validation, then Codex solved 24 of 24 attempts, so the tasks do not discriminate. The pilot was stopped (NO-GO) ([pilot](experiments/mutation_pilot/README.md)).

## Recomputing the results

[evidence/README.md](evidence/README.md) maps each claim to its file and the command that recomputes it. In short: `python scripts/analyze_test.py` and `scripts/audit_test_recount.py` rebuild the policy table and repair-choice counts from the scored rollouts, `scripts/make_figures.py` redraws the figures, `scripts/check_readme.py` checks the numbers above against those files, and `shasum -a 256 -c SHA256SUMS` in `evidence/results/` and the two experiment folders verifies every hash-pinned file.

## What can be recomputed

| Status | What | Notes |
| --- | --- | --- |
| Recompute here | Policy table, repair-choice counts, red-team recount, figures, README numbers, checksums | Commands under "Recomputing the results" and in the [evidence map](docs/EVIDENCE_MAP.md) (class A) |
| Historical | GPU training and evaluation (`modal_train/`, `rl/`), the sandbox replays, `scripts/redteam_suite.py` | Run before the split, not re-run after it; scripts left as used. The `redteam_suite.py` import of `traindoctor_env.modal_backend` was checked against the pinned library, the sandbox runs were not repeated. Modal image setup: [modal_train/README.md](modal_train/README.md) |
| Cannot be regenerated | Sealed test set (only the seed-list sha256 is published); cross-agent prompts, raw answers, outcomes and harness; mutation-pilot receipts; the gate experiment; per-run development JSON behind the six-seed development comparison | Withheld or lost; see evidence map class C |

The sealed-test protocol is copied unmodified into `evidence/prereg/`; its sha256 is in [evidence/README.md](evidence/README.md). The freeze happened in a private repository that is not published, so neither the copy nor its hash establishes when the freeze happened, and this repository's commit dates are not evidence of preregistration.

## Limitations

- Two synthetic task families; splits differ by seed, not by failure mechanism.
- Each repair menu has one recovering option, so V1 and V2 agree on 97.7% of development rollouts (a development-split figure, not measured on the sealed test). The results do not show that recovery-based reward improves repair selection.
- Six paired seeds is enough to see whether the sign is stable, not enough for statistical power.
- No claim about production NCCL/FSDP failures, arbitrary training programs, or difficulty for frontier agents.

Related documents: [evidence map](docs/EVIDENCE_MAP.md), [rerun example](docs/RERUN_EXAMPLE.md), [findings](docs/FINDINGS.md), [realism of the fault families](docs/REALISM.md), [related work](docs/RELATED_WORK.md). Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)); citation in [CITATION.cff](CITATION.cff); security reports in [SECURITY.md](SECURITY.md).
