# TrainDoctor research artifacts

Training and evaluation code, sealed-test evidence, experiments and analysis scripts behind the results reported for TrainDoctor, an environment for diagnosing and repairing failed distributed PyTorch (Gloo) training jobs. The environment, jobs runtime, verifier, Modal repair backend and quickstart live in the library, [traindoctor-env](https://github.com/TarangKhanna/traindoctor-env). This repository holds what substantiates the numbers. It was split from the library at commit `6a8f565` and has a fresh history ([PROVENANCE.md](PROVENANCE.md)).

![Four ranks: rank 2 calls broadcast while ranks 0, 1 and 3 call all_reduce, so every rank times out with the same error](docs/figures/fig_failure.svg)

All results are exploratory. Raw results and checksums are in [`evidence/`](evidence/README.md).

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

The recovery-aware arm gained 10.36 percentage points over the label-only arm (95% bootstrap interval +5.83 to +15.26; positive in all six seeds). The gain came from evidence grounding, not better repair selection: both arms chose the recovering repair at the same rate (71.4% vs 71.5%) and label accuracy did not change. Continued SFT is not clearly different from the recovery-aware arm (+3.7 points; interval −1.7 to +9.1) and is a single run. Training used precomputed outcomes from a development container; test scores come from 369 live sandbox replays. Report: [evidence/results/TEST_REPORT.md](evidence/results/TEST_REPORT.md).

![Full-task success and working-repair selection for the two reward arms](docs/figures/fig_reward_outcome.svg)

**Attacking the verifier.** None of 86 fake repairs (swallow the exception, exit 0, shrink the world, raise timeouts, forge logs) was accepted: 48 were rejected by a correctness invariant and 38 failed during execution, which shows those cheats do not run, not that an invariant covers them. The reference repair was accepted 5/5 and alternative valid repairs 9/10 (one false rejection, fixed in a prepared v1.1 verifier). One scripted cheat, `world_size: 1` on the rank-divergence task, completes the job but leaves 72 missing and 8 duplicated samples. I wrote the attacks; there was no external review. See [REDTEAM_TABLE.md](evidence/results/REDTEAM_TABLE.md) and [the audit write-up](experiments/verifier_audit/VERIFIER_AUDIT.md).

**Agent baselines.** On six development tasks, Claude Sonnet and Codex both produced patches that recovered all six; one Claude answer omitted a required evidence record (strict score 5/6 vs 6/6). The tasks are easy for strong agents and do not rank them. Per-task results are published; prompts and transcripts are not ([report](experiments/cross_agent_dev6/BENCHMARK_REPORT.md)).

**Mutation pilot.** Eight procedurally mutated tasks passed trusted re-execution validation, then Codex solved 24 of 24 attempts, so the tasks do not discriminate. The pilot was stopped (NO-GO) ([pilot](experiments/mutation_pilot/README.md)).

## Recomputing the results

[evidence/README.md](evidence/README.md) maps each claim to its file and the command that recomputes it. In short: `python scripts/analyze_test.py` and `scripts/audit_test_recount.py` rebuild the policy table and repair-choice counts from the scored rollouts, `scripts/make_figures.py` redraws the figures, `scripts/check_readme.py` checks the numbers above against those files, and `shasum -a 256 -c SHA256SUMS` in `evidence/results/` and the two experiment folders verifies every hash-pinned file.

## What is not reproducible here

- Cross-agent prompts, raw answers, outcomes and harness; mutation-pilot validation receipts; the gate experiment; per-run development JSON behind the six-seed development comparison.
- GPU training and evaluation (`modal_train/`, `rl/`) need Modal credit and an Apple Silicon machine for the MLX parts. They were not re-run after the split. The `modal_train` image builds expect the library source tree at a pinned version. `scripts/redteam_suite.py` imports `traindoctor_env.modal_backend` from the pinned library (import checked against `requirements.txt` on Python 3.12; the sandbox runs themselves were not repeated). The remaining files are left as they were used for the published runs.
- The sealed test seed list is withheld; only its sha256 is published, so the test set cannot be regenerated.
- The sealed-test protocol was frozen in a private repository before any test episode was built. It is copied unmodified here, and this repository's commit dates are not evidence of preregistration (see [evidence/README.md](evidence/README.md)).

## Limitations

- Two synthetic task families; splits differ by seed, not by failure mechanism.
- Each repair menu has one recovering option, so V1 and V2 agree on 97.7% of development rollouts. The results do not show that recovery-based reward improves repair selection.
- Six paired seeds is a robustness minimum, not statistical power.
- No claim about production NCCL/FSDP failures, arbitrary training programs, or difficulty for frontier agents.

Related documents: [evidence map](docs/EVIDENCE_MAP.md), [rerun example](docs/RERUN_EXAMPLE.md), [findings](docs/FINDINGS.md), [realism of the fault families](docs/REALISM.md), [related work](docs/RELATED_WORK.md). Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)); citation in [CITATION.cff](CITATION.cff); security reports in [SECURITY.md](SECURITY.md).
