# TrainDoctor Research

Experiments, analysis and execution evidence for [TrainDoctor](https://github.com/TarangKhanna/traindoctor-env), a distributed-training repair environment.

## Recovery example

A two-rank CPU/Gloo job resumes from a checksum-valid checkpoint assembled from different committed steps. Every tested recovery completes, including those that change the intended computation.

| Delivered state | Restore sampler | Restore optimizer | Restore both |
| --- | --- | --- | --- |
| Stale sampler cursor | Correct | Incorrect | Correct |
| Stale optimizer momentum | Incorrect | Correct | Correct |

The verifier checks actual continuations against sample and state-transition contracts. A separate NumPy implementation cross-checks the training updates. Full restoration works in both states.

[Experiment](experiments/state_recovery_v1/REPORT.md) · [Latest reproduction](docs/REPRODUCIBILITY.md) · [Trust assumptions and counterexamples](docs/REFERENCE_FREE_REPORT.md)

## Reproduce the saved local example

Python 3.12; the environment dependency is pinned in `requirements.txt`.

```bash
git clone https://github.com/TarangKhanna/traindoctor-research
cd traindoctor-research
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
export TD_KEY=state-recovery-demo-key PYTHONPATH=.
python -m zipfile -e evidence/reproduction_20261031/raw_evidence.zip /tmp/traindoctor-evidence
python -m experiments.state_recovery_v1.recount --run /tmp/traindoctor-evidence
```

Use a new extraction directory. Expected output:

```text
PASS: 12 raw continuation recounts; 60 capture hashes; common optimal actions: ['restore_full']
```

This recount launches no workers or model inference. See [fresh execution and reproduction status](docs/REPRODUCIBILITY.md#fresh-execution) for the macOS sandbox requirements and results.

## Results

Qwen3-4B performance on the 80 faulted incidents in a 100-incident held-out evaluation, using new seeds from the same task generator. Each policy generated four rollouts per incident; GRPO rows average six training seeds.

| Policy | Full-task success |
| --- | ---: |
| Base model | 5.0% |
| SFT | 17.2% |
| Continued SFT | 29.7% |
| GRPO, diagnosis-only reward | 23.0% |
| GRPO, recovery-aware reward | 33.4% |

Recovery-aware GRPO improved full-task success by 10.4 percentage points over diagnosis-only GRPO. Repair-choice rates remained similar (71.4% versus 71.5%), and an advantage over continued SFT was not established. [Analysis, uncertainty and limitations](docs/PROJECT_DETAIL.md#results-historical-v01) · [Evidence map](docs/EVIDENCE_MAP.md)

## Explore

- [Harbor validation](docs/HARBOR_VALIDATION.md): correct, no-op, completing-invalid and tampering controls with a separate Docker verifier.
- [Local coding-agent study](evidence/masked_mean_v1/REPORT.md): tool harness, semantic regressions and recountable model traces; successful autonomous repair remains unproven.
- [Tests](docs/TESTING.md), [historical evidence](evidence/README.md) and [protocol errata](docs/ERRATUM_mmd.md).

Apache-2.0: [LICENSE](LICENSE) · [Citation](CITATION.cff) · [Provenance](PROVENANCE.md) · [Security](SECURITY.md)
