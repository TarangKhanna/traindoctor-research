# TrainDoctor Research

**Does a repaired training job preserve the intended computation—or merely finish?**

TrainDoctor runs failing distributed PyTorch/Gloo jobs, exposes inspection and editing tools, and checks whether repairs preserve samples, parameters and optimizer state. Experiments and receipts are here; the [environment](https://github.com/TarangKhanna/traindoctor-env) contains the runtime and verifier.

Tier A is single-turn diagnosis, evidence citation and repair-menu selection. Tier B is multi-turn inspection, source edits, actual distributed relaunch and feedback.

## Results

**Table A — Single-turn post-training (Qwen3-4B)**

| Policy | Full-task success |
|---|---:|
| Base Qwen3-4B | 5.0% |
| SFT | 17.2% |
| Continued SFT | 29.7% |
| GRPO V0 | 23.0% |
| GRPO V2 | 33.4% |

Full-task success requires diagnosis, grounded evidence and a verified repair. Scores cover **80 faulted incidents among 100 evaluation incidents**, with four completions per incident. GRPO means cover **six paired training seeds** per arm; continued SFT is one run, without a properly compute-matched comparison.

V2 improved evidence grounding alongside full-task success, without demonstrating better repair selection: **71.4% versus 71.5%** for V0/V2. No V1-only training arm isolates the effect of repair verification. [Evaluation and uncertainty](evidence/results/TEST_REPORT.md) · [Independent recount](evidence/results/test/AUDIT_RECOUNT.txt) · [Historical detail](docs/PROJECT_DETAIL.md).

**Table B — Actual multi-turn coding-agent episodes**

| Task / model | Turns | Valid tool calls | Applied edits | Agent relaunches | Final semantic result |
|---|---:|---:|---:|---:|---|
| Checkpoint agent-v2 / Gemma | 12 | 12 | 0 | 0 | No execution; reward 0 |
| Checkpoint agent-v2 / Qwen | 12 | 12 | 1 | 0 | No agent execution; reward 0 |
| Checkpoint agent-v2 / Qwen retest | 9 | 8 | 0 | 0 | Resource interruption; unscored |
| Masked-loss original / Qwen | 11 | 10 | 0 | 0 | No execution; reward 0 |
| Masked-loss original / Gemma | 2 | 1 | 0 | 0 | No execution; reward 0 |
| Masked-loss corrected / Qwen 512 | 16 | 14 | 1 | 1 | Python failure; reward 0 |
| Masked-loss corrected / Gemma 512 | 9 | 8 | 1 | 0 | Context limit; reward 0 |
| Masked-loss corrected / Qwen 1024 | 16 | 16 | 1 | 2 | Python failure; reward 0 |

These are individual attempts across different tasks and controller versions, not success rates or a model ranking. Models are cached Qwen3-4B-Instruct-2507 and Gemma-4-E2B-it, both 4-bit. Corrected runs reserve 512 or 1,024 output tokens inside an 8,192-token context; the larger budget also reduces prompt room. Configurations: [checkpoint study](docs/AGENT_V2_REPORT.md), [original masked-loss packet](evidence/masked_mean_v1/REPORT.md), [corrected-controller recount](docs/CODING_AGENT_RECOUNT.md).

**Autonomous semantically correct repair has not yet been demonstrated.** Qwen did complete edit → relaunch → failure feedback → further action. Safety review supplied no repaired code. Scripted successes and separately executed/normalized proposals receive no autonomous credit.

## What the task-design audit establishes

V1/V2 advantages differ in only 20 of 384 saved eight-completion development groups. These are evaluation samples, not recovered training gradients. Menu redesigns added completing-but-wrong repairs, but fixed choices remained strong; full restoration solves both state-recovery twins, and one global-normalization patch solves all masked-loss schedules. Two source-only Qwen checks failed to produce a correct formula or patch. These cases do not yet justify further RL. [Learning-signal audit](LEARNING_SIGNAL_AUDIT.md).

## Reproduce the saved local example

Python 3.12; install the pinned runtime, then recount the saved executions:

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

Use a new extraction directory. This launches no workers or models. [Reproduction status and fresh execution](docs/REPRODUCIBILITY.md) · [Agent recount](docs/CODING_AGENT_RECOUNT.md).

Two synthetic families and seed holdouts do not establish unseen-mechanism or production generalization. Captured-state checks depend on trusted inputs and reviewed source. [Evidence map](docs/EVIDENCE_MAP.md) · [Harbor controls](docs/HARBOR_VALIDATION.md) · [Testing boundaries](docs/TESTING.md) · [Provenance](PROVENANCE.md).

Apache-2.0: [LICENSE](LICENSE), [NOTICE](NOTICE), [citation](CITATION.cff).
