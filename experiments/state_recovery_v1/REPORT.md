# State only recovery and local model profiles

The engineering demonstration works: correct, identical training source can resume a checksum-valid checkpoint and finish with an incorrect sample or optimizer trajectory. Two selective repairs have opposite outcomes under the two constructed states. A separate NumPy continuation oracle agrees with all twelve executed continuations.

The learning-task verdict remains NO-GO. Full committed restoration is legitimate and optimal in both states, including under the new illustrative recomputation penalty. Thinking-off Gemma chose that common optimal action without inspection. This is a useful recovery example, not evidence of a difficult RL problem.

## Contract and architecture

Public baselines were verified as environment `0e8cf6a` and research `53fa710`; clean local method-fixes worktrees matched GitHub. All changes are on isolated `demo/state-recovery-v1` branches. Historical Task B, `Replay.outcomes()`, verifier v1.2, its scratch allowance, dependency pins and historical evidence remain unchanged.

The historical protocol checks repaired source through a fresh fault injection. The new `state_recovery_v1` instead preserves genuine pre-failure captures and grades the selected continuation. It reuses the existing launcher, manifest checks and semantic/provenance checks, with cost excluded from the correctness boolean. Raw redone optimizer steps are reported separately, without a scratch exemption. Scratch remains correct when it reconstructs the intended computation. Its checkpoint-continuity field is not applicable.

One existing MLP runs on two CPU/Gloo ranks, checkpointing at steps 4 and 8 before an actual worker preemption. External artifact assembly then delivers either step-8 model/optimizer with step-4 sampler state, or step-8 model/sampler with step-4 momentum. Manifest checksums are recomputed correctly; the failure is semantic consistency, not byte-integrity failure. The source stays identical to the pristine template.

The public initial summaries and available interventions are identical. Read-only checkpoint measurements and genuine committed history distinguish the components. They expose neither a hidden cause label nor an oracle verdict. These are two controlled explanations consistent with the same initial observation; practitioner validation of their operational realism has not been performed.

## Executed outcomes

Final evidence: [execution table](../../results/state_recovery_v1_20261029_verified/TABLE.md), [raw-data recount](../../results/state_recovery_v1_20261029_verified/recount.json), and [full summary](../../results/state_recovery_v1_20261029_verified/summary.json).

| Intervention | Sampler mixture | Optimizer mixture | Redone steps | Illustrative score when valid |
|---|---|---|---:|---:|
| Resume delivered checkpoint | completes, invalid | completes, invalid | 2 | 0 |
| Restore sampler only | valid | completes, invalid | 2 | 0.8333 |
| Restore optimizer only | completes, invalid | valid | 2 | 0.8333 |
| Restore full committed state | valid | valid | 2 | 0.8333 |
| Resume coherent step 4 | valid | valid | 6 | 0.5000 |
| Scratch | valid | valid | 10 | 0.1667 |

All twelve jobs completed. The independent recount performs sixty capture-hash checks, including copied prefix captures, and identifies `restore_full` as a common optimal action. The score is zero for invalid outcomes and `max(0, 1-redone_steps/12)` otherwise. It is an illustrative normalization, not a preregistered utility function. It excludes checkpoint I/O, investigation, verification and transfer costs; no GPU-hour equivalence was measured.

The oracle independently implements MLP forward/backward, rank averaging, momentum, SGD and logical sample-history reconstruction using NumPy float64. It shares the specified data service and starts from genuine committed anchors; it is an independent continuation computation conditional on those inputs, not fully independent end-to-end validation. Accepted trajectories have maximum model/momentum discrepancy about `5.22e-8`, below the existing `1e-5` contract. Finite-difference gradient tests pass. The existing reference helper records every step rather than every fourth step; its Python source and update rule remain the pristine template's.

The fixed data challenge preserves the pre-step-8 prefix and uses the same continuation salt for all actions and the reference. Source hashes and original captures are retained. Preemption and continuations use macOS `sandbox-exec`; the existing reference helper runs trusted local code without OS isolation. Seatbelt's loopback permissions are not equivalent to historical network-less Modal isolation.

Evidence ablation makes the two initial observations indistinguishable for selecting a *selective* repair. It does not make successful recovery impossible: full restore remains optimal. Display-name perturbations are structural noninterference controls, not a population robustness experiment. The recorded multi-turn reference policy is explicitly scripted, never represented as a model transcript. All observations share one paired incident; no independent-trial confidence intervals are claimed.

## Local inference and code fixing

The requested model research was split into task design, Mac models/runtimes and training efficiency, then combined and criticized before implementation. Actual inference used cached 4-bit models, MLX 0.31.2 / MLX-LM 0.31.3, AC power and serialized Metal runs on the M4 Air with 32 GB. No downloads, cloud jobs, paid APIs or dependency changes were needed.

Each timing shape has one discarded warmup and five measured requests. These are approximately equal native-token contexts, not identical byte strings across tokenizers. Model order and thermal conditions were not counterbalanced, so the table is an observed hardware/runtime profile, not a controlled architecture ranking.

| Model | Native input tokens | Median TTFT | Median decode tokens/s | Peak MLX GB |
|---|---:|---:|---:|---:|
| Qwen3-4B-Instruct-2507 | 2,068 | 17.23 s | 17.18 | 3.18 |
| Qwen3-4B-Instruct-2507 | 8,212 | 79.35 s | 13.54 | 4.00 |
| Gemma 4 E2B, thinking off | 2,055 | 3.26 s | 33.12 | 3.33 |
| Gemma 4 E2B, thinking off | 8,217 | 13.47 s | 29.90 | 3.51 |

The [profile recount](../../results/local_profile_recount_20261029/summary.json) checks these medians against raw rows. Available memory stayed above 10.7 GiB at recorded checks, and swap growth was zero. Model-load times are process-load measurements with uncontrolled filesystem cache, not cold-boot latency.

Gemma's supported prefix cache matched uncached greedy output in all five requests. Including cache preparation, the five-request total fell from 35.76 to 24.30 seconds: a measured 1.47x improvement on this repeated-prompt probe. Warm-cache median TTFT was 0.52 seconds. This does not establish multi-turn episode acceleration, stochastic G=4/G=8 reward diversity, or shared-prefix backpropagation equivalence.

The bounded JSON-tool control used each model's chat template, not its native function-calling parser. Qwen immediately selected optimizer restoration in both states: correct once, incorrect once. Gemma initially exhausted its answer budget in thinking text because the installed tokenizer wrapper enables thinking by default. Inspection isolated that default; the explicit thinking-off rerun selected full restore in both states, correctly and optimally. These are single greedy controls on a dependent pair, not population accuracy estimates. They demonstrate neither an investigation benefit nor a need for a larger model.

Free-form code repair was also probed on the existing visible B_DUP source bug. Qwen returned an applicable edit that redundantly checks an already enclosing condition and never saves the missing sampler. The exact reviewed proposal was replayed through the unchanged sandboxed v1.2 protocol: the job completed but parameters and sample history remained wrong ([receipt](../../results/reviewed_qwen_code_20261029/receipt.json)). Gemma's code answer exhausted 512 tokens before a complete edit; functional correctness is unknown. No general generated-code execution service was introduced. The exploratory generic validator was discarded after review found security-boundary problems.

## Acceleration findings and limits

The immediate measured opportunity is inference prefill reuse and bounded relevant tool context. Token throughput alone does not measure useful recoveries per hour. A larger Qwen 8B capacity control was not cached or profiled; the existing 4B and Gemma controls should first be interpreted as prompt/interface checks.

[Liquid DSpark](https://www.liquid.ai/blog/lfm2.5-dspark) supplies model-specific speculative drafts for LFM2.5 with llama.cpp/SGLang integration. Its advertised gains are not transferable to Qwen or Gemma. LFM3 is a separate candidate; installed-runtime compatibility was not established and no Liquid runtime/model was installed. [Gemma thinking documentation](https://ai.google.dev/gemma/docs/capabilities/thinking) and the locally inspected MLX wrapper explain why explicit template settings matter.

For training, prefer [MLX-native quantized LoRA](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) when it solves a memory bottleneck. The original [QLoRA](https://arxiv.org/abs/2305.14314) NF4/double-quantization/paged-optimizer results do not prove those mechanisms or speedups on Apple hardware. Checkpointing trades computation for memory; smaller microbatches must preserve the effective group/objective. No new training was run because the worked task still has a common optimum.

The saved local 0.6B plumbing microbenchmarks were recounted, not rerun: stock versus completion-head-only update medians are 14.04 versus 12.29 seconds at G=1, and 26.97 versus 23.83 at G=2, with two saved runs per cell. Sanitized raw rows and provenance are in the profile recount. These are historical update measurements, not end-to-end RL speedups or 4B timing evidence. Shared-prefix training, speculative rollouts and different RL algorithms require dtype-specific loss/gradient/optimizer parity and a measured bottleneck before implementation; current upstream [mlxrl](https://github.com/inductiveML/mlxrl) features cannot simply be assumed present in the installed version.

## Reproduction and decision

From this research checkout, using the matching environment checkout and a prepared Python 3.12 environment with the existing pinned CPU dependencies:

```bash
export TD_KEY=state-recovery-demo-key
export PYTHONPATH=/path/to/matching/traindoctor-env:.
uv run --no-project --offline /path/to/prepared/python \
  -m experiments.state_recovery_v1.demo --out results/new-reproduction
uv run --no-project --offline /path/to/prepared/python \
  -m experiments.state_recovery_v1.recount --run results/new-reproduction
```

The first command executes jobs; the second independently recomputes continuation semantics and validates saved provenance without launching jobs. Mac OS sandbox availability is required for execution. Optional inference profiles require the separately installed versions listed above and already downloaded model snapshots; they are not dependencies of the CPU demonstration. The historical dependency pin is intentionally unchanged, so matching local environment source must be placed on `PYTHONPATH` for this unreleased extension.

The committed [raw evidence archive](../../results/state_recovery_v1_20261029_verified/raw_evidence.zip) contains the captures, ledgers, source copies and observations. Its adjacent SHA256 file checks the archive; local stderr containing interpreter paths is excluded. A saved-evidence recount needs no Gloo job:

```bash
python -m zipfile -e results/state_recovery_v1_20261029_verified/raw_evidence.zip /tmp/td-saved-evidence
uv run --no-project --offline /path/to/prepared/python \
  -m experiments.state_recovery_v1.recount --run /tmp/td-saved-evidence
```

To repeat an optional model profile, supply a local snapshot directory to `scripts/profile_mac_models.py` together with `--model-id`, `--demo`, `--env`, `--out` and `--reps 5`. `--thinking off` reproduces the final setting; `--thinking default` reproduces the wrapper-default control. Every output path must be new.

Verified locally: the worked incident, all declared interventions, the NumPy oracle, capture/source checks, focused tests and the two model profiles. Not verified: Linux execution, a clean external machine, independent practitioner reproduction, native-function-call model quality, stochastic reward groups, new training performance or new-mechanism generalization. A preliminary execution with stale full-restore manifest metadata and the thinking-on Gemma run are retained separately and are not the final evidence.

The source commits are environment `d568f69` and research `6e158ce`. Fresh local clones passed 18 environment tests (`test_state_recovery_v1.py`, `test_menu_v3.py`) and 18 research tests (`test_state_recovery_oracle.py`, `test_menu_v3_analysis.py`). The archive checksum and both existing README checkers pass. A fresh end-to-end execution from those frozen clones again passed all twelve continuations and the raw-data recount. These checks reused the prepared local Python environment; they are not a fresh dependency installation or an external reproduction.

Exact demo command for the provided worktree layout, from the research worktree:

```bash
caffeinate -s env TD_KEY=state-recovery-demo-key \
  PYTHONPATH=../state-recovery-v1-env:. UV_CACHE_DIR=/private/tmp/td-audit-uv \
  uv run --no-project --offline ../../.venv/bin/python \
  -m experiments.state_recovery_v1.demo --out results/a-new-local-run
```

Environment changes are `traindoctor_env/state_recovery_v1.py`, `tests/test_state_recovery_v1.py`, `README.md`, `docs/STATE_RECOVERY_V1.md` and `docs/STATUS.md`. Research changes are the new `experiments/state_recovery_v1/` demonstration/oracle/recount/packaging/profile-summary/reviewed-proposal modules and this report, `scripts/profile_mac_models.py`, `tests/test_state_recovery_oracle.py`, `README.md`, `.gitignore`, `docs/STATUS.md`, and the explicitly named generated evidence packets under `results/`. The two source commits provide the complete file manifest. No historical evidence was edited; local traceback paths are excluded from the archives.

GO for the bounded engineering artifact. NO-GO for claiming a challenging cost-conditioned RL task or starting GRPO on this pair. The highest-value next action is independent reproduction and critique of this exact example, preserving the common robust recovery rather than manufacturing a penalty to defeat it. No additional incident pairs are warranted by these results.
