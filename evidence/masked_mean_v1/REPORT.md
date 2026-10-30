# Masked-loss coding-agent study

Script-generated from the completed local receipts.

A working multi-turn harness and an executable source-repair task are implemented. Successful autonomous repair was not demonstrated. These are individual episodes, not repair rates.

| Model | Responses | Valid calls | Applied edits | Relaunches | Reward | Inference s | Prefill s |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen | 11 | 10 | 0 | 0 | 0 | 387.504 | 325.281 |
| gemma | 2 | 1 | 0 | 0 | 0 | 9.655 | 2.142 |

Correct controls: maximum per-step absolute error 9.43949809e-08, tolerance 1e-5 (rtol=0).

The unchanged source completes all cases, passes balanced/all-valid regressions and fails unequal-count cases. The revised visible-schedule hardcode passes the visible case and fails other hidden schedules. The earlier fixed-count control was not a true visible-overfit baseline; original receipts remain local.

Qwen made model-selected inspection calls, then emitted a clipped prose/code proposal. Its sole complete Python block was separately security-reviewed and inserted with indentation only. That supervised diagnostic failed with AttributeError; it was neither model-applied nor model-requested execution and earns no agent credit.

Gemma ended on a tool-response delimiter. Why it emitted that delimiter is unresolved. The frozen controller prematurely accepted both endings as final. Later protocol corrections were regression-tested without model reruns; their effect on outcomes is unknown.

Reduced-evidence arms were gated off. No same-policy reward groups, model training or evidence-benefit comparison were run. Two zero rewards do not estimate RL reward variation.

## Recount without a model or workers

```bash
TD_KEY=state-recovery-demo-key PYTHONPATH=. python -m experiments.masked_mean_v1.public_packet --check evidence/masked_mean_v1
```

This checks file/capture hashes, independently reconstructs masked NumPy updates for each recorded control, and recounts model actions. It does not relaunch training or authenticate the original execution. Captures are job-serialized rank0 state; fidelity is conditional on exact-source safety review. Input records show delivery; consumption is inferred from numerical agreement. Non-rank0 state is unobserved.

## Next experiment (not performed)

Keep this task and verifier fixed. Version a protocol-corrected baseline separately before changing output budget; report the budget change as a confound. Validate applicable edits and relaunch/feedback before SFT/RL. Qwen's recorded prefill time motivates a separate prefix-cache parity/latency investigation; no cache speedup is established here.
