# Local milestone status

2026-10-29: one state-only recovery pair implemented on isolated `demo/state-recovery-v1` branches. All twelve continuations complete and agree with an independent NumPy continuation oracle. Full restore is optimal in both states; scratch is correct but repeats more work. Historical verifier v1.2 and evidence remain unchanged.

Cached Qwen 4B and Gemma E2B were profiled locally with no swap growth. A parity-checked Gemma prefix-cache probe improved the five-request total by 1.47x including setup. No new training was run. See [the report](../experiments/state_recovery_v1/REPORT.md) for exact evidence, model-control limitations and reproduction commands.

Verdict: engineering demonstration GO; conditional-selection RL benchmark NO-GO. Stop after this worked example. Independent reproduction remains outstanding. No cloud compute, dependency changes, paid actions, pushes, merges or publication occurred.
