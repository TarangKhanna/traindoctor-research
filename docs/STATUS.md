# Local milestone status

2026-10-29: one state-only recovery pair implemented on isolated `demo/state-recovery-v1` branches. All twelve continuations complete and agree with an independent NumPy continuation oracle. Full restore is optimal in both states; scratch is correct but repeats more work. Historical verifier v1.2 and evidence remain unchanged.

Cached Qwen 4B and Gemma E2B were profiled locally with no swap growth. A parity-checked Gemma prefix-cache probe improved the five-request total by 1.47x including setup. No new training was run. See [the report](../experiments/state_recovery_v1/REPORT.md) for exact evidence, model-control limitations and reproduction commands.

Verdict: engineering demonstration GO; conditional-selection RL benchmark NO-GO. Stop after this worked example. Independent reproduction remains outstanding. No cloud compute, dependency changes, paid actions, pushes, merges or publication occurred.

Frozen source commits: environment d568f69, research 6e158ce. Fresh local clones pass 36 focused tests, archive checksum and README checks. A fresh sandboxed execution and independent recount again pass all twelve continuations. The prepared local dependency environment was reused; external-machine reproduction remains outstanding.

Final transport hardening: environment 65e4bea fixes a live-observed wildcard TCPStore bind with a launcher-owned loopback store. Live lsof regression and 241 focused tests pass; all twelve continuations and their recount pass again with unchanged observations and outcomes. Final evidence is results/state_recovery_v1_loopback_verified; earlier packets are preserved.

Local follow-up: [scoped agent instructions](../AGENTS.md) and [append-only learning notes](STATE_RECOVERY_LEARNINGS.md) capture the scientific corrections, discarded unsafe approach and measured performance limits. They require reuse before new infrastructure and preserve the current stop decision. This is a documentation-only update.

2026-10-29 reference-free follow-up, isolated `audit/reference-free-v1`: primary-source gate finds direct interval-replay prior art in Proof-of-Learning (2021). A bounded checker audits the existing twelve continuations without fault-free/reference/verdict inputs; eight accept and four reject, agreeing with historical labels. Two new CPU/Gloo continuations are bitwise repeatable. A coherent poisoned-anchor chain demonstrates why the claim must remain conditional. GO for local engineering validation; NO-GO for replay novelty, paper/release now or RL. See [the report](REFERENCE_FREE_REPORT.md). No environment code, v1.2 or historical receipts changed; no experimental model training/inference or external publication/outreach occurred.

2026-10-29 owner-authorized local-agent milestone, isolated `demo/local-agent-v1`: one checkpoint-code migration incident and positive/negative/scratch/skip-work controls execute correctly. Cached Gemma made three real native inspection calls, then hit the RAM/swap guard before proposing a patch. It is unscored infrastructure failure, not model repair failure. Memory remained below the 8 GiB floor with our model stopped, so Qwen and the reduced-evidence conditions remain unrun. Actual model-generated executable repair is **not yet demonstrated**. See [the bounded report](LOCAL_AGENT_V1_REPORT.md); no training or multi-agent experiment was started.
