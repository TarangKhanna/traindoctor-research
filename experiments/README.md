# Experiments

| folder | what it is |
|---|---|
| [`cross_agent_dev6/`](cross_agent_dev6/) | Claude Sonnet vs Codex on six development tasks; report in `BENCHMARK_REPORT.md`, hashes in `SHA256SUMS` |
| [`mutation_pilot/`](mutation_pilot/) | procedural mutation pilot (negative result); see its README |
| [`menu_v3_dev/`](menu_v3_dev/) | menu v3 (repairs that finish but break training): verified outcome table, Codex and Qwen3-4B checkpoint answers, reward-disagreement analysis; decision NO-GO, see `REPORT.md` |
| [`a2v2_audit/`](a2v2_audit/) | re-run of the A2v2 red-team cases: S02 trains identically to the reference (environment flaw, not a false acceptance) |
| [`verifier_audit/`](verifier_audit/) | verifier audit write-up (`VERIFIER_AUDIT.md`) and the output of the leak-test mutation check |

Raw results of the sealed TEST and the red team are in [`../evidence/results/`](../evidence/results/).
