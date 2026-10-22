# Errata

Corrections to published files. Hash-pinned files are not edited; the right value is given here. The preregistration discrepancy is in [ERRATUM_mmd.md](ERRATUM_mmd.md).

| File | Wrong value | Right value | Evidence |
|---|---|---|---|
| `experiments/cross_agent_dev6/BENCHMARK_REPORT.md` (null row, "recovery only"; also "null 0/6" in the Reading paragraph) | null control recovery-only 0/6 | recovery-only 1/6; strict V2 stays 0/6 | `experiments/cross_agent_dev6/results/null/t06.json` (`verified_recovery` true) |
| `experiments/cross_agent_dev6/RESULTS.md` (null row, "verified recovery only" column) | 0/6 | 1/6 | same |
| `experiments/verifier_audit/VERIFIER_AUDIT.md` (red team, before this commit) | pooled rule-of-three 95% bound <= 0.035 for 0/86 | removed: the 86 attacks are author-chosen over 5 task instances, not independent trials; the counts stand | `evidence/results/REDTEAM_TABLE.md` |
| `experiments/verifier_audit/VERIFIER_AUDIT.md` (before this commit) | results "reported overall and by those two strata" | the prespecified per-stratum contrasts were not reported | `evidence/prereg/strata_v0_v2.md` |
| `README.md`, `docs/EVIDENCE_MAP.md`, `docs/FINDINGS.md`, `experiments/menu_v3_dev/*` (before this commit) | continued SFT "at similar compute"; semantic oracle "independent" of the verifier | compute comparability was not established (no run record of steps, tokens or GPU-seconds); classification uses the verifier's own computed outcome fields | `evidence/results/TEST_REPORT.md`, `experiments/menu_v3_dev/accounting.py` |
