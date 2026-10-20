# A2v2: archival status

A2v2 is a second version of the "one rank stalls before communication" task (A2). The stall comes from a slow producer instead of a deletable wait, because the red-team case S01 (skip the shard wait) was accepted on A2 v1 as a surface patch that removes the injected cause ([verifier audit](../experiments/verifier_audit/VERIFIER_AUDIT.md)).

What is public:

- The scenario code and tests: `jobs/a2v2.py`, `jobs/template_a2v2/`, `scripts/redteam_a2v2.py` and `tests/test_a2v2.py` in the library at the commit pinned in `requirements.txt`. A2v2 is opt-in and the generator does not draw it.
- A2v2 is not part of any headline number in the README or in [FINDINGS](FINDINGS.md).

What is withheld (removed from the public tree on 2026-10-19 and kept in a private archive):

- The red-team result tables and raw records for one scenario (world size 3, seed 101): 11 cases from the first run and a single-case rerun of F06. In the first run F06 errored (sandbox unavailable) and S02 was accepted; S01 and R1 were rejected by sample accounting.
- A scenario write-up. [FINDINGS](FINDINGS.md) referred to `A2V2_SCENARIO.md` in the library; that file is not present at the pinned commit or at the library head, so the reference there is stale.

Why: the tables are one scenario, one seed, with an errored case, and were never an input to a reported result, so they were pruned with other development material rather than kept as evidence. Their absence means the A2v2 verifier behaviour is not independently recountable from this repository. To check it, run the `tests/test_a2v2.py` tests (CPU) and `scripts/redteam_a2v2.py` (Modal Sandbox, needs a Modal account) from the library.

## 2026-10-20 audit: S02 is a valid repair, and that is an environment flaw

The earlier unexpected acceptance (S02: remove the readiness wait and derive batch positions locally) was re-run on the same
scenario (seed 101, world 3), together with the gold repair, both legitimate alternatives, S01, R1 and a no-op, under
verifier v1 and v1.1. Raw rows: [`experiments/a2v2_audit/results/local1/`](../experiments/a2v2_audit/results/local1/).
These runs are exploratory: repo-authored cases executed on a development Mac without OS isolation, not verified outcomes.

| case | v1 | v1.1 | evidence |
|---|---|---|---|
| gold, alt1, alt2 | accepted | accepted | every committed state equal to the reference (fast path) |
| S02 | accepted | accepted | every committed state equal to the reference (fast path); 0 sample gaps, 0 duplicates |
| S01, R1 | rejected | rejected | sample accounting: 53 gaps, 53 duplicates (the placeholder index is read) |
| no-op | rejected | rejected | the job does not complete |

S02 trains exactly what the reference trains. The producer's index only ever holds `divmod(step, steps_per_epoch)`
(`jobs/template_a2v2/data.py`, `_produce`), which the sampler already knows, and the batches themselves come from the data
service by position. So S02 is not a verifier false acceptance: the verifier's contract is training correctness, and that
holds. It shows that A2v2 still does not make the producer necessary. A version 3 would have the trusted data service own
the batch order (for example, positions served per shard under a launcher-held permutation), so that positions derived
locally fetch the wrong batches and sample accounting rejects them. Until then A2v2 stays opt-in and outside the generator.
The same property means Task A2 has no completing-but-invalid repair in menu v3.
