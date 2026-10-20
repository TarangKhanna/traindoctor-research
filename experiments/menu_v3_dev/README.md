# Menu v3 development experiment (`menu_v3_dev`)

A new, separately versioned experiment. It is not part of the sealed TEST preregistration, does not change any historical
result, and uses a fresh development seed range (50000 upward, disjoint from DEV 10000-10399, TRAIN 20000-20399, the
sealed TEST pool and the burned seeds; `build_tables.py` asserts this). No model was trained for it.

**Question.** With repair choices that let a job finish while breaking training semantics, does the verified-recovery
reward (V2) carry information that label plus evidence (V1) does not? Historical figures (different menus and task
distribution, not comparable): V1 != V2 on 2.3% of 3,072 development rollouts with menu v1 and 2.6% with menu v2.

## Menu v3 (library `traindoctor_env/menu_v3.py`)

Twelve items, identical text in every variant, each shown as the code at its location after the edit. Which items recover a
job is decided by re-execution only. The intended mechanism is written down here so it can be compared with what was
observed; it is never used to score.

| role | intended mechanism | intended expectation |
|---|---|---|
| `sync:canonical` | gradient sync is a plain all_reduce (average) | recovers A1 variant 0; no-op elsewhere |
| `sync:reduce_bcast` | reduce to rank 0, broadcast, divide by world size: the same average by other collectives | valid alternative for A1 v0 (and harmless elsewhere) |
| `sync:align` | every rank broadcasts bucket 0 from rank 0 and all-reduces bucket 1: collectives agree, gradients are not averaged | completes A1 v0 but trains wrongly; harmful elsewhere |
| `perrank:canonical` | the per-rank override is all_reduce | recovers A1 variant 1 |
| `perrank:global_reduce_bcast` | the global sync mode becomes reduce_bcast (with A1 v1's override, every rank then uses the same averaging) | valid alternative for A1 v1; makes other jobs diverge |
| `shard:canonical` | rank r reads its own shard directory | recovers A2 |
| `ckpt:canonical` | checkpoints store the sampler position, restore loads it | recovers B_SKIP and B_DUP |
| `ckpt:from_step` | restore derives the sampler position from the checkpoint's step; the save line is left as the job has it | behaviourally valid for B_SKIP and B_DUP; B_DUP's checkpoints then store no position (rejected by verifier v1's format check, by design) |
| `ckpt:from_progress` | restore takes the sampler position from the progress file the crashed run wrote | completes Task B but skips the samples between the checkpoint and the crash |
| `ckpt:fresh_optimizer` | restore loads model and sampler but not the optimizer momentum | completes Task B but the first update after resume is wrong |
| `relaunch` | no change | recovers B_OK only |
| `exclude_rank` | world size minus one (destructive) | never valid: changes the distributed computation |

A2 has no completing-but-invalid item: its shard files carry no data (the data service serves batches by position), so
every way of getting past the stalled wait trains identically. This is the same environment property that makes A2 S01
and A2v2 S02 behaviourally valid (see `docs/A2V2_STATUS.md`).

## Steps and files

1. `build_tables.py --run dev1 --per-kind 12 --unsandboxed-trusted`: 60 incidents (12 per kind), each candidate executed
   once and judged under verifier format checks v1, v1.1 and v1.2 from that one run. Writes `outcomes.jsonl`,
   `bundles.jsonl`, `config.json`.
2. `bundle_records.py --run dev1`: re-executes each incident, requires identical menu ids and executed edits, writes the
   records, Truth and prompt used for scoring (`records.jsonl`).
3. `agent_eval.py --run dev1 --agent codex-medium --samples 2`: strong-agent answers on those prompts, from an empty
   directory with a read-only sandbox (prompt only).
4. `analyze.py --run dev1`: scores with the library's own `parse_answer` and `verdict`, writes `analysis/`.

**Execution mode.** Every outcome in `dev1` is `local_unsandboxed_trusted`: the candidates are repo-authored code, run on
a development Mac without OS isolation. These outcomes are exploratory. The library's strict V2 requires a sandboxed run,
so it is 0 for answers scored against local `dev1`; that analysis reports `v2_exploratory` (the same verdict with the isolation requirement
waived) and never calls it verified. The Modal verification subsequently ran as `dev1_modal_s1b` and `dev1_modal_s2`:
720 sandboxed rows match the local verdicts 720/720 under v1, v1.1 and v1.2. Verified scoring is in
`results/dev1/analysis_verified/`. The result is **NO-GO for RL**; see [REPORT.md](REPORT.md),
[ACCOUNTING.md](ACCOUNTING.md) and the completed [Modal run plan](RUN_PLAN_MODAL.md).
Actual sandbox spend was $2.92610876, including the stalled `dev1_modal_s1` attempt and one probe, as recorded in the
Modal billing report for 2026-10-20, app `traindoctor-sandbox` (the local spend ledger is not published); this is not solely the cost of the 720 rows.

## Criteria, stated before the results were read

- Headline check: verifier v1.2 (v1.1 plus the history check), with v1 and v1.1 reported beside it.
- Candidate level (independent of any model): at least one class of item that completes the run with normal status and
  is rejected by a correctness invariant, not by an execution failure; every reference repair accepted; behaviourally
  valid alternatives accepted. "Semantically valid" is judged by an independent reference criterion: the final model equals
  the uninterrupted reference run's and every sample was consumed as in the reference.
- Reward information (policy level): P(V2 = 0 | V1 = 1) and P(V1 != V2), with seed-clustered bootstrap intervals, by kind,
  role and policy. The engineering target is roughly 10% overall disagreement; more important is whether a meaningful
  fraction of V1-passing answers choose completing-but-invalid repairs and whether those are exactly the ones rejected.
- Go/no-go for designing (not running) an RL comparison: GO only if (a) the candidate-level criteria hold under v1.2 with
  no false acceptance of a semantically invalid item, (b) a strong agent with V1 credit chooses a completing-invalid repair
  on a non-trivial fraction of seeds, and (c) the extra V2 signal is not fully explained by an avoid-list of fixed items.
  The completing-invalid items have the same text in every menu, so a learner can memorize "never pick these" without
  reasoning about semantics; the analysis reports how much of the V1-pass/V2-fail mass a menu-only avoid-list learner
  (leave-one-seed-out) removes. If it removes all of it, the reward teaches item identities, not training semantics, and
  an RL run on this menu could not support a claim about semantic repair learning. Otherwise NO-GO with the evidence.
  Modal verification is complete; the development result remains exploratory.
- The task set is not tuned against these numbers: the 60 seeds and the menu were fixed before any agent answer was read.
