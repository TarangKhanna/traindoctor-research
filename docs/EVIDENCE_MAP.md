# Evidence map

Each public claim, the file it comes from and a command that recounts it. Run from the repository root with `PYTHONPATH=. TD_KEY=<any hex>` and the install from the README. Class: **A** recomputable from published results; **B** reproducible from public assets (needs the pinned library, and possibly Modal); **C** historical, not fully reproducible (withheld seeds, transcripts or external infrastructure). "Printed" is what the command printed on a fresh clone (Python 3.12).

Shorthand: `AWK` is the red-team recount in [evidence/README.md](../evidence/README.md); `R` is `evidence/results`.

| claim | source | recount command | printed | class |
| --- | --- | --- | --- | --- |
| Success on the sealed test: base 5.0, SFT 17.2, continued SFT 29.7; 100 episodes | `R/test/test_scored.json`, `R/test/TEST_REPORT_analysis.json` | `python scripts/analyze_test.py evidence/results/test/test_scored.json $TMPDIR/an`, then compare `$TMPDIR/an.json` with the committed analysis | identical JSON; 5.0 / 17.2 / 29.7; n=100 | A |
| GRPO arms 23.0% / 33.4% (means of 6 seeds) | same | `python scripts/check_readme.py` | "label-only arm mean 23.0", "verified arm mean 33.4" ok | A |
| V2 arm minus V0 arm +10.36 points, 95% bootstrap interval +5.83 to +15.26 | `R/test/TEST_REPORT_analysis.json` | `analyze_test.py` as above | 10.36 [5.83, 15.26]  (continued SFT at similar compute is not distinguishable from V2; the gain is evidence grounding, not repair choice) | A |
| V2 arm minus continued SFT +3.7 (-1.7 to +9.1) | same | same | 3.70 [-1.67, 9.11] | A |
| Working repair chosen at 71.4% vs 71.5% | `R/test/AUDIT_RECOUNT.txt` | `python scripts/audit_test_recount.py`; `python scripts/repair_choice_by_seed.py` | v0 71.41, v2 71.46 (README rounds to 71.4 / 71.5) | A |
| 369 unique scenario/repair pairs replayed in sandboxes | `R/test/test_replay_r*.json` | `python scripts/audit_test_recount.py` | "replay pairs 369 ... modal_sandbox 369", 0 conflicting duplicates | A for the count; the replays themselves need Modal (C) |
| Fake repairs: 0 of 86 accepted, 48 by invariant, 38 execution failure | `R/REDTEAM_TABLE.md` | `AWK` on the table | cheat/invariant 48, cheat/execution_failure 38, none accepted | A |
| Reference repairs 5/5, alternatives 9/10, no-op 0/5 | `R/REDTEAM_TABLE.md` | `AWK` | gold accepted 5; alt accepted 9, rejected 1; null rejected 5 | A |
| F11 on A1_src leaves 72 missing and 8 duplicated samples | `R/REDTEAM_TABLE.md` | `python scripts/check_readme.py` | "F11 A1_src 72 gaps / 8 dups" ok | A |
| Execution-failure rejections are not invariant coverage | same table | the 38 and the 48 are separate classes in the `AWK` output | | A |
| Verifier v1.1: alternatives 10/10, 0 of 85 verified cheats accepted | `R/REDTEAM_TABLE_v11.md` | `AWK` on the v1.1 table | alt accepted 10; cheats 48 invariant + 37 execution failure = 85, plus 1 error row | A (v1.1 is opt-in; the test used v1) |
| V1 and V2 agree on 97.7% of 3072 development rollouts (development figure) | `R/v1v2_seeds_out.json` | `python -c "import json;d=json.load(open('evidence/results/v1v2_seeds_out.json'))['part1'];print(d['n'],d['frac_equal'],d['v1_not_v2'])"` | 3072 0.9772 70 | A |
| Six-seed development comparison +4.64 points, CI -0.69 to +9.98 | `R/SEEDS_3_5_REPORT.md` | none: per-run inputs are not published (the paired-t arithmetic is covered by `tests/test_analyze_test.py::test_paired_t_known_sample`) | not recountable here | C |
| Minimum meaningful difference 7.2 points | `evidence/prereg/test_protocol.md` line 42 | `grep -n "7.2" evidence/prereg/test_protocol.md` | the protocol says 7.2 and cites `power_analysis_amendment.md`, which states an MMD of 20 points; the two frozen files disagree and I did not rerun the simulation | C |
| Both agents recover 6/6 development tasks; strict score Claude 5/6, Codex 6/6 | `experiments/cross_agent_dev6/table.md` | `cd experiments/cross_agent_dev6 && shasum -a 256 -c SHA256SUMS` | 27 files OK; table not recounted (prompts, raw answers, outcomes withheld) | C |
| Mutation pilot: 8 tasks validated, Codex 24/24, NO-GO | `experiments/mutation_pilot/results/` | `shasum -a 256 -c SHA256SUMS` there; offline tests per its README (clean-clone steps: `experiments/mutation_pilot/REPRODUCE.md`) | 17 files OK; 31 tests passed; 24/24 stated in `FINAL_VERDICT.md`, not recounted (receipts withheld) | C |
| Distractor-menu gate: V1 != V2 on 2.6% of 3072 (interval 0.6 to 5.3) | `docs/FINDINGS.md` | none: gate artifacts and analysis script are withheld | not recountable here | C |
| README numbers match their sources | `README.md` | `python scripts/check_readme.py` | all checks ok, exit 0 | A |
| Hash-pinned results unchanged | `R/SHA256SUMS` | `cd evidence/results && shasum -a 256 -c SHA256SUMS` | 35 OK, 0 failed | A |
| Sealed-test protocol was frozen before the test | `evidence/prereg/test_protocol.md` | `shasum -a 256 evidence/prereg/test_protocol.md` against the hash in `evidence/README.md`; the freeze commit is in a private repository | the hash links the file to the freeze; dates here are not evidence | C |
| Sealed test seeds | sha256 only, in `R/TEST_REPORT.md` | seed list withheld | the test set cannot be regenerated | C |
| Training and GPU evaluation | `modal_train/`, `rl/` | needs Modal credit and Apple Silicon | not re-run after the split | C |
| Failure, reference repair and fake repair, locally | library `scripts/quickstart.py` | [RERUN_EXAMPLE.md](RERUN_EXAMPLE.md) | invariants printed; the verified verdict needs Modal | B |
| Verified replay of gold and F11 on A2 | `R/quickstart_verified_replay.txt` | recorded output; a rerun needs Modal | gold accepted, F11 rejected (I1) | C |
| A2v2 behaviour | library tests | [A2V2_STATUS.md](A2V2_STATUS.md) | result tables withheld | C |

See [ERRATUM_mmd.md](ERRATUM_mmd.md) for the 7.2 vs 20 point MMD discrepancy.
