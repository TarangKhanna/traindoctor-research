# Evidence

See the [evidence map](../docs/EVIDENCE_MAP.md) for claims, sources, recount commands and reproducibility limits. [`results/`](results/) holds sealed-TEST rollouts, live replays, red-team records and reports; [`prereg/`](prereg/) holds frozen protocols and the power-analysis and token-cap amendments, written before their runs. After the README setup, run commands from the repository root with `PYTHONPATH=. TD_KEY=<any hex>`.

## Provenance of the preregistration

The sealed-TEST protocol (`prereg/test_protocol.md`, sha256 `7f2c403ea96ba7e5944fa5c18a48983a1c54321d069e42a878c426504af941da`) was frozen in a private development repository (commit `e2e2560`) before any TEST episode was built and is copied here unmodified. The seed-list sha256 (`52fb377289876036db6bc6766af5b3bcd0c67a7faeb8bd99b6a74f306aa4146d`) is recorded in `results/TEST_REPORT.md`. The private freeze commit is absent here; these hashes link to the original freeze, but this repository's commit dates do not establish preregistration.

Historical power-analysis command: `python evidence/prereg/power_analysis_sim.py 300` (slow; overwrites `power_analysis_results.json`). The 7.2-point MMD conflicts with the amendment; see the [erratum](../docs/ERRATUM_mmd.md).

## Red-team recount

Red-team recount from the committed table (the raw v1 JSON was overwritten by a later errored run; this table is the surviving record):

```bash
awk -F'|' 'NF==11 && $2!~/ID|---/ {id=$2; gsub(/ /,"",id); acc=$4; gsub(/ /,"",acc); cl=$5; gsub(/ /,"",cl);
  r=(id=="null")?"null":(id=="gold")?"gold":(id~/^alt/)?"alt":(id~/^S[0-9]/)?"borderline":"cheat"; n[r" "acc" "cl]++}
  END{for(k in n)print k, n[k]}' evidence/results/REDTEAM_TABLE.md | sort
```

Expected: `alt no invariant 1`, `alt yes accepted 9`, `borderline yes accepted 1`, `cheat no execution_failure 38`, `cheat no invariant 48`, `gold yes accepted 5`, `null no execution_failure 3`, `null no invariant 2` (no-op 0/5). The same command recounts the v1.1 table; supporting records: [`redteam_modal_v11.json`](results/redteam_modal_v11.json), [`VERIFIER_V11.md`](results/VERIFIER_V11.md). `python scripts/make_figures.py` regenerates the funnel from the table.

Both red-team tables used `exec_mode = modal_sandbox`: repairs ran in network-less gVisor Modal Sandboxes, with invariants recomputed outside. `dev_container` results used trusted pinned code locally and are exploratory training-reward views only. The 38 execution failures do not establish invariant coverage.

The isolation runs `results/redteam_isolate*.json` are the surviving raw red-team records. Hash-pinned results and frozen protocols remain byte-identical and may reference absent tooling or documents. In `results/TEST_REPORT.md` and the protocol, historical paths `docs/results/...`, `docs/prereg/...`, `modal_train/results/...` correspond to `evidence/results/...`, `evidence/prereg/...` and unpublished files. Verify results with `cd evidence/results && shasum -a 256 -c SHA256SUMS`.
