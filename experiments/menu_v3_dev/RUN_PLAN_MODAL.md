# Verified rerun of the menu v3 outcome table on Modal (NOT RUN: needs the owner's approval to spend)

The `dev1` outcomes were executed without OS isolation (exploratory). This plan turns them into verified outcomes on the
library's existing Modal gVisor backend (`reexec.Replay(..., backend="modal")`, the path used by `scripts/modal_regression.py`
in the library), with no new infrastructure. The incident itself is repo-owned and runs locally; only the candidate
re-executions run in Modal sandboxes.

**Cost basis.** The library's Modal regression run (20 sandboxed re-executions, commit f36a0f6) reports $0.074 on its
script counter, about $0.004 per re-execution. That counter is an estimate from a CPU rate, not a bill.

| step | scope | re-executions | estimate | hard cap |
|---|---|---|---|---|
| 1 smallest | 2 seeds per kind (first two `dev1` seeds of each kind), all 12 items | 120 | ~$0.45 | $1.00 |
| 2 full (only if step 1 agrees with `dev1` on every row's accepted / rejected / execution-failure class) | all 60 `dev1` seeds | 720 | ~$2.70 | $4.00 |

Command (step 1; same TD_KEY as `dev1`, library at the commit recorded in `dev1/config.json`):

    TD_KEY=<dev1 key> TD_MODAL_SITE=<site-packages with modal 1.6.2> PYTHONPATH=<library>:. \
      python experiments/menu_v3_dev/build_tables.py --run dev1_modal_s1 --modal --workers 4 --seeds <10 seeds>

Network note (isolated 2026-10-20): the sandbox exec stream connects to `task-*.w.modal.host`. Behind an egress allowlist
that omits `*.modal.host` the client retries silently forever (the first attempt, run `dev1_modal_s1`, produced no rows
and was stopped; it is kept, empty). Allow `*.modal.com`, `*.modal2.com` and `*.modal.host`. Stop conditions: any infrastructure error
on more than 2 of 120 rows, or any row whose class differs from `dev1` (investigate before step 2, never re-roll). Record
actual spend in the ledger (`kind == "actual"`) before reporting any number.
