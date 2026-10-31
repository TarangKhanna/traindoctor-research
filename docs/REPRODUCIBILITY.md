# Reproduction status

The 2026-10-31 pass used fresh local clones at environment
`221096d37690f21ab1a57cb79e387c3911064802` and research
`ed262d0b7ea37fd4f1643ea6b80bb076e5717d71`, with new dependency environments.
The existing Python interpreter and package/build caches were reused. This was
same-host reproduction, not independent external validation.

| Check | Outcome |
| --- | --- |
| Published recovery evidence | Recount passed: 12 continuations, 60 capture hashes |
| Fresh macOS recovery execution | All 12 continuations completed; NumPy recount exactly matched the published evidence |
| Fresh Harbor correct repair | Completed, reward 1; worker isolation checks passed |
| Fresh Harbor no-op | Two attempts interrupted by the host-memory guard; no valid semantic result |
| Fresh Harbor completing-invalid and tampered controls | Not started after the repeated resource stops |
| Focused environment/grader/listener checks | 23 passed, 1 listener-test failure |
| NumPy oracle tests | 2 passed |

The [new evidence packet](../evidence/reproduction_20261031/) contains the fresh
recovery archive, its recount, correct-repair receipt, resource samples and
machine-readable summary. Existing [Harbor evidence](HARBOR_VALIDATION.md)
records all four passing controls; its saved-evidence recount passed again.
The full fresh Harbor rerun remains incomplete.

## Fresh execution

After the README installation, from the research repository:

```bash
export TD_KEY=state-recovery-demo-key PYTHONPATH=.
python -m experiments.state_recovery_v1.demo --out /tmp/traindoctor-new-run
python -m experiments.state_recovery_v1.recount --run /tmp/traindoctor-new-run
```

The output path must not exist. This launches actual PyTorch/Gloo workers on
macOS using `sandbox-exec`; it refuses an unavailable sandbox. The trusted
reference helper runs locally without that sandbox. The archive recount in the
README launches no workers and is suitable for CPU CI.

Harbor uses the [existing Docker task and commands](https://github.com/TarangKhanna/traindoctor-env/blob/main/docs/HARBOR.md).
Run from an empty working directory with dotenv disabled. Keep at least 8 GiB
host RAM available throughout execution, with headroom for image builds. The
recorded correct control took 40.19 s; the fresh recovery demo took 67.40 s.
These are one-run wall times, not performance comparisons.

## Limits observed in this pass

The listener test observed the parent TCPStore on `127.0.0.1`, but no worker
listener within its four-second polling window. The assertion aborted the
parent; retained worker stderr subsequently reported `data service closed`.
No `uv_bind` denial was captured; the cause of the missing listener remains
unresolved. A live sample during the successful demo found launcher and both
workers listening on loopback. Historical Gloo flakiness is not declared fixed.

The no-op attempts reached 7.892 and 7.833 GiB available memory and were stopped.
They are infrastructure interruptions, not rejected semantic repairs. Memory
was sampled once per second; recorded swap growth within each attempt was zero.
No guard, isolation policy, verifier or test expectation was weakened.

Docker containers used read-only roots, network mode `none`, disjoint volumes
and workers without capabilities. Actual image source hashes matched the pinned
runtime and grader. macOS isolation and Docker isolation provide different
boundaries; neither this same-host pass nor author-written controls establish
general security or external-machine reproducibility.
