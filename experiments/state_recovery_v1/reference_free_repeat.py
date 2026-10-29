"""Two local CPU/Gloo continuations: numerical repeatability, no reference run."""
import argparse
import dataclasses
import os
import platform
import subprocess
import time
from pathlib import Path

import torch

from jobs import sandbox, trusted
from jobs.launcher import run_job

from .reference_free import KEYS, _read_capture, compare, recipe, transition
from .reference_free_audit import write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists() or os.environ.get("TD_KEY") != "state-recovery-demo-key":
        ap.error("require new output path and explicit public demo key")
    if not sandbox.available():
        ap.error("macOS sandbox required; no fallback")
    args.out.mkdir(parents=True)
    cfg, schedule = recipe(args.inputs)
    anchor_jd = args.inputs / "base/prefix"
    anchor_caps = trusted.ledger(anchor_jd)[0]["captures"]
    anchor8 = _read_capture(anchor_jd, next(c for c in anchor_caps if c["step"] == 8))
    saved_jd = args.inputs / "case0/restore_full/jobdir"
    saved_cap = next(c for c in trusted.ledger(saved_jd)[-1]["captures"] if c["step"] == 12)
    saved_final = _read_capture(saved_jd, saved_cap)
    results = []
    for rep in range(2):
        jd = args.out / f"repeat{rep}/jobdir"
        jd.parent.mkdir()
        trusted.copy_job(anchor_jd, jd)
        sockets = []

        def inspect(pid, port):
            # The launcher-owned TCPStore is observable immediately after spawning.
            live = subprocess.run(["lsof", "-a", "-p", str(os.getpid()), f"-iTCP:{port}",
                                   "-sTCP:LISTEN", "-P", "-n"],
                                  capture_output=True, text=True, timeout=5)
            sockets.extend(line.split()[-2] for line in live.stdout.splitlines()[1:])
            assert sockets and all(s.startswith("127.0.0.1:") for s in sockets), sockets

        start = time.perf_counter()
        result = run_job(jd, run_id="run1", resume_from="ck0008", start_step=8,
                         on_spawn=inspect, sandbox_jobs=True, salt=schedule)
        wall = time.perf_counter() - start
        record = trusted.ledger(jd)[-1]
        assert record["completed"] and record["sandboxed"]
        final = _read_capture(jd, next(c for c in record["captures"] if c["step"] == 12))
        comparison = compare(final, saved_final)
        oracle = transition(anchor8, final, 8, 12, cfg, schedule)
        bitwise = all(torch.equal(final[k], saved_final[k]) for k in KEYS)
        assert comparison["ok"] and oracle["ok"]
        row = {"source": "executed", "repeat": rep, "job": dataclasses.asdict(result),
               "exec_mode": "local_os_sandbox", "backend": "CPU/Gloo; macOS sandbox-exec",
               "continuation_steps": [8, 12], "wall_s": wall, "listeners": sockets,
               "bitwise_equal_to_saved_continuation": bitwise,
               "difference_from_saved_continuation": comparison,
               "numpy_transition_comparison": oracle}
        results.append(row)
        print("repeat", rep, "completed; bitwise equal:", bitwise, "wall_s", wall, flush=True)
    write(args.out / "repeatability.json", {"source": "executed", "platform": platform.platform(),
          "torch": torch.__version__, "config": cfg, "rows": results,
          "scope": "two additional 8-to-12 continuations compared with one existing continuation; no initialization or GPU claim"})
    print("PASS: two sandboxed CPU/Gloo continuations; loopback store; NumPy checks")


if __name__ == "__main__":
    main()
