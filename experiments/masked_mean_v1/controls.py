"""Execute and freeze one task's authored controls. No model or training run."""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

from jobs import manifest, trusted
from jobs.dataserver import salt_for
from jobs.resubmit import resubmit
from experiments.state_recovery_v1.reference_free_audit import write
from . import task


def controls():
    broken = task.source()
    old = "        loss = total / count.clamp_min(1)\n"
    gold = broken.replace(old, task.CORRECT)
    grad = broken.replace(old, "        loss = total\n").replace(
        "        sync_grads(comm, flat, step, rank, cfg)\n",
        "        sync_grads(comm, flat, step, rank, cfg)\n"
        "        dist.all_reduce(count, group=group)\n"
        "        flat.mul_(world / count)\n")
    totals = [sum(counts) for counts in task.CASES["visible"]]
    hardcoded = broken.replace(old, f"        loss = total * world / ({totals!r}[step] * y.shape[1])\n")
    return {"correct_loss": gold, "correct_gradient": grad, "unchanged": broken,
            "constant_scaling": broken.replace(old, "        loss = 2 * total / count.clamp_min(1)\n"),
            "visible_count_hardcoded": hardcoded,
            "skip_update": broken.replace("        opt.step()\n", "        opt.step()\n        # Missing gradients must not count as training.\n").replace(
                "        opt.step()\n", "        for p in params:\n            p.grad.zero_()\n        opt.step()\n", 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--only", choices=tuple(controls()))
    args = ap.parse_args()
    if args.out.exists() or os.environ.get("TD_KEY") != "state-recovery-demo-key":
        ap.error("require new output directory and the explicit public demonstration key")
    args.out.mkdir(parents=True)
    import psutil
    swap0 = psutil.swap_memory().used

    def guard():
        sample = {"available_bytes": psutil.virtual_memory().available,
                  "swap_used_bytes": psutil.swap_memory().used}
        sample["swap_growth_bytes"] = sample["swap_used_bytes"] - swap0
        write(args.out / "latest_resources.json", sample)
        if sample["available_bytes"] < 8 * 1024**3 or sample["swap_growth_bytes"] > 128 * 1024**2:
            raise RuntimeError("local resource guard; no automatic retry")

    guard()
    sources = controls()
    write(args.out / "config.json", {"source": "executed", "version": task.VERSION,
          "platform": platform.platform(), "source_review": "authored controls reviewed before execution",
          "seed": 777, "world": 2, "steps": 4, "atol": 1e-5, "rtol": 0,
          "cases": task.CASES, "task_sha256": task.source_sha(Path(task.__file__).read_text()),
          "controls_sha256": task.source_sha(Path(__file__).read_text()),
          "sources": {k: task.source_sha(v) for k, v in sources.items()}})
    rows = []
    for name, source in sources.items():
        if args.only and name != args.only:
            continue
        for case in task.CASES:
            if args.probe_only and (name != "correct_loss" or case != "visible"):
                continue
            started = time.perf_counter()
            guard()
            verdict = task.execute_case(args.out / name / case, source, case)
            rows.append({"control": name, "case": case, "wall_s": time.perf_counter() - started, **verdict})
            write(args.out / "rows.json", rows)
            print(name, case, verdict["completed"], verdict["semantic_valid"], round(rows[-1]["wall_s"], 3), flush=True)
            if name.startswith("correct"):
                assert verdict["semantic_valid"], "authored correct repair failed; stop before model trials"
                assert max(s["max_abs"] for s in verdict["per_step"]) < 1e-6, "correct repair lacks tolerance margin"
    if args.probe_only:
        print("PASS: trusted per-step rank0 captures match independent NumPy oracle")
        return
    if args.only:
        assert all(r["completed"] for r in rows)
        if args.only == "visible_count_hardcoded":
            assert next(r for r in rows if r["case"] == "visible")["semantic_valid"]
            assert any(not r["semantic_valid"] for r in rows if r["case"] != "visible")
        write(args.out / "summary.json", {"source": "executed", "control": args.only,
              "control_gate": "PASS for this control only", "rows": len(rows), "models_run": False})
        print("PASS: bounded single-control recheck")
        return
    for name in sources:
        if name.startswith("correct"):
            continue
        failed = [r for r in rows if r["control"] == name and not r["semantic_valid"]]
        assert failed, f"{name} was not discriminated"
        assert all(r["completed"] for r in rows if r["control"] == name), f"{name} did not complete"
        if name == "constant_scaling":
            assert len(failed) >= 2
    # Restart the identical, unchanged source: no edits and no copied oracle outputs.
    jd = args.out / "unchanged/visible/jobdir"
    guard()
    manifest.path(jd).write_text("[]\n")
    result = resubmit(jd, "none", sandbox=True, salt=salt_for(777))
    assert result.run_id is not None
    restart = task.grade(jd, task.config("visible"), task.source_sha(sources["unchanged"]))
    assert restart["completed"] and not restart["semantic_valid"]
    # A public success claim cannot substitute for independently recomputed state.
    (jd / "runs" / result.run_id / "claimed_success.json").write_text('{"semantic_valid":true}\n')
    forged = task.grade(jd, task.config("visible"), task.source_sha(sources["unchanged"]))
    assert not forged["semantic_valid"]
    write(args.out / "restart_and_forgery.json", {"restart_only": restart, "forged_public_evidence": forged})
    # Reuse the executed visible incident as the model-facing frozen workspace.
    trusted.copy_job(args.out / "unchanged/visible/jobdir", args.out / "fixture/prefix")
    summary = {"source": "executed", "control_gate": "PASS", "rows": len(rows),
               "restart_only_valid": restart["semantic_valid"], "forged_public_evidence_valid": forged["semantic_valid"],
               "models_run": False, "fixture": "fixture/prefix"}
    write(args.out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
