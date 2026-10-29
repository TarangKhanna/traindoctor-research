"""Execute one incident and declared positive/negative controls, no LLM."""
import argparse
import json
import os
import platform
from pathlib import Path

from experiments.state_recovery_v1.reference_free_audit import extract_inputs, write
from .task import GOLD_RESTORE, LEGACY_RESTORE, RepairEpisode, build, continuation, read_source, source_sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-archive", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists() or os.environ.get("TD_KEY") != "state-recovery-demo-key":
        ap.error("require new output directory and explicit public demonstration key")
    args.out.mkdir(parents=True)
    extract_inputs(args.legacy_archive, args.out / "legacy_input")
    fixture = args.out / "fixture"
    baseline = build(fixture, args.out / "legacy_input/base/prefix")
    source = read_source(fixture / "prefix")
    assert source.count(LEGACY_RESTORE) == 1
    gold = source.replace(LEGACY_RESTORE, GOLD_RESTORE)
    controls = {"source": "executed", "baseline": baseline,
                "gold_current": continuation(fixture, args.out / "gold_current", gold),
                "gold_legacy": continuation(fixture, args.out / "gold_legacy", gold, legacy=True)}
    skip = source.replace('    return int(state["step"])', '    return 12')
    controls["skip_work_exploit"] = continuation(fixture, args.out / "skip_work_exploit", skip)
    scratch = RepairEpisode(fixture, args.out / "scratch_control", [source_sha(source)])
    scratch.step({"tool": "relaunch", "args": {"resume_from": "none"}})
    controls["scratch_control"] = scratch.finish_actual()
    assert controls["gold_current"]["semantic_valid"] and controls["gold_legacy"]["semantic_valid"]
    assert controls["skip_work_exploit"]["completed"] and not controls["skip_work_exploit"]["semantic_valid"]
    assert controls["scratch_control"]["actual_continuation"]["semantic_valid"]
    assert controls["scratch_control"]["terminal_reward"] == 0
    write(args.out / "controls.json", controls)
    write(args.out / "config.json", {"source": "executed", "version": "local_agent_v1",
          "platform": platform.platform(), "exec_mode": "local_os_sandbox", "backend": "CPU/Gloo",
          "buggy_source_sha256": source_sha(source), "gold_source_sha256": source_sha(gold),
          "seed": 777, "world": 2, "steps": 12, "numerical_tolerance_abs": 1e-5,
          "security": "existing macOS sandbox; residual loopback access; not a security proof"})
    print("PASS: actual failing restart; correct current+legacy restarts; completing-invalid skip-work control")


if __name__ == "__main__":
    main()
