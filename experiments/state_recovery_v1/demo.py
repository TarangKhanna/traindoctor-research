"""Execute exactly one paired state-only incident and its declared controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

from jobs import trusted
from traindoctor_env.state_recovery_v1 import ACTIONS, RecoveryFixture, state_rule

from .oracle import evaluate


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("refusing to overwrite evidence")
    args.out.mkdir(parents=True)
    write(args.out / "config.json", {"source": "executed", "version": "state_recovery_v1",
          "seed": 777, "world": 2, "steps": 12, "checkpoint_every": 4, "worker_failure": [0, 9],
          "salt_boundary": 8, "same_schedule_for_all_actions": True,
          "platform": platform.platform(), "exec_mode": "local_os_sandbox",
          "cost": "raw repeated optimizer steps; no scratch exemption",
          "score": "0 if semantically invalid, otherwise max(0, 1-redone_steps/12)",
          "authored_controls_not_model_answers": True})
    print("Executing one genuine preemption and constructing its reference", flush=True)
    started = time.perf_counter()
    fixture = RecoveryFixture.create(args.out / "base")
    incidents, results, observations, transcripts = {}, {}, {}, {}
    for index, cause in enumerate(("sampler", "optimizer")):
        name = f"case{index}"
        incident = fixture.incident(args.out / name, cause)
        incidents[name] = incident
        status = incident.status()
        checkpoint, history = incident.inspect_checkpoint(), incident.committed_history()
        observations[name] = {"status": status, "checkpoint": checkpoint, "history": history}
        write(args.out / name / "observations.json", observations[name])
        selected = state_rule(checkpoint, history)
        transcript = [
            {"role": "scripted_policy", "tool": "job_status", "result": status},
            {"role": "scripted_policy", "tool": "inspect_checkpoint", "result": checkpoint},
            {"role": "scripted_policy", "tool": "committed_history", "result": history}]
        results[name] = {}
        for action in ACTIONS:
            print(name, action, flush=True)
            t0 = time.perf_counter()
            root = args.out / name / action
            receipt = incident.recover(action, root)
            independent = evaluate(root / "jobdir", fixture.root / "prefix", fixture.config, fixture.schedule)
            assert receipt["semantic_valid"] == independent["semantic_valid"], (name, action, receipt, independent)
            assert receipt["completed"] and receipt["source_preserved"], (name, action)
            write(root / "oracle.json", independent)
            results[name][action] = {"receipt": receipt, "oracle": independent,
                                     "wall_s": time.perf_counter() - t0}
            if action == selected:
                transcript += [{"role": "scripted_policy", "tool": "recover", "args": {"action": action},
                                "result": "recorded"},
                               {"role": "scripted_policy", "tool": "job_status",
                                "result": {"completed": receipt["completed"], "run_id": receipt["run_id"]}}]
        transcripts[name] = transcript
        write(args.out / name / "scripted_transcript.json", transcript)
    assert observations["case0"]["status"] == observations["case1"]["status"]
    # Same inputs without the decisive tool results: no deterministic policy can
    # distinguish the selective-repair targets. Robust full restore stays valid.
    ablated_equal = observations["case0"]["status"] == observations["case1"]["status"]
    controls = {"source": "executed", "initial_observation_identical": ablated_equal,
                "state_rule_actions": {k: state_rule(v["checkpoint"], v["history"]) for k, v in observations.items()},
                "metadata_perturbations": [], "oracle_agreement": {
                    "agree": sum(r["receipt"]["semantic_valid"] == r["oracle"]["semantic_valid"]
                                 for case in results.values() for r in case.values()),
                    "n": sum(len(case) for case in results.values())}}
    for name, incident in incidents.items():
        for metadata in ("renamed-run", "unrelated-node-99", "case1" if name == "case0" else "case0"):
            altered = incident.status(metadata)
            assert {k: v for k, v in altered.items() if k != "display_name"} == {
                k: v for k, v in observations[name]["status"].items() if k != "display_name"}
            picked = state_rule(incident.inspect_checkpoint(), incident.committed_history())
            assert picked == controls["state_rule_actions"][name]
            controls["metadata_perturbations"].append({"case": name, "display_name": metadata, "action": picked})
    for name, result in results.items():
        wanted = "restore_sampler" if name == "case0" else "restore_optimizer"
        other = "restore_optimizer" if name == "case0" else "restore_sampler"
        assert result[wanted]["receipt"]["semantic_valid"] and not result[other]["receipt"]["semantic_valid"]
        assert all(result[a]["receipt"]["semantic_valid"] for a in ("restore_full", "older", "scratch"))
        assert result["scratch"]["receipt"]["redone_optimizer_steps"] > result[wanted]["receipt"]["redone_optimizer_steps"]
    write(args.out / "controls.json", controls)
    write(args.out / "summary.json", {"source": "executed", "results": results,
          "wall_s": time.perf_counter() - started, "frontier": fixture.frontier,
          "source_sha256": fixture.source, "worker_failure_executed": trusted.ledger(fixture.root / "prefix")[0]["kill_done"],
          "verdict": "engineering demonstration; full restore remains a common optimal action; NO-GO as a challenging RL benchmark"})
    lines = ["# State recovery execution", "", "source: executed", "",
             "| state | action | completes | semantic valid | oracle agrees | redone steps | cost score |",
             "|---|---|---|---|---|---:|---:|"]
    for name, result in results.items():
        for action, row in result.items():
            r = row["receipt"]
            lines.append(f"| {name} | {action} | {r['completed']} | {r['semantic_valid']} | True | {r['redone_optimizer_steps']} | {r['cost_score']:.4f} |")
    (args.out / "TABLE.md").write_text("\n".join(lines) + "\n")
    hashes = []
    for path in sorted(args.out.rglob("*.json")):
        hashes.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(args.out)}")
    (args.out / "SHA256SUMS").write_text("\n".join(hashes) + "\n")
    print("PASS: 12 actual continuations, independent oracle agreement, source/evidence controls", flush=True)


if __name__ == "__main__":
    main()
