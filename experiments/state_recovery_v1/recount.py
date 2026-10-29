"""Recompute continuation semantics from saved raw captures, without launching jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from jobs import manifest, trusted
from jobs.build import base_config
from jobs.dataserver import SaltSchedule, salt_for
from traindoctor_env.state_recovery_v1 import ACTIONS, source_hashes, state_rule

from .oracle import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if "TD_KEY" not in os.environ or args.out and args.out.exists():
        ap.error("require explicit demonstration TD_KEY and a new output path")
    config = json.loads((args.run / "config.json").read_text())
    cfg = base_config(2, config["seed"], 5, 5, 0)
    base = salt_for(config["seed"])
    schedule = SaltSchedule(base, hashlib.sha256(base + b"|state-recovery-v1").digest(), 8)
    anchor = args.run / "base/prefix"
    hashes = source_hashes(anchor)
    frontier = max(trusted.ledger(anchor)[0]["acked"].values())
    rows, checked_captures, best_sets, observations = [], 0, [], []
    for case in ("case0", "case1"):
        obs = json.loads((args.run / case / "observations.json").read_text())
        observations.append(obs)
        best, scores = -1, {}
        for action in ACTIONS:
            jd = args.run / case / action / "jobdir"
            receipt = json.loads((jd.parent / "receipt.json").read_text())
            verdict = evaluate(jd, anchor, cfg, schedule)
            assert verdict["semantic_valid"] == receipt["semantic_valid"]
            assert source_hashes(jd) == hashes
            manifest.validate(jd, manifest.load(jd))
            for record in trusted.ledger(jd):
                for cap in record["captures"]:
                    data = (trusted.trusted_dir(jd) / cap["file"]).read_bytes()
                    assert hashlib.sha256(data).hexdigest() == cap["sha256"]
                    checked_captures += 1
            last = trusted.ledger(jd)[-1]
            assert last["completed"] and last["sandboxed"]
            redo = max(0, min(frontier, cfg["steps"]) - last["start_step"])
            assert redo == receipt["redone_optimizer_steps"]
            score = 1 - redo / cfg["steps"] if verdict["semantic_valid"] else 0
            assert score == receipt["cost_score"]
            scores[action] = score
            best = max(best, score)
            rows.append({"case": case, "action": action, "semantic_valid": verdict["semantic_valid"],
                         "redone_steps": redo, "cost_score": score, "oracle": verdict})
        best_sets.append({a for a, s in scores.items() if s == best})
    assert observations[0]["status"] == observations[1]["status"]
    result = {"source": "executed", "continuations_recounted": len(rows),
              "captured_hashes_checked": checked_captures,
              "common_optimal_actions": sorted(set.intersection(*best_sets)), "rows": rows,
              "state_rule": [state_rule(o["checkpoint"], o["history"]) for o in observations],
              "reference_backend": "trusted local subprocess, local_unsandboxed_trusted",
              "continuation_backend": "macOS sandbox-exec, local_os_sandbox",
              "oracle_scope": "independent NumPy continuation computation conditional on genuine anchors and shared input generation"}
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("PASS:", len(rows), "raw continuation recounts;", checked_captures,
          "capture hashes; common optimal actions:", result["common_optimal_actions"])


if __name__ == "__main__":
    main()
