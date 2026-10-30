"""Recount fresh receipts; preserve model transcripts and bounded raw archives."""
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

from experiments.local_agent_v1.audit_v2 import trajectory_metrics
from experiments.state_recovery_v1.reference_free_audit import write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", type=Path, required=True)
    ap.add_argument("--smoke", type=Path, required=True)
    ap.add_argument("--trajectory", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("require a fresh output directory")
    args.out.mkdir(parents=True)
    read = lambda p: json.loads(p.read_text())
    rows = read(args.controls / "rows.json")
    smoke = read(args.smoke / "receipt.json")
    assert smoke["not_model_evidence"] and smoke["result"]["terminal_reward"] == 1
    assert all(r["completed"] for r in rows)
    positive = [r for r in rows if r["control"].startswith("correct")]
    assert all(r["semantic_valid"] for r in positive)
    metrics = []
    for path in args.trajectory:
        t = read(path)
        m = trajectory_metrics(path)
        m["observed_execution"] = any(r.get("tool") == "relaunch" and
            '"execution_status"' in r.get("observation", "") and i + 1 < len(t["rows"])
            for i, r in enumerate(t["rows"]))
        m["reduced"] = t["reduced"]
        metrics.append(m)
        transcript = ["# Actual local-model transcript", "", "Model: " + t["model_path"],
                      "", "Result: " + json.dumps(t.get("result")), ""]
        for row in t["rows"]:
            transcript.extend([f"## Response {row['turn']}", "", "```text", row["raw"], "```", "",
                               "Observation:", "```text", row.get("observation", row.get("protocol_error", "")), "```", ""])
        if t.get("partial_generation"):
            transcript.append("Generation interrupted; partial token IDs and prompt remain in trajectory.json.")
        (path.parent / "TRANSCRIPT.md").write_text("\n".join(transcript) + "\n")
    summary = {"source": "executed", "version": "masked_mean_v1",
        "research_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "controls": [{"control": r["control"], "case": r["case"], "completed": r["completed"],
                       "semantic_valid": r["semantic_valid"], "wall_s": r["wall_s"],
                       "max_abs": max((s.get("max_abs") or 0 for s in r["per_step"]), default=None)} for r in rows],
        "positive_max_abs": max(s["max_abs"] for r in positive for s in r["per_step"]),
        "scripted_smoke_reward": smoke["result"]["terminal_reward"], "trajectories": metrics,
        "rollout_groups_sampled": 0, "informative_group_fraction": None,
        "limitations": ["job-serialized rank0 state; fidelity conditional on exact-source security review",
                        "input delivery is trusted; consumption inferred from numerical agreement",
                        "no non-rank0 state attestation", "authored single mechanism, not held-out generalization",
                        "individual episodes, no repair-rate or evidence-benefit estimate", "no policy training"]}
    write(args.out / "summary.json", summary)
    inputs = [args.controls / "config.json", args.controls / "rows.json", args.smoke / "receipt.json", *args.trajectory]
    write(args.out / "config.json", {"source": "executed", "analysis": "recount of fresh local receipts",
          "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}})
    lines = ["# Masked-mean bounded demonstration", "", "Script-produced recount of executed receipts.", "",
             f"Correct-control maximum per-step absolute error: {summary['positive_max_abs']:.9g}.",
             "", "| Control | Case | Completes | Semantic success |", "|---|---|---|---|"]
    lines += [f"| {r['control']} | {r['case']} | {r['completed']} | {r['semantic_valid']} |" for r in rows]
    lines += ["", "The complete inspect/edit/relaunch/observe/finish smoke is scripted, not model evidence.", "",
              "| Model | Reduced | Responses | Valid calls | Applied edits | Observed execution | Reward | Failure stage |",
              "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {m['model_revision']} | {m['reduced']} | {m['response_turns']} | {m['protocol_valid']} | "
              f"{m['applied_edits']} | {m['observed_execution']} | {m['terminal_reward']} | {m['failure_stage']} |" for m in metrics]
    lines += ["", "No sampled reward groups or learning comparison. No diagnosis-accuracy estimate.", "",
              "Future work: first require successful autonomous policy trajectories and structural heldouts; "
              "then propose assistant-only quantized LoRA SFT and matched continued SFT. RL needs separately "
              "sampled same-policy groups with semantic reward variation. No training is authorized here.", "",
              "An investigator/reviewer comparison would share the single-agent token, tool and execution "
              "budgets; reviewer messages count against the same totals and reveal no private evidence. "
              "It remains unimplemented.", "", "Limits:", ""]
    lines += ["- " + limit for limit in summary["limitations"]]
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n")
    for folder in [args.controls, args.smoke, *(p.parent for p in args.trajectory)]:
        archive = args.out / (folder.name + ".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(folder.rglob("*")):
                if (p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts
                        and "secrets" not in p.parts and not p.name.startswith(".env")
                        and p.name != "credentials.json"):
                    z.write(p, folder.name + "/" + str(p.relative_to(folder)))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir() if p.is_file()}
    write(args.out / "hashes.json", hashes)
    print(json.dumps({"control_rows": len(rows), "model_episodes": len(metrics),
                      "positive_max_abs": summary["positive_max_abs"], "report": str(args.out / "REPORT.md")}))


if __name__ == "__main__":
    main()
