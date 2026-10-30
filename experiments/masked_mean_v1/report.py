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
    ap.add_argument("--diagnostic", type=Path)
    ap.add_argument("--extra-controls", type=Path)
    ap.add_argument("--checks", type=Path)
    ap.add_argument("--review", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("require a fresh output directory")
    args.out.mkdir(parents=True)
    read = lambda p: json.loads(p.read_text())
    rows = read(args.controls / "rows.json")
    extra_rows = read(args.extra_controls / "rows.json") if args.extra_controls else []
    replacements = {(r["control"], r["case"]): r for r in extra_rows}
    rows = [replacements.get((r["control"], r["case"]), r) for r in rows]
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
        last = t["rows"][-1] if t["rows"] else {}
        m["last_finish_reason"] = last.get("finish_reason")
        m["last_raw"] = last.get("raw")
        m["tool_level_errors"] = sum(r.get("observation", "").startswith("error:") for r in t["rows"])
        if last.get("finish_reason") == "length" and not m["applied_edits"]:
            m["failure_stage"] = "output_cap_before_tool_patch"
        elif last.get("raw") == "<|tool_response>":
            m["failure_stage"] = "tool_protocol_termination"
        metrics.append(m)
        transcript = ["# Actual local-model transcript", "", "Model: " + t["model_path"],
                      "", "Result: " + json.dumps(t.get("result")), ""]
        for row in t["rows"]:
            transcript.extend([f"## Response {row['turn']}", "", "```text", row["raw"], "```", "",
                               "Observation:", "```text", row.get("observation", row.get("protocol_error", "")), "```", ""])
        if t.get("partial_generation"):
            transcript.append("Generation interrupted; partial token IDs and prompt remain in trajectory.json.")
        # This is a rendered view; trajectory.json retains the verbatim model/tool bytes.
        rendered = "\n".join(transcript)
        (path.parent / "TRANSCRIPT.md").write_text("\n".join(line.rstrip() for line in rendered.splitlines()).rstrip() + "\n")
    diagnostic = read(args.diagnostic / "receipt.json") if args.diagnostic else None
    diagnostic_stderr = args.diagnostic / "execution/jobdir/runs/run0/stderr/rank0.txt" if args.diagnostic else None
    if diagnostic_stderr and diagnostic_stderr.is_file():
        diagnostic["captured_stderr_tail"] = diagnostic_stderr.read_text().splitlines()[-3:]
    checks = read(args.checks / "config.json") if args.checks else None
    if checks:
        assert checks["returncode"] == 0
        checks["terminal_output"] = (args.checks / "tests.txt").read_text()
    summary = {"source": "executed", "version": "masked_mean_v1",
        "research_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "controls": [{"control": r["control"], "case": r["case"], "completed": r["completed"],
                       "semantic_valid": r["semantic_valid"], "wall_s": r["wall_s"],
                       "max_abs": max((s.get("max_abs") or 0 for s in r["per_step"]), default=None)} for r in rows],
        "positive_max_abs": max(s["max_abs"] for r in positive for s in r["per_step"]),
        "scripted_smoke_reward": smoke["result"]["terminal_reward"], "trajectories": metrics,
        "supervised_diagnostic": diagnostic,
        "automated_checks": checks,
        "rollout_groups_sampled": 0, "informative_group_fraction": None,
        "limitations": ["job-serialized rank0 state; fidelity conditional on exact-source security review",
                        "input delivery is trusted; consumption inferred from numerical agreement",
                        "no non-rank0 state attestation", "authored single mechanism, not held-out generalization",
                        "individual episodes, no repair-rate or evidence-benefit estimate", "no policy training"]}
    write(args.out / "summary.json", summary)
    inputs = [args.controls / "config.json", args.controls / "rows.json", args.smoke / "receipt.json", *args.trajectory,
              *([args.diagnostic / "receipt.json"] if args.diagnostic else []),
              *([diagnostic_stderr] if diagnostic_stderr and diagnostic_stderr.is_file() else []),
              *([args.extra_controls / "config.json", args.extra_controls / "rows.json"] if args.extra_controls else []),
              *([args.checks / "config.json", args.checks / "tests.txt"] if args.checks else []), *args.review]
    write(args.out / "config.json", {"source": "executed", "analysis": "recount of fresh local receipts",
          "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
    lines += ["", "| Model revision | Output tokens | Inference s | Prefill s | Decode s | Peak MLX GB |",
              "|---|---|---|---|---|---|"]
    lines += [f"| {m['model_revision']} | {m['generated_tokens']} | {m['inference_s']:.3f} | "
              f"{m['prefill_s']:.3f} | {m['decode_s']:.3f} | {m['peak_mlx_gb']} |" for m in metrics]
    if diagnostic:
        result = diagnostic["result"]
        lines += ["", "## Separate supervised code diagnostic", "",
                  "One complete Qwen-emitted Python block was inserted with indentation only and exact-source "
                  "security review, then executed through the existing sandbox. The model neither applied nor "
                  "requested execution of this code. Its episode reward remains unchanged.", "",
                  f"Completed: {result['completed']}; error class: {result['error_class']}; "
                  f"semantic success: {result['semantic_valid']}."]
        lines += ["", "```text", *diagnostic.get("captured_stderr_tail", []), "```"]
    lines += ["", "No sampled reward groups or learning comparison. No diagnosis-accuracy estimate.", "",
              "The recorded model episodes use the frozen controller identified in each trajectory. "
              "Later rejection of clipped finals/tool-role delimiters is regression-tested for future runs; "
              "these episodes were not rerun or rescored. Reduced-evidence arms were not unlocked.", "",
              "Future work: first require successful autonomous policy trajectories and structural heldouts; "
              "then propose assistant-only quantized LoRA SFT and matched continued SFT. RL needs separately "
              "sampled same-policy groups with semantic reward variation. No training is authorized here.", "",
              "An investigator/reviewer comparison would share the single-agent token, tool and execution "
              "budgets; reviewer messages count against the same totals and reveal no private evidence. "
              "It remains unimplemented.", "", "Limits:", ""]
    lines += ["- " + limit for limit in summary["limitations"]]
    if checks:
        lines += ["", "## Automated verification", "", "```text", checks["terminal_output"].strip(), "```"]
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n")
    for i, review in enumerate(args.review):
        (args.out / f"review_{i}.txt").write_bytes(review.read_bytes())
    for folder in [args.controls, args.smoke, *(p.parent for p in args.trajectory),
                   *([args.diagnostic] if args.diagnostic else []), *([args.extra_controls] if args.extra_controls else [])]:
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
