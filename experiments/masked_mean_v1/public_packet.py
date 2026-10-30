"""Export/recount this completed study without inference, workers or private paths.

The packet contains deidentified trajectory/receipt copies and original captured
tensor bytes. Recount checks recorded states against the fixed NumPy recipe; it
does not attest the original execution or a rank's live memory independently.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import zipfile
from pathlib import Path

import torch

from experiments.state_recovery_v1.reference_free import compare, history_check
from . import task

VERSION = "masked_mean_public_v1"
CONTROL_NAMES = {"correct_loss", "correct_gradient", "unchanged", "constant_scaling",
                 "visible_count_hardcoded", "skip_update"}
PRIVATE_PATH = re.compile(r"/Users/|/home/|/private/(?:var|tmp)/|/var/folders/")


def read(path):
    return json.loads(path.read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def deidentify(value, replacements):
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        if PRIVATE_PATH.search(value):
            raise ValueError("unmapped host path; do not publish this value")
        return value
    if isinstance(value, list):
        return [deidentify(v, replacements) for v in value]
    if isinstance(value, dict):
        return {deidentify(k, replacements): deidentify(v, replacements) for k, v in value.items()}
    return value


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def metrics(state):
    rows = state["rows"]
    return {"revision": state["revision"], "responses": len(rows),
            "valid_calls": sum(bool(r.get("protocol_valid")) for r in rows),
            "applied_edits": sum("security_review" in r for r in rows),
            "relaunches": state["relaunches"], "reward": state["result"]["terminal_reward"],
            "output_tokens": sum(r["generation_tokens"] for r in rows),
            "inference_s": sum(r["wall_s"] for r in rows),
            "prefill_s": sum(r.get("prefill_s") or 0 for r in rows),
            "decode_s": sum(r.get("decode_s") or 0 for r in rows),
            "peak_mlx_gb": max((r["peak_mlx_gb"] for r in rows), default=None),
            "controller_sha256": state["controller_sha256"],
            "last_finish_reason": rows[-1]["finish_reason"]}


def export(source, out):
    out.mkdir(parents=True, exist_ok=False)
    base = source / "masked_mean_v1_controls_20261030"
    revised = source / "masked_mean_v1_overfit_recheck_20261030"
    rows = read(base / "rows.json")
    replacements = {(r["control"], r["case"]): r for r in read(revised / "rows.json")}
    aliases = [(str(source.parent), "<RESEARCH_CHECKOUT>"),
               (str(source.parent.with_name("masked-mean-v1-env")), "<ENV_CHECKOUT>"),
               (str(Path.home()), "<HOME>")]
    original_hashes = {}

    def receipt(path):
        original_hashes[str(path.relative_to(source))] = digest(path.read_bytes())
        return deidentify(read(path), aliases)

    controls = []
    with zipfile.ZipFile(out / "captures.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for original in rows:
            key = original["control"], original["case"]
            row = replacements.get(key, original)
            origin = revised if key in replacements else base
            ledger = origin / key[0] / key[1] / "jobdir.trusted/ledger.jsonl"
            records = [json.loads(line) for line in ledger.read_text().splitlines()]
            record = next(r for r in records if r["run_id"] == row["run_id"])
            for cap in record["captures"]:
                relative = Path(cap["file"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("invalid captured-state path")
                data = (ledger.parent / relative).read_bytes()
                if digest(data) != cap["sha256"] or b"/Users/" in data or b"/home/" in data:
                    raise ValueError("capture integrity or privacy check failed")
                z.writestr(f"{key[0]}/{key[1]}/{cap['file']}", data)
            controls.append({"row": deidentify(row, aliases), "record": deidentify(record, aliases)})
    model_metrics = {}
    for name in ("qwen", "gemma"):
        state = receipt(source / f"masked_mean_v1_{name}_full_20261030/trajectory.json")
        write(out / f"{name}_trajectory.json", state)
        model_metrics[name] = metrics(state)
    diagnostic_root = source / "masked_mean_v1_qwen_diagnostic_20261030"
    diagnostic = receipt(diagnostic_root / "receipt.json")
    candidate = (diagnostic_root / "pending.py").read_bytes()
    if digest(candidate) != diagnostic["candidate_sha256"]:
        raise ValueError("reviewed diagnostic source changed")
    (out / "qwen_candidate.txt").write_bytes(candidate)
    stderr = diagnostic_root / "execution/jobdir/runs/run0/stderr/rank0.txt"
    diagnostic["stderr_tail"] = deidentify(stderr.read_text().splitlines()[-3:], aliases)
    write(out / "diagnostic.json", diagnostic)
    write(out / "scripted_smoke.json", receipt(source / "masked_mean_v1_smoke_final_20261030/receipt.json"))
    original_hashes["original_controls_rows"] = digest((base / "rows.json").read_bytes())
    original_hashes["revised_overfit_rows"] = digest((revised / "rows.json").read_bytes())
    write(out / "packet.json", {"source": "executed", "version": VERSION,
        "frozen_controller_commit": "e14a0f5", "post_run_fix_commit": "099d3d6",
        "cases": {case: task.config(case) for case in task.CASES}, "controls": controls,
        "models": model_metrics, "original_local_receipt_sha256": original_hashes,
        "redaction": "Host path prefixes replaced by placeholders; numerical values and captured tensor bytes unchanged.",
        "scope": "Recorded author-run episodes and rank0 checkpoints; no independent runtime attestation or model rerun."})
    lines = ["# Masked-loss coding-agent study", "", "Script-generated from the completed local receipts.", "",
        "A working multi-turn harness and an executable source-repair task are implemented. "
        "Successful autonomous repair was not demonstrated. These are individual episodes, not repair rates.", "",
        "| Model | Responses | Valid calls | Applied edits | Relaunches | Reward | Inference s | Prefill s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in model_metrics.items():
        lines.append(f"| {name} | {m['responses']} | {m['valid_calls']} | {m['applied_edits']} | "
                     f"{m['relaunches']} | {m['reward']} | {m['inference_s']:.3f} | {m['prefill_s']:.3f} |")
    maximum = max(s["max_abs"] for c in controls if c["row"]["control"].startswith("correct")
                  for s in c["row"]["per_step"])
    lines += ["", f"Correct controls: maximum per-step absolute error {maximum:.9g}, tolerance 1e-5 (rtol=0).", "",
        "The unchanged source completes all cases, passes balanced/all-valid regressions and fails unequal-count cases. "
        "The revised visible-schedule hardcode passes the visible case and fails other hidden schedules. "
        "The earlier fixed-count control was not a true visible-overfit baseline; original receipts remain local.", "",
        "Qwen made model-selected inspection calls, then emitted a clipped prose/code proposal. Its sole complete "
        "Python block was separately security-reviewed and inserted with indentation only. That supervised diagnostic "
        "failed with AttributeError; it was neither model-applied nor model-requested execution and earns no agent credit.", "",
        "Gemma ended on a tool-response delimiter. Why it emitted that delimiter is unresolved. "
        "The frozen controller prematurely accepted both endings as final. Later protocol corrections were "
        "regression-tested without model reruns; their effect on outcomes is unknown.", "",
        "Reduced-evidence arms were gated off. No same-policy reward groups, model training or evidence-benefit "
        "comparison were run. Two zero rewards do not estimate RL reward variation.", "",
        "## Recount without a model or workers", "", "```bash",
        "TD_KEY=state-recovery-demo-key PYTHONPATH=. python -m experiments.masked_mean_v1.public_packet --check evidence/masked_mean_v1",
        "```", "", "This checks file/capture hashes, independently reconstructs masked NumPy updates for each "
        "recorded control, and recounts model actions. It does not relaunch training or authenticate the original "
        "execution. Captures are job-serialized rank0 state; fidelity is conditional on exact-source safety review. "
        "Input records show delivery; consumption is inferred from numerical agreement. Non-rank0 state is unobserved.", "",
        "## Next experiment (not performed)", "", "Keep this task and verifier fixed. Version a protocol-corrected "
        "baseline separately before changing output budget; report the budget change as a confound. Validate "
        "applicable edits and relaunch/feedback before SFT/RL. Qwen's recorded prefill time motivates a separate "
        "prefix-cache parity/latency investigation; no cache speedup is established here."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    write(out / "manifest.json", {p.name: digest(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()})


def check(root):
    if os.environ.get("TD_KEY") != "state-recovery-demo-key":
        raise ValueError("require the explicit public demonstration key; never consult an owner key")
    if (root / "manifest.json").is_symlink():
        raise ValueError("packet symlink is not allowed")
    manifest = read(root / "manifest.json")
    if set(manifest) != {p.name for p in root.iterdir() if p.is_file() and p.name != "manifest.json"}:
        raise ValueError("packet file set differs from manifest")
    for name, sha in manifest.items():
        if Path(name).name != name or (root / name).is_symlink():
            raise ValueError("invalid packet filename or symlink")
        if digest((root / name).read_bytes()) != sha:
            raise ValueError("packet file integrity failure")
    for path in root.glob("*.json"):
        deidentify(read(path), [])  # reject any host paths still present
    packet = read(root / "packet.json")
    if packet["version"] != VERSION or packet["cases"] != {c: task.config(c) for c in task.CASES}:
        raise ValueError("wrong experiment or update recipe")
    if set(packet["models"]) != {"qwen", "gemma"}:
        raise ValueError("packet model set must contain exactly qwen and gemma")
    keys = [(c["row"]["control"], c["row"]["case"]) for c in packet["controls"]]
    if len(keys) != len(set(keys)) or set(keys) != {(n, c) for n in CONTROL_NAMES for c in task.CASES}:
        raise ValueError("incomplete or duplicate control matrix")
    captures = 0
    with zipfile.ZipFile(root / "captures.zip") as z:
        for control in packet["controls"]:
            row, record = control["row"], control["record"]
            cfg = packet["cases"][row["case"]]
            expected = task.expected(cfg)
            if not record["completed"] or not record["sandboxed"] or not history_check([record], cfg)["ok"]:
                raise ValueError("recorded execution/history failure")
            steps = []
            if sorted(c["step"] for c in record["captures"]) != list(range(1, 5)):
                raise ValueError("missing/duplicate per-step capture")
            for cap in sorted(record["captures"], key=lambda c: c["step"]):
                data = z.read(f"{row['control']}/{row['case']}/{cap['file']}")
                if digest(data) != cap["sha256"]:
                    raise ValueError("captured state integrity failure")
                state = torch.load(io.BytesIO(data), weights_only=True, map_location="cpu")
                actual = compare(state, expected[cap["step"] - 1])
                saved = row["per_step"][cap["step"] - 1]
                if actual["ok"] != saved["ok"] or not math.isclose(actual["max_abs"], saved["max_abs"], abs_tol=1e-10, rel_tol=1e-7):
                    raise ValueError("saved numerical comparison differs from independent recount")
                steps.append(actual["ok"])
                captures += 1
            if bool(all(steps)) != row["semantic_valid"]:
                raise ValueError("semantic success claim differs from independent recount")
    for name, m in packet["models"].items():
        state = read(root / f"{name}_trajectory.json")
        if metrics(state) != m or m["applied_edits"] or m["relaunches"] or m["reward"] != 0:
            raise ValueError("model action/outcome recount differs")
    smoke, diagnostic = read(root / "scripted_smoke.json"), read(root / "diagnostic.json")
    if not smoke["not_model_evidence"] or smoke["result"]["terminal_reward"] != 1:
        raise ValueError("scripted smoke mislabelled")
    if not diagnostic["not_autonomous_agent_execution"] or diagnostic["result"]["completed"] or diagnostic["result"]["error_class"] != "PYTHON_ERROR":
        raise ValueError("diagnostic mislabelled")
    if digest((root / "qwen_candidate.txt").read_bytes()) != diagnostic["candidate_sha256"]:
        raise ValueError("diagnostic source differs from reviewed candidate")
    from .diagnostic import candidate
    _, derived = candidate(read(root / "qwen_trajectory.json"))
    if digest(derived.encode()) != diagnostic["candidate_sha256"]:
        raise ValueError("diagnostic is not the verbatim model fragment with indentation only")
    print(f"PASS: {len(keys)} controls / {captures} captured states independently recounted; "
          f"{len(packet['models'])} recorded model episodes, no autonomous repair; supervised diagnostic remains separate")


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source", type=Path, help="original local results directory (export only)")
    mode.add_argument("--check", type=Path, help="public packet; no local receipts needed")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.source:
        if not args.out:
            ap.error("--source requires a fresh --out directory")
        export(args.source, args.out)
    else:
        check(args.check)


if __name__ == "__main__":
    main()
