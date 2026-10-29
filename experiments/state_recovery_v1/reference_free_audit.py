"""Bounded saved-evidence assay and two trust-boundary falsifications."""
import argparse
import copy
import hashlib
import json
import os
import platform
import zipfile
from pathlib import Path

import numpy as np
import torch

from jobs import trusted

from .oracle import advance
from .reference_free import KEYS, _read_capture, check, history_check, recipe, transition

ACTIONS = ("resume", "restore_sampler", "restore_optimizer", "restore_full", "older", "scratch")
ARCHIVE_SHA = "0579811fa306765a6a5a5b1c279efaca1acdbf4e123235dbd336af4f80bc1671"


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def extract_inputs(archive, out):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise ValueError("frozen archive checksum mismatch")
    with zipfile.ZipFile(archive) as z:
        names = []
        for n in z.namelist():
            if Path(n).is_absolute() or ".." in Path(n).parts:
                raise ValueError("unexpected archive path")
            if n == "config.json" or n.startswith(("base/prefix/", "base/prefix.trusted/")) or any(
                    n.startswith(f"{c}/{a}/jobdir/") or n.startswith(f"{c}/{a}/jobdir.trusted/")
                    for c in ("case0", "case1") for a in ACTIONS):
                if "_private" in Path(n).parts:
                    raise ValueError("private inputs forbidden")
                names.append(n)
        for n in names:
            z.extract(n, out)
    return names


def falsifications(root):
    cfg, schedule = recipe(root)
    jd = root / "base/prefix"
    caps = trusted.ledger(jd)[0]["captures"]
    anchor = _read_capture(jd, next(c for c in caps if c["step"] == 4))
    poisoned = {k: np.asarray(anchor[k], dtype=np.float64).copy() for k in KEYS}
    poisoned["model.fc1.weight"][0, 0] += .1
    bad8 = advance(poisoned, 4, 8, cfg, schedule)
    bad12 = advance(bad8, 8, 12, cfg, schedule)
    pairs = [transition(poisoned, bad8, 4, 8, cfg, schedule),
             transition(bad8, bad12, 8, 12, cfg, schedule)]
    pinned = transition(poisoned, bad8, 4, 8, cfg, schedule, pinned_anchor=anchor)
    correct12 = advance(anchor, 4, 12, cfg, schedule)
    difference = max(float(np.abs(bad12[k] - correct12[k]).max()) for k in KEYS)
    assert all(r["ok"] for r in pairs) and not pinned["ok"] and difference > 1e-5
    records = trusted.ledger(root / "case0/restore_full/jobdir")
    removed = copy.deepcopy(records)
    del removed[-1]["served"]["0"][0]
    missing = history_check(removed, cfg)
    falsified = copy.deepcopy(records)
    falsified[-1]["served"]["0"][0][2][0] += 2
    conflict = history_check(falsified, cfg)
    assert missing["missing"] and conflict["conflicts"]
    return {"source": "simulated", "execution": "checker executed on constructed in-memory counterexamples; no new training job",
            "poisoned_anchor": {"relative_transition_checks": pairs,
                "pinned_original_anchor_check": pinned, "final_max_abs_from_original": difference,
                "false_acceptance_of_original_trajectory_claim": True,
                "violation_of_conditional_contract": False},
            "missing_history": missing, "falsified_history": conflict,
            "checkpoint_bytes_changed_for_history_controls": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True,
                    help="old recount read ONLY after all checker calls, never supplied to them")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists() or os.environ.get("TD_KEY") != "state-recovery-demo-key":
        ap.error("require new output path and explicit public demo TD_KEY")
    args.out.mkdir(parents=True)
    inputs = args.out / "inputs"
    names = extract_inputs(args.archive, inputs)
    # No reference, verdict, oracle or receipt file is extracted into checker inputs.
    assert not any("reference" in Path(n).parts or Path(n).name in
                   ("receipt.json", "oracle.json", "summary.json", "observations.json") for n in names)
    write(args.out / "config.json", {"source": "executed", "version": "reference_free_v1",
          "input_archive_sha256": ARCHIVE_SHA, "input_files": names,
          "platform": platform.platform(), "python": platform.python_version(),
          "torch": torch.__version__, "numpy": np.__version__,
          "mode": "saved-artifact CPU analysis; no distributed job or reference run",
          "public_demo_key": "state-recovery-demo-key", "absolute_tolerance": 1e-5})
    rows = []
    for case in ("case0", "case1"):
        for action in ACTIONS:
            row = check(inputs, case, action)
            rows.append(row)
            print(case, action, row["status"], row["max_abs"], f'{row["wall_s"]:.6f}s', flush=True)
    # Persist checker outputs before reading any old labels, even on disagreement.
    write(args.out / "matrix.json", {"source": "executed", "rows": rows,
          "historical_comparison_pending": True, "timing_imports_excluded": True,
          "cache_order_not_counterbalanced": True})
    # This separate comparison is historical agreement, not checker ground truth.
    baseline = json.loads(args.baseline.read_text())
    previous = {(r["case"], r["action"]): r for r in baseline["rows"]}
    for row in rows:
        old = previous[row["case"], row["action"]]
        row["agrees_with_historical_numpy_oracle"] = row["accepted"] == old["semantic_valid"]
        row["historical_errors_exact"] = old["oracle"]
    write(args.out / "matrix.json", {"source": "executed", "rows": rows,
          "all_twelve_agree": all(r["agrees_with_historical_numpy_oracle"] for r in rows),
          "timing_imports_excluded": True, "cache_order_not_counterbalanced": True})
    write(args.out / "falsifications.json", falsifications(inputs))
    lines = ["# Reference-free saved-evidence detection", "", "source: executed", "",
             "| case | recovery | complete | hashes | reference-free | prior oracle agrees | max abs | max relative* | wall s |",
             "|---|---|---|---|---|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f'| {r["case"]} | {r["action"]} | {r["completed"]} | {r["manifest_hashes_valid"] and r["capture_hashes_valid"]} | {r["status"]} | {r["agrees_with_historical_numpy_oracle"]} | {r["max_abs"]:.17g} | {r["max_relative_floor_1e_12"]:.17g} | {r["wall_s"]:.6f} |')
    lines += ["", "*Coordinate relative error uses denominator max(abs(expected), 1e-12); diagnostic only.",
              "Maximum includes original 4→8 and observed overlap checks, not only final tensors.",
              "All acceptance is conditional on trusted checkpoint 4 and the specified input/update recipe."]
    (args.out / "TABLE.md").write_text("\n".join(lines) + "\n")
    if not all(r["agrees_with_historical_numpy_oracle"] for r in rows):
        raise SystemExit("Historical disagreement: primary decisions and diagnostics preserved")
    print("PASS: 12 reference-free saved-artifact audits; historical agreement; two trust-boundary controls")


if __name__ == "__main__":
    main()
