"""Reference-free *conditional* transition audit for the existing fixed MLP.

Not a general verifier: checkpoint 4, the data recipe and the update specification
are trusted. No reference trajectory or prior correctness verdict is read here.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from jobs import manifest, trusted
from jobs.build import base_config
from jobs.dataserver import DataSpec, SaltSchedule, salt_for, schedule_ids

from .oracle import PARAMS, TOLERANCE, advance

KEYS = (*PARAMS, *(f"opt.{i}" for i in range(4)), "step", "sampler")
LIMITS = ["trusted checkpoint-4 anchor; initialization to step 4 not replayed",
          "input values depend on trusted data-service implementation and demo key/recipe",
          "no per-rank load-time checkpoint-byte attestation",
          "fixed LR; no scheduler or post-initialization stochastic model RNG"]


def compare(actual, expected):
    """Absolute-only acceptance; relative metrics are diagnostics near zeros."""
    if set(actual) != set(expected) or set(actual) != set(KEYS):
        return {"ok": False, "reason": "state_schema", "max_abs": None,
                "max_relative_floor_1e_12": None, "normalized_inf": None}
    errors, relatives, normalized, failed = [], [], [], []
    for key in KEYS:
        a, e = np.asarray(actual[key]), np.asarray(expected[key])
        if a.shape != e.shape or not np.isfinite(a).all() or not np.isfinite(e).all():
            return {"ok": False, "reason": "shape_or_nonfinite", "key": key,
                    "max_abs": None, "max_relative_floor_1e_12": None, "normalized_inf": None}
        d = np.abs(a.astype(np.float64) - e.astype(np.float64))
        err = float(d.max())
        errors.append(err)
        relatives.append(float((d / np.maximum(np.abs(e), 1e-12)).max()))
        normalized.append(err / max(float(np.abs(e).max()), 1e-12))
        bad = not np.array_equal(a, e) if key in ("step", "sampler") else err > TOLERANCE
        if bad:
            failed.append(key)
    return {"ok": not failed, "reason": "state_difference" if failed else None,
            "failed_components": failed, "max_abs": max(errors),
            "tensor_max_abs": max(errors[:-2]), "discrete_state_max_abs": max(errors[-2:]),
            "max_relative_floor_1e_12": max(relatives), "normalized_inf": max(normalized)}


def transition(anchor, target, start, end, cfg, schedule, pinned_anchor=None):
    if pinned_anchor is not None and any(
            not np.array_equal(np.asarray(anchor[k]), np.asarray(pinned_anchor[k])) for k in KEYS):
        return {"ok": False, "reason": "anchor_commitment"}
    return compare(target, advance(anchor, start, end, cfg, schedule))


def history_check(records, cfg):
    """Validate consumed logical inputs, never bless the supplied schedule itself."""
    spec = DataSpec(cfg["world_size"], cfg["data"]["num_samples"], cfg["batch_size"], cfg["seed"])
    history = {str(rank): {} for rank in range(spec.world)}
    missing, conflicts = [], []
    for rec in records:
        end = cfg["steps"] if rec["completed"] else max(
            [rec["start_step"], *(c["step"] for c in rec["captures"])])
        for rank in range(spec.world):
            r = str(rank)
            history[r] = {k: v for k, v in history[r].items() if k < rec["start_step"]}
            if rec.get("acked", {}).get(r, -1) < end - rec["start_step"]:
                missing.append([r, "acknowledgements"])
            windows = {}
            for epoch, index, ids, window in rec.get("served", {}).get(r, []):
                step = rec["start_step"] + window
                if step >= end:
                    continue  # uncommitted tail of a failed attempt
                value = (epoch, index, ids)
                if step in windows and windows[step] != value:
                    conflicts.append([r, step, "conflicting duplicate"])
                windows[step] = value
            history[r].update(windows)
    for rank in range(spec.world):
        for step in range(cfg["steps"]):
            epoch, index = divmod(step, spec.steps_per_epoch)
            actual = history[str(rank)].get(step)
            if actual is None:
                missing.append([str(rank), step])
            elif actual != (epoch, index, schedule_ids(spec, rank, epoch, index)):
                conflicts.append([str(rank), step, "intended sample order"])
    return {"ok": not missing and not conflicts, "missing": missing, "conflicts": conflicts}


def recipe(root):
    if os.environ.get("TD_KEY") != "state-recovery-demo-key":
        raise ValueError("require the explicit public demo TD_KEY; never consult an owner key")
    raw = json.loads((Path(root) / "config.json").read_text())
    if raw["version"] != "state_recovery_v1" or raw["seed"] != 777 or raw["steps"] != 12:
        raise ValueError("only the frozen state-recovery-v1 demonstration is supported")
    cfg = base_config(2, raw["seed"], 5, 5, 0)
    base = salt_for(raw["seed"])
    return cfg, SaltSchedule(base, hashlib.sha256(base + b"|state-recovery-v1").digest(), 8)


def _read_capture(jd, cap):
    raw = (trusted.trusted_dir(jd) / cap["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != cap["sha256"]:
        raise ValueError("capture_byte_integrity")
    import io
    return torch.load(io.BytesIO(raw), weights_only=True)


def _source(jd):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (jd / "job").glob("*.py")}


def _missing(case, action, reason, started):
    return {"source": "executed", "case": case, "action": action,
            "status": "insufficient_evidence", "accepted": False, "completed": None,
            "manifest_hashes_valid": None, "capture_hashes_valid": None,
            "capture_hash_checks": 0, "failures": [], "insufficient": [reason],
            "component_checks": [], "replay_checks": [], "tolerance_abs": TOLERANCE,
            "max_abs": None, "tensor_max_abs": None, "discrete_state_max_abs": None,
            "max_relative_floor_1e_12": None, "normalized_inf": None,
            "limitations": LIMITS, "wall_s": time.perf_counter() - started}


def check(root, case, action):
    """Audit one saved execution; filesystem inputs contain no reference/receipts."""
    started = time.perf_counter()
    root = Path(root)
    cfg, schedule = recipe(root)
    anchor_jd, jd = root / "base/prefix", root / case / action / "jobdir"
    records = trusted.ledger(jd)
    anchor_records = trusted.ledger(anchor_jd)
    if not records or not anchor_records:
        return _missing(case, action, "required_ledger_missing_or_empty", started)
    required = {"captures", "completed", "start_step", "served", "acked"}
    if any(not required.issubset(r) for r in [*records, anchor_records[0]]):
        return _missing(case, action, "required_ledger_fields_missing", started)
    anchor_record = anchor_records[0]
    last = records[-1]
    failures, insufficient, checked, component_checks, replay_checks = [], [], 0, [], []
    try:
        manifest.validate(jd, manifest.load(jd))
        manifest_ok = bool(manifest.load(jd))
    except (ValueError, OSError, manifest.ManifestError):
        manifest_ok = False
    if not manifest_ok:
        failures.append("manifest_integrity")
    try:
        anchors = {c["step"]: _read_capture(anchor_jd, c) for c in anchor_record["captures"]}
        checked += len(anchors)
        if set(anchors) != {4, 8}:
            raise ValueError("missing_anchor")
        for record in records:
            for cap in record["captures"]:
                _read_capture(jd, cap)
                checked += 1
        capture_ok = True
    except (ValueError, OSError):
        capture_ok = False
        failures.append("capture_integrity_or_anchor")
    if _source(jd) != _source(anchor_jd):
        failures.append("source_identity")
    h = history_check(records, cfg)
    if h["missing"]:
        insufficient.append("input_history_missing")
    if h["conflicts"]:
        failures.append("input_history_conflict")
    if not last["completed"]:
        failures.append("execution_incomplete")
    if capture_ok:
        replay_checks.append({"interval": [4, 8], **transition(anchors[4], anchors[8], 4, 8, cfg, schedule)})
        # Only the final run's captures describe its observed overlap/final states.
        selected = [c for c in last["captures"] if c["step"] >= max(4, last["start_step"])]
        if not any(c["step"] == 12 for c in selected):
            insufficient.append("final_capture_missing")
        if last["start_step"] and not any(c["step"] == last["start_step"] for c in selected):
            insufficient.append("observed_resume_checkpoint_missing")
        if last["start_step"] == 0 and not any(c["step"] == 4 for c in selected):
            insufficient.append("scratch_checkpoint4_missing")
        for cap in selected:
            actual = _read_capture(jd, cap)
            step = cap["step"]
            if step not in (4, 8, 12):
                failures.append("unexpected_capture_step")
                continue
            if step in anchors:
                component_checks.append({"step": step, **compare(actual, anchors[step])})
            if step == 4:
                result = compare(actual, anchors[4])
            else:
                # Interval expectations are conditional on genuine original anchors.
                start = 4 if step == 8 else 8
                result = transition(anchors[start], actual, start, step, cfg, schedule)
            replay_checks.append({"interval": [4 if step <= 8 else 8, step], **result})
        if any(not r["ok"] for r in component_checks):
            failures.append("component_consistency")
        if any(not r["ok"] for r in replay_checks):
            failures.append("anchored_transition_consistency")
    status = "reject" if failures else "insufficient_evidence" if insufficient else "accept"
    numeric = [r for r in replay_checks if r.get("max_abs") is not None]
    return {"source": "executed", "case": case, "action": action, "status": status,
            "accepted": status == "accept", "completed": last["completed"],
            "manifest_hashes_valid": manifest_ok, "capture_hashes_valid": capture_ok,
            "capture_hash_checks": checked, "failures": sorted(set(failures)),
            "insufficient": insufficient, "component_checks": component_checks,
            "replay_checks": replay_checks, "history": h, "tolerance_abs": TOLERANCE,
            "max_abs": max((r["max_abs"] for r in numeric), default=None),
            "tensor_max_abs": max((r["tensor_max_abs"] for r in numeric), default=None),
            "discrete_state_max_abs": max((r["discrete_state_max_abs"] for r in numeric), default=None),
            "max_relative_floor_1e_12": max((r["max_relative_floor_1e_12"] for r in numeric), default=None),
            "normalized_inf": max((r["normalized_inf"] for r in numeric), default=None),
            "legacy_resume_sha_is_load_attestation": False, "limitations": LIMITS,
            "wall_s": time.perf_counter() - started}
