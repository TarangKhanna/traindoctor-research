"""One checkpoint migration incident; reuse Tier B reads and isolated execution."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import torch

from jobs import manifest, safefs, trusted
from jobs.build import base_config, materialize
from jobs.dataserver import salt_for
from jobs.resubmit import resubmit
from traindoctor_env.snapshot import Snapshot, public_records
from traindoctor_env.tools import ACK, Episode

from experiments.state_recovery_v1.oracle import PARAMS, advance
from experiments.state_recovery_v1.reference_free import compare, history_check

NAMES = tuple(k[6:] for k in PARAMS)
LEGACY_RESTORE = '    sd["state"] = {int(k[4:]): {"momentum_buffer": v.clone()} for k, v in state.items() if k.startswith("opt.")}\n'
GOLD_RESTORE = '''    names = {name: i for i, (name, _) in enumerate(model.named_parameters())}
    sd["state"] = {}
    for key, value in state.items():
        if key.startswith("opt."):
            index = int(key[4:])
        elif key.startswith("optimizer."):
            index = names[key[len("optimizer."):]]
        else:
            continue
        sd["state"][index] = {"momentum_buffer": value.clone()}
'''


def read_source(jd):
    return safefs.read_bytes(jd, "job/ckpt.py").decode()


def source_sha(source):
    return hashlib.sha256(source.encode()).hexdigest()


def canonical(state):
    out = {k: v for k, v in state.items() if not k.startswith("optimizer.")}
    for i, name in enumerate(NAMES):
        if f"optimizer.{name}" in state:
            out[f"opt.{i}"] = state[f"optimizer.{name}"]
    return out


def captured(jd, cap):
    blob = (trusted.trusted_dir(jd) / cap["file"]).read_bytes()
    if hashlib.sha256(blob).hexdigest() != cap["sha256"]:
        raise ValueError("captured checkpoint hash mismatch")
    return canonical(torch.load(io.BytesIO(blob), weights_only=True))


def anchor(fixture, step=8, legacy=False):
    jd = fixture / ("legacy" if legacy else "prefix")
    cap = next(c for c in trusted.ledger(jd)[0]["captures"] if c["step"] == step)
    return captured(jd, cap)


def grade(jd, fixture, approved_sha):
    records = trusted.ledger(jd)
    rec = records[-1]
    cfg = base_config(2, 777, 5, 5, 0)
    try:
        extra_sources_ok = all(safefs.read_bytes(jd, "job/" + name) ==
                               safefs.read_bytes(fixture / "prefix", "job/" + name)
                               for name in ("train.py", "data.py", "config.py", "config.yaml"))
        actual_sha = source_sha(read_source(jd))
    except (OSError, ValueError):
        extra_sources_ok, actual_sha = False, None
    stable = actual_sha == approved_sha
    caps = rec.get("captures", [])
    final = next((c for c in reversed(caps) if c["step"] == cfg["steps"]), None)
    try:
        manifest.validate(jd, manifest.load(jd))
        for r in records:
            for cap in r["captures"]:
                captured(jd, cap)
        integrity = bool(manifest.load(jd))
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, manifest.ManifestError):
        integrity = False
    history = history_check(records, cfg)
    comparison = compare(captured(jd, final), advance(anchor(fixture), 8, cfg["steps"], cfg, salt_for(777))) \
        if final and integrity else {"ok": False, "reason": "missing_final_capture"}
    valid = bool(rec["completed"] and rec["sandboxed"] and integrity and stable and extra_sources_ok
                 and comparison["ok"] and history["ok"])
    return {"source": "executed", "completed": rec["completed"], "semantic_valid": valid,
            "comparison": comparison, "history": history, "hashes_valid": integrity,
            "source_stable": stable, "other_sources_unchanged": extra_sources_ok,
            "sandboxed": rec["sandboxed"], "run_id": rec["run_id"],
            "approved_source_sha256": approved_sha, "actual_source_sha256": actual_sha,
            "start_step": rec["start_step"], "acked": rec["acked"]}


def continuation(fixture, out, source, legacy=False):
    """Only trusted host code calls this; edited code is never imported here."""
    jd = out / "jobdir"
    out.mkdir(parents=True, exist_ok=False)
    trusted.copy_job(fixture / ("legacy" if legacy else "prefix"), jd)
    safefs.write_bytes(jd, "job/ckpt.py", source.encode())
    start = 4 if legacy else 8
    entries = [e for e in manifest.load(jd) if e["step"] <= start]
    manifest.path(jd).write_text(json.dumps(entries) + "\n")
    resume = f"ck{start:04d}"
    blob = safefs.read_bytes(jd, f"ckpt/{resume}/state.pt")
    before = source_sha(read_source(jd))
    result = resubmit(jd, resume, sandbox=True, salt=salt_for(777))
    if result.run_id is None or result.status == "refused":
        raise RuntimeError("infrastructure refused the declared restart")
    verdict = grade(jd, fixture, before)
    verdict.update(delivered_checkpoint_sha256=hashlib.sha256(blob).hexdigest(),
                   execution_status=result.status, error_class=result.error_class)
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    return verdict


class RepairEpisode(Episode):
    """A single experiment's grading override, not a new agent framework."""

    def __init__(self, fixture, out, approved, reduced=False):
        self.fixture, self.out = Path(fixture), Path(out)
        self.jd, self.approved, self.reduced = self.out / "jobdir", set(approved), reduced
        if not self.jd.exists():
            self.out.mkdir(parents=True, exist_ok=True)
            trusted.copy_job(self.fixture / "baseline/jobdir", self.jd)
        super().__init__(max_tool_calls=12, max_relaunches=2)
        snapshot = Snapshot(self.out, 777, tuple(public_records(self.jd, 777)), {}, [], {})
        self.reset(snapshot)
        self.last_execution = None

    @property
    def root(self):
        return self.jd

    def _apply_fix(self, name, args):
        if name == "write_file":
            if args["path"] != "job/ckpt.py" or source_sha(args["content"]) not in self.approved:
                return "error: exact patch security approval required"
            safefs.write_bytes(self.jd, args["path"], args["content"].encode())
            return ACK
        if name == "relaunch":
            source = read_source(self.jd)
            sha = source_sha(source)
            if sha not in self.approved:
                return "error: source hash is not approved"
            resume = args.get("resume_from") or "ck0008"
            if resume not in ("ck0004", "ck0008", "none") or args.get("chaos"):
                return "error: unsupported restart request"
            start = {"none": 0, "ck0004": 4, "ck0008": 8}[resume]
            manifest.path(self.jd).write_text(json.dumps([e for e in manifest.load(self.jd) if e["step"] <= start]) + "\n")
            blob = safefs.read_bytes(self.jd, f"ckpt/{resume}/state.pt") if start else b""
            result = resubmit(self.jd, resume, sandbox=True, salt=salt_for(777))
            if result.run_id is None or result.status == "refused":
                raise RuntimeError("infrastructure refused the declared restart")
            self.last_execution = grade(self.jd, self.fixture, sha)
            self.last_execution.update(execution_status=result.status, error_class=result.error_class,
                                       delivered_checkpoint_sha256=hashlib.sha256(blob).hexdigest() if start else None)
            self.phase += 1
            self.records = public_records(self.jd, 777, self.phase)
            self.phase_records.append(self.records)
            return ACK
        return "error: disabled intervention"

    def checkpoint_summary(self, cid="ck0008"):
        if self.reduced:
            return "checkpoint tensor summaries unavailable in this evidence condition"
        if cid not in ("ck0004", "ck0008"):
            return "error: unknown checkpoint"
        state = torch.load(io.BytesIO(safefs.read_bytes(self.jd, f"ckpt/{cid}/state.pt")), weights_only=True)
        return json.dumps({"checkpoint": cid, "tensors": {k: {"shape": list(v.shape),
                           "sum": float(v.double().sum()), "l2": float(v.double().norm())}
                           for k, v in state.items()}}, sort_keys=True)

    def finish_actual(self):
        if self.done and self.result is not None and "terminal_reward" in self.result:
            return self.result
        if self.done and (self.result or {}).get("status") == "infrastructure_error":
            self.result["terminal_reward"] = None
            return self.result
        termination = (self.result or {}).get("status")
        sha = source_sha(read_source(self.jd))
        actual = self.last_execution
        hidden = []
        changed = read_source(self.jd) != read_source(self.fixture / "prefix")
        bound = bool(actual and actual.get("approved_source_sha256") == sha and actual.get("actual_source_sha256") == sha)
        if actual and actual["semantic_valid"] and bound and changed and sha in self.approved:
            source = read_source(self.jd)
            for label, legacy in (("current", False), ("legacy", True)):
                hidden.append(continuation(self.fixture, self.out / f"hidden_{label}", source, legacy))
        success = bool(actual and actual["semantic_valid"] and changed and len(hidden) == 2
                       and all(r["semantic_valid"] for r in hidden))
        self.result = {"source": "executed", "terminal_reward": int(success),
                       "actual_continuation": actual, "code_changed": changed, "hidden_restarts": hidden,
                       "final_patch_bound_to_actual_execution": bound,
                       "diagnosis": self.diagnosis, "termination_reason": termination,
                       "status": "verified_repair" if success else "unsuccessful_repair"}
        self.done = True
        return self.result

    def _dispatch(self, name, args):
        if name == "finish":
            result = self.finish_actual()
            return json.dumps({"status": result["status"], "terminal_reward": result["terminal_reward"]}), True
        result = super()._dispatch(name, args)
        if name == "read_file" and args.get("path", "").strip("/") == "job/ckpt.py":
            return "source_sha256=" + source_sha(read_source(self.jd)) + "\n" + result[0], result[1]
        if name == "relaunch" and result[0] == ACK and self.last_execution:
            return json.dumps({"completed": self.last_execution["completed"],
                               "status": self.last_execution["execution_status"],
                               "run_id": self.last_execution["run_id"]}), False
        return result


def build(fixture, legacy_jd):
    fixture.mkdir(parents=True, exist_ok=False)
    cfg = base_config(2, 777, 5, 5, 0)
    jd = fixture / "prefix"
    materialize(jd, cfg)
    source = read_source(jd)
    old = 'state.update({f"opt.{i}": opt.state[p]["momentum_buffer"].detach().clone()\n                      for i, p in enumerate(model.parameters())})'
    new = 'state.update({f"optimizer.{name}": opt.state[p]["momentum_buffer"].detach().clone()\n                      for name, p in model.named_parameters()})'
    assert source.count(old) == 1
    safefs.write_bytes(jd, "job/ckpt.py", source.replace(old, new).encode())
    result = resubmit(jd, kill=(0, 9), sandbox=True, salt=salt_for(777))
    assert not result.ok and trusted.ledger(jd)[0]["kill_done"]
    trusted.copy_job(legacy_jd, fixture / "legacy")
    assert all(torch.equal(v, anchor(fixture, 4, legacy=True)[k]) for k, v in anchor(fixture, 4).items())
    baseline = continuation(fixture, fixture / "baseline", read_source(jd))
    assert baseline["completed"] and not baseline["semantic_valid"]
    return baseline
