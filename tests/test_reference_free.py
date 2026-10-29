"""Contract tests; no distributed jobs, reference trajectories or LLMs."""
import copy
import os
from pathlib import Path

import numpy as np
import pytest

from experiments.state_recovery_v1.reference_free import compare, check, recipe
from experiments.state_recovery_v1.reference_free_audit import falsifications


def evidence():
    root = os.environ.get("TD_REFERENCE_FREE_INPUT")
    if not root:
        pytest.skip("set TD_REFERENCE_FREE_INPUT to the archive's filtered extraction")
    return Path(root)


def test_missing_explicit_demo_key_never_falls_back_to_owner_key(monkeypatch):
    monkeypatch.delenv("TD_KEY", raising=False)
    with pytest.raises(ValueError, match="explicit public demo"):
        recipe(Path("must-not-be-read"))


def test_all_twelve_without_reference_or_verdict_reads(monkeypatch):
    root = evidence()
    monkeypatch.setenv("TD_KEY", "state-recovery-demo-key")
    original = Path.open
    forbidden = {"reference", "_private", "receipt.json", "oracle.json", "summary.json", "observations.json"}

    def guarded(path, *args, **kwargs):
        assert not forbidden.intersection(path.parts), path
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    for case in ("case0", "case1"):
        for action in ("resume", "restore_sampler", "restore_optimizer", "restore_full", "older", "scratch"):
            result = check(root, case, action)
            want = action in ("restore_full", "older", "scratch") or (case, action) in (
                ("case0", "restore_sampler"), ("case1", "restore_optimizer"))
            assert result["accepted"] == want
            assert result["completed"] and result["capture_hashes_valid"] and result["manifest_hashes_valid"]


def test_trust_boundary_counterexamples(monkeypatch):
    monkeypatch.setenv("TD_KEY", "state-recovery-demo-key")
    result = falsifications(evidence())
    assert result["poisoned_anchor"]["false_acceptance_of_original_trajectory_claim"]
    assert not result["poisoned_anchor"]["pinned_original_anchor_check"]["ok"]
    assert result["missing_history"]["missing"]
    assert result["falsified_history"]["conflicts"]


def test_top_level_missing_history_is_inconclusive_and_conflict_rejected(monkeypatch):
    from jobs import trusted
    root = evidence()
    monkeypatch.setenv("TD_KEY", "state-recovery-demo-key")
    original = trusted.ledger
    jd = root / "case0/restore_full/jobdir"
    records = copy.deepcopy(original(jd))
    del records[-1]["served"]["0"][0]
    monkeypatch.setattr(trusted, "ledger", lambda p: records if p == jd else original(p))
    result = check(root, "case0", "restore_full")
    assert result["status"] == "insufficient_evidence" and not result["accepted"]
    records = copy.deepcopy(original(jd))
    records[-1]["served"]["0"][0][2][0] += 2
    assert check(root, "case0", "restore_full")["status"] == "reject"


def test_numerical_contract_rejects_nonfinite_and_schema_change(monkeypatch):
    from experiments.state_recovery_v1.reference_free import _read_capture
    from jobs import trusted
    jd = evidence() / "base/prefix"
    state = _read_capture(jd, trusted.ledger(jd)[0]["captures"][0])
    assert compare(state, state)["ok"]
    changed = {k: np.asarray(v).copy() for k, v in state.items()}
    changed["model.fc1.weight"][0, 0] = np.nan
    assert not compare(changed, state)["ok"]
    changed["model.fc1.weight"][0, 0] = np.inf
    assert not compare(changed, state)["ok"]
    changed = copy.copy(state)
    changed["extra_scheduler"] = np.array([4])
    assert not compare(changed, state)["ok"]


@pytest.mark.parametrize("missing", ["continuation", "anchor", "captures", "scratch4"])
def test_missing_required_ledger_and_scratch_overlap_are_inconclusive(monkeypatch, missing):
    from jobs import trusted
    root = evidence()
    monkeypatch.setenv("TD_KEY", "state-recovery-demo-key")
    original = trusted.ledger
    jd = root / "case0/scratch/jobdir"
    anchor_jd = root / "base/prefix"

    def incomplete(path):
        records = copy.deepcopy(original(path))
        if (missing == "continuation" and path == jd) or (missing == "anchor" and path == anchor_jd):
            return []
        if path == jd and missing == "captures":
            del records[-1]["captures"]
        if path == jd and missing == "scratch4":
            records[-1]["captures"] = [c for c in records[-1]["captures"] if c["step"] != 4]
        return records

    monkeypatch.setattr(trusted, "ledger", incomplete)
    result = check(root, "case0", "scratch")
    assert result["status"] == "insufficient_evidence" and not result["accepted"]


def test_primary_decisions_survive_unavailable_historical_comparison(monkeypatch, tmp_path):
    import json
    import sys
    from experiments.state_recovery_v1 import reference_free_audit as audit
    out = tmp_path / "new-audit"
    monkeypatch.setenv("TD_KEY", "state-recovery-demo-key")
    monkeypatch.setattr(audit, "extract_inputs", lambda *args: [])
    monkeypatch.setattr(audit, "check", lambda root, case, action:
                        {"case": case, "action": action, "status": "accept", "max_abs": 0., "wall_s": 0.})
    monkeypatch.setattr(sys, "argv", ["audit", "--archive", "unused", "--baseline",
                                    str(tmp_path / "missing-baseline"), "--out", str(out)])
    with pytest.raises(FileNotFoundError):
        audit.main()
    primary = json.loads((out / "matrix.json").read_text())
    assert len(primary["rows"]) == 12 and primary["historical_comparison_pending"]
