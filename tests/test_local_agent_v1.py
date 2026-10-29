"""Focused application and reward boundary checks; no model or Gloo execution."""
import json
from pathlib import Path

import pytest
import torch

from experiments.local_agent_v1.run import SPECS, native_call, proposed, tools
from experiments.local_agent_v1.task import RepairEpisode, canonical
from traindoctor_env.tools import _validate


def test_literal_patch_is_unique_bounded_and_not_a_shell_interface():
    assert proposed("a\nb\n", {"old": "b", "new": "c"}) == "a\nc\n"
    for old in ("", "a"):
        with pytest.raises(ValueError):
            proposed("aa", {"old": old, "new": "z"})
    with pytest.raises(ValueError):
        proposed("a", {"old": "a", "new": "x" * 65537})
    assert "shell" not in SPECS and "write_file" not in SPECS
    assert _validate({"old": "a", "new": "b", "path": "../../outside"}, SPECS["apply_patch"])


def test_native_tool_schemas_and_argument_types():
    schemas = tools()
    assert {t["function"]["name"] for t in schemas} == set(SPECS)
    assert all(t["function"]["parameters"]["additionalProperties"] is False for t in schemas)
    assert _validate({"step": True}, SPECS["sample_ids"])
    json.dumps(schemas)


def test_native_blocks_are_not_silently_truncated():
    from types import SimpleNamespace
    tok = SimpleNamespace(tool_call_start="<tool>", tool_call_end="</tool>",
                          tool_parser=lambda payload, _: json.loads(payload))
    block = '<tool>{"name":"job_status","arguments":{}}</tool>'
    assert native_call(tok, block)["name"] == "job_status"
    assert native_call(tok, "No tool requested") is None
    for malformed in (block + block, block.replace("</tool>", "")):
        with pytest.raises(ValueError):
            native_call(tok, malformed)


def test_outer_runtime_failure_persists_an_unscored_result(monkeypatch, tmp_path):
    import sys
    from experiments.local_agent_v1 import run
    out = tmp_path / "unscored"
    monkeypatch.setattr(sys, "argv", ["runner", "--fixture", "unused", "--model", "unused", "--out", str(out)])
    monkeypatch.setattr(run, "execute", lambda args: (_ for _ in ()).throw(RuntimeError("test-only infrastructure fault")))
    with pytest.raises(RuntimeError):
        run.main()
    state = json.loads((out / "trajectory.json").read_text())
    assert state["result"]["terminal_reward"] is None and state["status"] == "infrastructure_error"


def test_bad_security_approval_preserves_pending_trajectory(monkeypatch, tmp_path):
    import sys
    from experiments.local_agent_v1 import run
    out, model = tmp_path / "episode", tmp_path / "model"
    out.mkdir(); model.mkdir()
    path = out / "trajectory.json"
    original = json.dumps({"model_path": str(model), "reduced": False, "status": "pending_security_review",
                           "pending": {"source": "candidate"}})
    path.write_text(original)
    monkeypatch.setenv("TD_KEY", "state-recovery-demo-key")
    monkeypatch.setattr(run, "read_source", lambda p: "original")
    monkeypatch.setattr(sys, "argv", ["runner", "--fixture", "unused", "--model", str(model),
                                    "--out", str(out), "--approve-hash", "wrong"])
    with pytest.raises(SystemExit) as exc:
        run.main()
    assert exc.value.code == 2 and path.read_text() == original


def test_completed_result_is_preserved_before_any_fixture_read(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from experiments.local_agent_v1 import run
    out, model = tmp_path / "episode", tmp_path / "model"
    out.mkdir(); model.mkdir()
    path = out / "trajectory.json"
    original = json.dumps({"result": {"terminal_reward": 1}, "status": "finished"})
    path.write_text(original)
    monkeypatch.setenv("TD_KEY", "state-recovery-demo-key")
    monkeypatch.setattr(run, "read_source", lambda p: pytest.fail("completed episodes must not read fixtures"))
    run.execute(SimpleNamespace(model=model, out=out, fixture=tmp_path / "missing", reduced=False, approve_hash=None))
    assert path.read_text() == original


def test_checkpoint_normalization_is_explicit_name_mapping():
    values = [torch.tensor([i]) for i in range(4)]
    names = ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias")
    named = {f"optimizer.{n}": v for n, v in zip(names, values)}
    assert set(canonical(named)) == {"opt.0", "opt.1", "opt.2", "opt.3"}
    assert all(torch.equal(canonical(named)[f"opt.{i}"], v) for i, v in enumerate(values))


def test_write_requires_exact_source_approval_and_fixed_path(tmp_path):
    episode = object.__new__(RepairEpisode)
    episode.approved = set()
    assert "approval required" in episode._apply_fix("write_file", {"path": "job/ckpt.py", "content": "anything"})
    assert "approval required" in episode._apply_fix("write_file", {"path": "../outside", "content": "anything"})


def test_infrastructure_failure_is_terminal_and_unscored():
    episode = object.__new__(RepairEpisode)
    episode.snap, episode.done, episode.tool_calls, episode.max_tool_calls = object(), False, 0, 12
    episode.relaunches, episode.max_relaunches, episode.calls, episode.diagnosis = 0, 2, [], None
    episode._dispatch = lambda *args: (_ for _ in ()).throw(OSError("injected test-only host failure"))
    _, done = episode.step({"tool": "relaunch", "args": {}})
    assert done and episode.result["status"] == "infrastructure_error"
    assert episode.result["verified_fixed"] is None


def test_final_patch_must_be_the_patch_that_actually_executed(monkeypatch, tmp_path):
    from experiments.local_agent_v1 import task
    episode = object.__new__(RepairEpisode)
    episode.done, episode.result, episode.diagnosis = False, None, None
    episode.jd, episode.fixture, episode.out = tmp_path / "job", tmp_path / "fixture", tmp_path
    episode.approved = {task.source_sha("final-patch")}
    episode.last_execution = {"semantic_valid": True, "approved_source_sha256": task.source_sha("earlier-patch"),
                              "actual_source_sha256": task.source_sha("earlier-patch")}
    monkeypatch.setattr(task, "read_source", lambda p: "final-patch" if p == episode.jd else "original")
    monkeypatch.setattr(task, "continuation", lambda *a, **k: pytest.fail("must not credit stale actual run"))
    result = episode.finish_actual()
    assert result["terminal_reward"] == 0 and not result["final_patch_bound_to_actual_execution"]
