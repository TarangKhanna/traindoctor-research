"""Paused-run provenance checks; no model inference or job execution."""
import json
from types import SimpleNamespace as NS

import pytest

from experiments.local_agent_v1 import run


def paused(tmp_path, monkeypatch):
    fixture, model, out = [tmp_path / n for n in ("fixture", "model", "episode")]
    (fixture / "prefix/job").mkdir(parents=True)
    (fixture / "prefix/job/ckpt.py").write_text("x = 1\n")
    (fixture / "prefix.trusted").mkdir()
    (fixture / "prefix.trusted/ledger.jsonl").write_text('{}\n')
    model.mkdir(); out.mkdir()
    identity = run.resume_identity(fixture, run.checkpoint_task.__file__)
    state = dict(model_path=str(model), reduced=False, task="checkpoint", approved=[],
                 rows=[], pending={"source": "x = 2\n", "sha256": run.source_sha("x = 2\n")},
                 status="pending_security_review", resume_identity=identity,
                 controller_sha256=identity["controller_sha256"], task_sha256=identity["task_sha256"])
    path = out / "trajectory.json"
    path.write_text(json.dumps(state))
    monkeypatch.setenv("TD_KEY", "state-recovery-demo-key")
    return NS(fixture=fixture, model=model, out=out, reduced=False, approve_hash=None), state


@pytest.mark.parametrize("change", ["fixture_path", "source", "trusted", "add", "remove", "controller", "task", "legacy"])
def test_changed_identity_refuses_before_reconstruction_or_approval(tmp_path, monkeypatch, change):
    args, state = paused(tmp_path, monkeypatch)
    if change == "fixture_path":
        import shutil
        other = tmp_path / "other"; shutil.copytree(args.fixture, other); args.fixture = other
    elif change in ("source", "trusted"):
        path = args.fixture / ("prefix/job/ckpt.py" if change == "source" else "prefix.trusted/ledger.jsonl")
        path.write_text("changed\n")
    elif change == "add":
        (args.fixture / "extra").write_text("added")
    elif change == "remove":
        (args.fixture / "prefix.trusted/ledger.jsonl").unlink()
    elif change in ("controller", "task"):
        source = tmp_path / "changed.py"; source.write_text("# different executed code\n")
        monkeypatch.setattr(run if change == "controller" else run.checkpoint_task, "__file__", str(source))
    else:
        state.pop("resume_identity")
        (args.out / "trajectory.json").write_text(json.dumps(state))
    args.approve_hash = state["pending"]["sha256"]
    before = (args.out / "trajectory.json").read_bytes()
    monkeypatch.setattr(run, "RepairEpisode", lambda *a, **k: pytest.fail("must reject before reconstructing"))
    with pytest.raises(run.InputError, match="identity"):
        run.execute(args)
    assert (args.out / "trajectory.json").read_bytes() == before
    assert not (args.out / "jobdir").exists()


def test_identical_paused_episode_remains_reviewable_without_model_load(tmp_path, monkeypatch):
    args, _ = paused(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(run, "RepairEpisode", lambda *a, **k: calls.append(a) or NS(done=False))
    monkeypatch.setattr(run, "psutil", None)
    run.execute(args)
    assert len(calls) == 1


def test_completed_legacy_receipt_is_not_modified_or_resumed(tmp_path, monkeypatch):
    args, state = paused(tmp_path, monkeypatch)
    state.pop("resume_identity"); state["result"] = {"terminal_reward": 0}
    path = args.out / "trajectory.json"; path.write_text(json.dumps(state)); before = path.read_bytes()
    args.fixture = tmp_path / "absent"
    monkeypatch.setattr(run, "RepairEpisode", lambda *a, **k: pytest.fail("terminated episode"))
    run.execute(args)
    assert path.read_bytes() == before


def test_fixture_symlink_is_rejected(tmp_path):
    real = tmp_path / "real"; real.mkdir()
    link = tmp_path / "linked"; link.symlink_to(real, target_is_directory=True)
    with pytest.raises(run.InputError, match="fixture"):
        run.resume_identity(link, run.checkpoint_task.__file__)
