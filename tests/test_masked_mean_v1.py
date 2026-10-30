"""Task-contract tests; live control execution is an explicit separate gate."""
from pathlib import Path

import numpy as np
import pytest
import torch

from jobs.dataserver import DataSpec, batch_tensors, salt_for, schedule_ids
from experiments.masked_mean_v1 import task
from experiments.masked_mean_v1.controls import controls


@pytest.mark.parametrize("case", tuple(task.CASES))
def test_numpy_oracle_matches_global_valid_target_mean(case, monkeypatch):
    monkeypatch.setenv("TD_KEY", "state-recovery-demo-key")
    cfg = task.config(case)
    wanted = task.expected(cfg)
    with torch.random.fork_rng():
        torch.manual_seed(cfg["seed"])
        layers = torch.nn.Sequential(torch.nn.Linear(8, 12), torch.nn.ReLU(), torch.nn.Linear(12, 4))
    opt = torch.optim.SGD(layers.parameters(), lr=cfg["lr"], momentum=0.9)
    spec = DataSpec(2, 96, 4, 777)
    for step in range(4):
        batches = [batch_tensors(salt_for(777), 777, schedule_ids(spec, r, 0, step)) for r in range(2)]
        x = torch.cat([v[0][:cfg["valid_counts"][str(step)][str(r)]] for r, v in enumerate(batches)])
        y = torch.cat([v[1][:cfg["valid_counts"][str(step)][str(r)]] for r, v in enumerate(batches)])
        opt.zero_grad()
        torch.nn.functional.mse_loss(layers(x), y).backward()
        opt.step()
        for i, (key, p) in enumerate(zip(task.PARAMS, layers.parameters())):
            np.testing.assert_allclose(p.detach().numpy(), wanted[step][key], atol=1e-6, rtol=0)
            np.testing.assert_allclose(opt.state[p]["momentum_buffer"].numpy(), wanted[step][f"opt.{i}"], atol=1e-6, rtol=0)


def test_authored_controls_compile_and_correct_forms_differ():
    variants = controls()
    for name, text in variants.items():
        compile(text, name, "exec")
    assert variants["correct_loss"] != variants["correct_gradient"]
    assert "clamp_min(1)" in variants["unchanged"]


def test_premature_finish_never_executes(monkeypatch):
    episode = task.RepairEpisode.__new__(task.RepairEpisode)
    episode.done, episode.result = False, None
    episode.last_execution, episode.diagnosis = None, None
    episode.jd = Path("unused")
    monkeypatch.setattr(task, "read_source", lambda _: "current")
    monkeypatch.setattr(task, "execute_case", lambda *a: pytest.fail("finish must not fabricate a relaunch"))
    result = episode.finish_actual()
    assert result["terminal_reward"] == 0
    assert not result["final_patch_bound_to_actual_execution"]
    assert episode.finish_actual() == result


def test_infrastructure_failure_stays_null():
    episode = task.RepairEpisode.__new__(task.RepairEpisode)
    episode.done, episode.result = True, {"status": "infrastructure_error"}
    assert episode.finish_actual()["terminal_reward"] is None


def test_current_hash_required_even_if_last_execution_was_correct(monkeypatch):
    episode = task.RepairEpisode.__new__(task.RepairEpisode)
    episode.done, episode.result, episode.diagnosis = False, None, None
    episode.jd = Path("unused")
    episode.last_execution = {"semantic_valid": True, "approved_source_sha256": "old", "actual_source_sha256": "old"}
    monkeypatch.setattr(task, "read_source", lambda _: "new")
    monkeypatch.setattr(task, "execute_case", lambda *a: pytest.fail("stale execution must not trigger hidden checks"))
    assert episode.finish_actual()["terminal_reward"] == 0


def test_exhausted_budget_cannot_run_hidden_checks(monkeypatch):
    episode = task.RepairEpisode.__new__(task.RepairEpisode)
    episode.done, episode.result = True, {"status": "budget_exhausted"}
    monkeypatch.setattr(task, "read_source", lambda *a: pytest.fail("must not regrade after budget exhaustion"))
    assert episode.finish_actual()["terminal_reward"] == 0


@pytest.mark.parametrize("status,error,signal", [("infrastructure_error", None, False),
    ("failed", "INFRASTRUCTURE_ERROR", False), ("failed", "PYTHON_ERROR", True)])
def test_infrastructure_result_with_run_id_is_not_semantic_failure(status, error, signal):
    from types import SimpleNamespace as NS
    result = NS(run_id="run1", status=status, error_class=error,
                job=NS(ranks=[NS(exitcode=-9 if signal else 1, status="error")]))
    with pytest.raises(RuntimeError):
        task.check_execution(result)


def test_model_syntax_failure_remains_a_policy_failure():
    from types import SimpleNamespace as NS
    task.check_execution(NS(run_id="run1", status="failed", error_class="PYTHON_ERROR",
                            job=NS(ranks=[NS(exitcode=1, status="error")])))


def test_new_edit_invalidates_previous_execution(monkeypatch):
    episode = task.RepairEpisode.__new__(task.RepairEpisode)
    episode.approved = {task.source_sha("new")}
    episode.jd, episode.last_execution = Path("unused"), {"semantic_valid": True}
    writes = []
    monkeypatch.setattr(task.safefs, "write_bytes", lambda *args: writes.append(args))
    assert episode._apply_fix("write_file", {"path": "job/train.py", "content": "new"}) == task.ACK
    assert episode.last_execution is None
    assert len(writes) == 1
    assert episode._apply_fix("write_file", {"path": "job/config.yaml", "content": "new"}).startswith("error")


@pytest.mark.parametrize("name,args", [("read_file", {"path": "runs/run0/logs/rank0.jsonl"}),
    ("read_file", {"path": "/runs/run0/logs/rank0.jsonl"}),
    ("read_file", {"path": "./runs/run0/logs/rank0.jsonl"}),
    ("read_file", {"path": "//ckpt/MANIFEST.json"}),
    ("read_file", {"path": "ckpt/MANIFEST.json"}), ("grep_logs", {"pattern": ""}),
    ("collective_trace", {"step": 0})])
def test_reduced_observation_routes_hide_runtime_telemetry(name, args):
    episode = task.RepairEpisode.__new__(task.RepairEpisode)
    episode.reduced = True
    obs, done = episode._dispatch(name, args)
    assert "unavailable" in obs and not done
