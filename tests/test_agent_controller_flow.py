"""Controller wiring with scripted test doubles, not model or repair-success evidence.

The real execute/parser/persistence code runs; the model runtime and episode are
fakes. No MLX computation, worker, shell or source-code execution occurs.
"""
import copy
import json
import shutil
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from experiments.local_agent_v1 import run
from experiments.masked_mean_v1 import task


def test_controller_recovers_protocol_then_pauses_approves_resumes_and_observes(monkeypatch, tmp_path):
    fixture, model, out = tmp_path / "fixture", tmp_path / "model", tmp_path / "episode"
    (fixture / "prefix/job").mkdir(parents=True)
    (fixture / "prefix/job/train.py").write_text("x = 1\n")
    model.mkdir()
    actions = []

    class ScriptedEpisode:
        def __init__(self, fixture, out, approved, reduced, max_tool_calls):
            out.mkdir(exist_ok=True)
            self.jd = out / "jobdir"
            if not self.jd.exists():
                shutil.copytree(fixture / "prefix", self.jd)
            self.approved = set(approved)
            self.done, self.result, self.diagnosis = False, None, None
            self.last_execution = None
            self.tool_calls = self.relaunches = self.phase = 0
            self.records, self.phase_records, self.observed_ids = [], [], set()

        def state(self):
            return {"done": self.done, "result": self.result, "diagnosis": self.diagnosis}

        def step(self, action):
            name, args = action["tool"], action["args"]
            actions.append(name)
            self.tool_calls += 1
            if name == "read_file":
                return task.read_source(self.jd), False
            if name == "write_file":
                assert args["path"] == "job/train.py"
                assert run.source_sha(args["content"]) in self.approved
                (self.jd / args["path"]).write_text(args["content"])
                return "recorded", False
            if name == "relaunch":
                assert args == {"resume_from": "none"}
                sha = run.source_sha(task.read_source(self.jd))
                assert sha in self.approved
                self.relaunches += 1
                self.last_execution = {"source": "simulated", "approved_source_sha256": sha}
                return '{"completed": true, "execution_status": "completed", "run_id": "fake-run"}', False
            assert name == "finish"
            self.finish_actual()
            return "scripted controller check complete", True

        def finish_actual(self):
            assert self.last_execution, "controller must not manufacture an execution before finish"
            self.done = True
            self.result = {"source": "simulated", "terminal_reward": 1,
                           "not_model_evidence": True, "status": "scripted_test_complete"}
            return self.result

    def call(name, arguments):
        return "<tool>" + json.dumps({"name": name, "arguments": arguments}) + "</tool>"

    responses = ["<|tool_response>", call("read_file", {"path": "job/train.py"}),
                 call("apply_patch", {"old": "x = 1\n", "new": "x = 2\n"}),
                 call("relaunch", {"resume_from": "none"}), call("finish", {})]
    prompts, loads = [], []

    class Tokenizer:
        has_tool_calling = True
        tool_call_start, tool_call_end = "<tool>", "</tool>"

        def apply_chat_template(self, messages, **kwargs):
            prompts.append(copy.deepcopy(messages))
            return json.dumps(messages)

        def encode(self, prompt, **kwargs):
            return [0, 1]

        def decode(self, token_ids, **kwargs):
            return responses[token_ids[0]]

        def tool_parser(self, payload, tools):
            return json.loads(payload)

    mlx, mx, lm, sampling = (ModuleType(n) for n in ("mlx", "mlx.core", "mlx_lm", "mlx_lm.sample_utils"))
    mx.random = NS(seed=lambda _: None)
    mx.eval = mx.synchronize = mx.clear_cache = lambda *a: None
    mx.get_active_memory = mx.get_cache_memory = lambda: 0
    mlx.core = mx
    tokenizer = Tokenizer()

    def load(path, **kwargs):
        loads.append(path)
        return NS(parameters=lambda: []), tokenizer

    generated = iter(range(len(responses)))

    def generate(*args, **kwargs):
        yield NS(token=next(generated), generation_tokens=1, prompt_tokens=2,
                 prompt_tps=1, generation_tps=1, peak_memory=0, finish_reason="stop")

    lm.load, lm.stream_generate = load, generate
    sampling.make_sampler = lambda **kwargs: object()
    for name, module in (("mlx", mlx), ("mlx.core", mx), ("mlx_lm", lm), ("mlx_lm.sample_utils", sampling)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(run.importlib.metadata, "version", lambda _: "test-double")
    monkeypatch.setattr(run, "psutil", NS(virtual_memory=lambda: NS(available=16 * 1024**3),
        swap_memory=lambda: NS(used=0), Process=lambda: NS(memory_info=lambda: NS(rss=0))))
    monkeypatch.setattr(task, "RepairEpisode", ScriptedEpisode)
    monkeypatch.setenv("TD_KEY", "state-recovery-demo-key")
    args = NS(task="masked_mean_v1", fixture=fixture, model=model, out=out, turns=16,
              temperature=0.2, seed=0, reduced=False, approve_hash=None)
    run.execute(args)
    pending = json.loads((out / "trajectory.json").read_text())
    assert pending["status"] == "pending_security_review"
    assert pending.get("result") is None
    assert task.read_source(out / "jobdir") == "x = 1\n"
    assert actions == ["read_file"]
    assert pending["rows"][0]["protocol_valid"] is False
    args.approve_hash = "not-the-approved-source"
    with pytest.raises(run.InputError, match="approval"):
        run.execute(args)
    assert json.loads((out / "trajectory.json").read_text()) == pending
    args.approve_hash = pending["pending"]["sha256"]
    run.execute(args)
    final = json.loads((out / "trajectory.json").read_text())
    assert final["status"] == "finished" and final["pending"] is None
    assert final["result"]["not_model_evidence"] and final["result"]["source"] == "simulated"
    assert actions == ["read_file", "write_file", "relaunch", "finish"]
    assert final["relaunches"] == 1 and len(loads) == 2
    assert task.read_source(out / "jobdir") == "x = 2\n"
    assert any(m.get("name") == "relaunch" and "fake-run" in m.get("content", "") for m in prompts[-1])
    assert any(m.get("name") == "apply_patch" and m.get("content") == "recorded" for m in prompts[3])
