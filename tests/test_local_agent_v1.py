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


@pytest.mark.parametrize("available,growth,low_ram,high_swap", [
    (7 * 1024**3, 0, True, False),
    (9 * 1024**3, 129 * 1024**2, False, True),
])
def test_resource_stop_preserves_trigger_and_null_reward_before_model_load(
        monkeypatch, tmp_path, available, growth, low_ram, high_swap):
    import sys
    from types import SimpleNamespace
    from experiments.local_agent_v1 import run
    out, model = tmp_path / "episode", tmp_path / "model"
    model.mkdir()
    monkeypatch.setenv("TD_KEY", "state-recovery-demo-key")
    monkeypatch.setattr(run, "read_source", lambda _: "unchanged")
    def episode(*a, **k):
        out.mkdir()
        return SimpleNamespace(done=False, state=lambda: {}, tool_calls=0, relaunches=0,
                               last_execution=None, phase=0, records=[], phase_records=[])
    monkeypatch.setattr(run, "RepairEpisode", episode)
    swap_samples = iter((1024**3, 1024**3 + growth))
    monkeypatch.setattr(run, "psutil", SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(available=available),
        swap_memory=lambda: SimpleNamespace(used=next(swap_samples))))
    monkeypatch.setattr(sys, "argv", ["runner", "--fixture", "unused", "--model", str(model),
                                    "--out", str(out)])
    with pytest.raises(RuntimeError, match="local resource guard"):
        run.main()
    state = json.loads((out / "trajectory.json").read_text())
    counters = state["resource_guard_failure"]
    assert counters["available_bytes"] == available and counters["swap_growth_bytes"] == growth
    assert counters["below_ram_floor"] is low_ram
    assert counters["above_swap_growth_limit"] is high_swap
    assert state["result"]["terminal_reward"] is None and state["loads"] == []


def test_bad_security_approval_preserves_pending_trajectory(monkeypatch, tmp_path):
    import sys
    from experiments.local_agent_v1 import run
    out, model = tmp_path / "episode", tmp_path / "model"
    out.mkdir(); model.mkdir()
    path = out / "trajectory.json"
    monkeypatch.setattr(run, "psutil", None)
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
    monkeypatch.setattr(run, "psutil", None)
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


def test_v2_complete_json_fallback_rejects_truncation_extras_and_multiple_calls():
    from types import SimpleNamespace
    tok = SimpleNamespace(tool_call_start='<tool>', tool_call_end='</tool>')
    call = '{"name":"job_status","arguments":{}}'
    assert native_call(tok, call)['name'] == 'job_status'
    assert native_call(tok, '```json\n' + call + '\n```')['name'] == 'job_status'
    for raw in (call[:-1], call + call, call[:-1] + ',"extra":0}', '{"name":"job_status"}',
                '{"name":"job_status","name":"finish","arguments":{}}', '```json\n'+call):
        with pytest.raises(ValueError):
            native_call(tok, raw)
    assert native_call(tok, 'I cannot repair this job.') is None


def test_v2_line_edits_require_current_hash_and_valid_bounds():
    from experiments.local_agent_v1.run import edit_lines
    from experiments.local_agent_v1.task import source_sha
    source = 'one\ntwo\nthree\n'
    args = dict(start=2, end=2, content='replacement\n', source_sha256=source_sha(source))
    assert edit_lines(source, args) == 'one\nreplacement\nthree\n'
    assert edit_lines(source, dict(args, content='')) == 'one\nthree\n'
    for changes in (dict(source_sha256='stale'), dict(start=0), dict(end=4), dict(start=True)):
        with pytest.raises(ValueError):
            edit_lines(source, dict(args, **changes))


def test_v2_premature_finish_and_repeated_finish_do_not_claim_repair(monkeypatch, tmp_path):
    from experiments.local_agent_v1 import task
    e = object.__new__(RepairEpisode)
    e.done, e.result, e.diagnosis, e.last_execution = False, None, None, None
    e.jd, e.fixture, e.out, e.approved = tmp_path/'job', tmp_path/'fixture', tmp_path, set()
    monkeypatch.setattr(task, 'read_source', lambda _: 'unchanged')
    monkeypatch.setattr(task, 'continuation', lambda *a, **k: pytest.fail('premature finish must not execute'))
    first = e.finish_actual()
    assert first['terminal_reward'] == 0 and first['actual_continuation'] is None
    monkeypatch.setattr(task, 'read_source', lambda _: pytest.fail('repeated finish must be idempotent'))
    assert e.finish_actual() is first


def test_v2_diagnosis_schema_uses_strings_for_evidence_and_boolean_claim():
    schema = next(s['function']['parameters'] for s in tools() if s['function']['name']=='submit_diagnosis')
    assert schema['properties']['evidence_ids']['items']['type'] == 'string'
    assert schema['properties']['claimed_fixed']['type'] == 'boolean'


def test_v2_identical_replacement_reports_no_change_before_match_error():
    with pytest.raises(ValueError, match='no source change'):
        proposed('same same', {'old':'same','new':'same'})


def test_v2_resume_cannot_change_decoding_configuration(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments.local_agent_v1 import run
    model,out=tmp_path/'model',tmp_path/'out';model.mkdir();out.mkdir()
    original={'model_path':str(model),'reduced':False,'status':'pending_security_review',
              'generation_config':{'temperature':0.2,'seed':0,'max_tokens':512}}
    p=out/'trajectory.json';p.write_text(json.dumps(original))
    monkeypatch.setenv('TD_KEY','state-recovery-demo-key')
    with pytest.raises(run.InputError,match='generation configuration'):
        run.execute(SimpleNamespace(model=model,out=out,fixture=tmp_path/'unused',reduced=False,
                                    approve_hash=None,temperature=0,seed=0))
    assert json.loads(p.read_text())==original


def test_v2_group_diagnostics_exclude_infrastructure_and_empty_groups():
    from experiments.local_agent_v1.audit_v2 import group_information
    result=group_information([[1,0,1,0],[0,0,0,0],[1,None,0,1]],4)
    assert result['groups']==3 and result['scored_groups']==2 and result['infrastructure_groups']==1
    assert result['mixed_fraction']==0.5 and result['zero_advantage_fraction']==0.5
    assert group_information([],8)['mixed_fraction'] is None
    with pytest.raises(ValueError):group_information([[0,1]],4)


def test_v2_line_editor_preserves_surviving_line_boundary_without_changing_content():
    from experiments.local_agent_v1.run import edit_lines
    from experiments.local_agent_v1.task import source_sha
    source='a = 1\nb = 2\nc = 3\n'
    def edit(start,end,content):return edit_lines(source,dict(start=start,end=end,content=content,source_sha256=source_sha(source)))
    assert edit(2,2,'b = 9')=='a = 1\nb = 9\nc = 3\n'
    assert edit(2,2,'')=='a = 1\nc = 3\n'
    assert edit(3,3,'c = 9')=='a = 1\nb = 2\nc = 9'


def test_v2_exhausted_response_budget_finalizes_without_model_load(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from experiments.local_agent_v1 import run
    model,out=tmp_path/'model',tmp_path/'out';model.mkdir();out.mkdir()
    state={'model_path':str(model),'reduced':False,'status':'running','approved':[],
           'rows':[{'wall_s':0} for _ in range(12)],'loads':[],'pending':None}
    (out/'trajectory.json').write_text(json.dumps(state))
    e=SimpleNamespace(done=False,result=None,state=lambda:{},tool_calls=12,relaunches=0,
        last_execution=None,phase=0,records=[],phase_records=[],
        finish_actual=lambda:{'terminal_reward':0,'status':'unsuccessful_repair'})
    monkeypatch.setenv('TD_KEY','state-recovery-demo-key')
    monkeypatch.setattr(run,'RepairEpisode',lambda *a,**k:e)
    monkeypatch.setattr(run,'psutil',None)
    run.execute(SimpleNamespace(model=model,out=out,fixture=tmp_path/'unused',reduced=False,approve_hash=None))
    actual=json.loads((out/'trajectory.json').read_text())
    assert actual['status']=='turn_budget' and actual['result']['terminal_reward']==0 and actual['loads']==[]


def test_inference_refuses_to_run_without_optional_memory_guard(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    from experiments.local_agent_v1 import run
    model, out = tmp_path / "model", tmp_path / "episode"
    model.mkdir()
    monkeypatch.setenv("TD_KEY", "state-recovery-demo-key")
    monkeypatch.setattr(run, "psutil", None)
    monkeypatch.setattr(run, "read_source", lambda _: "unchanged")
    monkeypatch.setattr(run, "RepairEpisode", lambda *a, **k: SimpleNamespace(done=False))
    monkeypatch.setattr(sys, "argv", ["runner", "--fixture", "unused", "--model", str(model), "--out", str(out)])
    with pytest.raises(RuntimeError, match="memory guard unavailable"):
        run.main()
    assert json.loads((out / "trajectory.json").read_text())["result"]["terminal_reward"] is None
