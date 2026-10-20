"""Rules for experiments/menu_v3_dev/analyze.py, on a synthetic 2-seed run built from real library objects."""
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("TD_KEY", "test-key")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "menu_v3_dev"))

import analyze  # noqa: E402
from traindoctor_env.reexec import FixOutcome  # noqa: E402
from traindoctor_env.scenario import Truth  # noqa: E402
from traindoctor_env.verifiers import evidence_requirements  # noqa: E402

LABEL = "COLLECTIVE_DIVERGENCE"


def outcome(**kw) -> dict:
    base = dict(ok=True, bit_identical=False, params_match_reference=True, samples_match_reference=True,
                restores_consistent=True, no_duplicate_updates=True, lost_steps=0, ranks_consistent=True, ckpt_valid=True,
                transitions_consistent=True, sandboxed=False, exec_mode="dev_container")
    base.update(kw)
    return dataclasses.asdict(FixOutcome(**{**dict(ok=True, bit_identical=False, params_match_reference=True,
                                                    samples_match_reference=True, restores_consistent=True,
                                                    no_duplicate_updates=True, lost_steps=0, ranks_consistent=True), **base}))


def records(seed):
    return [dict(id=f"E{seed}a", kind="trace", rank=1, step=3, text="[run0] rank 1 step=3 broadcast seq=7 enter"),
            dict(id=f"E{seed}b", kind="trace", rank=0, step=3, text="[run0] rank 0 step=3 all_reduce seq=7 enter"),
            dict(id=f"E{seed}c", kind="log", rank=0, step=1, text="noise"),
            dict(id=f"E{seed}j", kind="job", rank=0, step=0, text="status 12 failed")]


def truth(seed):
    return Truth(seed, "A1", 0, LABEL, "t", [], dict(rank=1, bad_op="broadcast", step=3))


def row(seed, cid, role, **o):
    chk = outcome(**o)
    return dict(seed=seed, kind="A1", variant=0, world=2, label=LABEL, cand_id=cid, role=role, destructive=False,
                exec_mode="local_unsandboxed_trusted", status="ran", checks={"v1": chk, "v1.1": chk, "v1.2": chk})


def make_run(tmp_path, completions=None):
    d = tmp_path / "run"
    d.mkdir()
    rows, recs, ag = [], [], []
    for s in (1, 2):
        rows += [row(s, f"R{s}good", "sync:canonical"),
                 row(s, f"R{s}fake", "ckpt:from_step", samples_match_reference=True, params_match_reference=False),  # ok, not semantic
                 row(s, f"R{s}fail", "relaunch", ok=False, ckpt_valid=False, samples_match_reference=False)]
        recs.append(dict(seed=s, truth=dataclasses.asdict(truth(s)), records=records(s), menu_ids=[r["cand_id"] for r in rows if r["seed"] == s]))
    (d / "outcomes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    for k, (s, c) in enumerate(completions or []):
        ag.append(dict(seed=s, sample=k, agent="a", completion=c))
    (d / "agent_a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ag))
    return d


def answer(s, repair, label=LABEL, ev=True):
    e = [f"E{s}a", f"E{s}b"] if ev else []
    return json.dumps(dict(label=label, evidence=e, repair=f"R{s}{repair}", claims_fixed=True))


def test_certificate_fixture_is_really_evidence():
    assert all(evidence_requirements(records(1), truth(1)))


def test_invalid_completion_scores_zero_on_every_rung(tmp_path):
    d = analyze.load_run(make_run(tmp_path))
    t, recs, menu = analyze.seed_context(d, 1)
    for bad in ("not json", None, '{"label": "x"}', answer(1, "good").replace("claims_fixed", "claims")):
        sc = analyze.score(t, bad, recs, menu, "v1.2")
        assert not (sc["valid"] or sc["v0"] or sc["v1"] or sc["v2"] or sc["v2x"])


def test_ladder_is_monotone_and_conditional_uses_v1_denominator(tmp_path):
    comps = [(1, answer(1, "good")), (1, answer(1, "good", ev=False)), (1, answer(1, "good", label="NO_FAULT")),
             (2, answer(2, "fake")), (2, "garbage"), (2, answer(2, "fail"))]
    d = analyze.load_run(make_run(tmp_path, comps))
    n_v1 = 0
    for s, c in comps:
        t, recs, menu = analyze.seed_context(d, s)
        sc = analyze.score(t, c, recs, menu, "v1.2")
        assert sc["v2x"] <= sc["v1"] <= sc["v0"]
        assert sc["v2"] <= sc["v2x"]
        n_v1 += sc["v1"]
    rep = analyze.agent_report(d, "v1.2", analyze.boot_idx(2))["a"]["overall"]
    assert rep["P(V2_exploratory=0|V1=1)"]["den"] == n_v1 == 3
    assert rep["P(V2_exploratory=0|V1=1)"]["num"] == 1          # of the 3 V1 rollouts only the "fail" repair is rejected ...
    assert rep["V2_strict"]["num"] == 0                           # ... and strict V2 is 0 on unsandboxed rows


def test_ok_but_not_semantically_valid_is_false_accept():
    r = row(1, "x", "ckpt:from_step", params_match_reference=False)
    c = analyze.classify(r, "v1.2")
    assert c["status"] == "accepted" and c["false_accept"] and c["completed_invalid"] and not c["false_reject"]
    g = analyze.classify(row(1, "y", "sync:canonical"), "v1.2")
    assert not g["false_accept"] and g["semantic_valid"]
    lw = analyze.classify(row(1, "z", "r", ok=False, lost_steps=9), "v1.2")
    assert lw["false_reject_lost_only"] and lw["invariants"] == ["lost_work"]


def test_bootstrap_moves_whole_seeds_together():
    idx = analyze.boot_idx(2)
    assert idx.shape == (analyze.N_BOOT, 2) and idx.max() == 1
    num, den = np.array([2.0, 0.0]), np.array([2.0, 2.0])      # seed 0: both rollouts hit, seed 1: both miss
    rates = set(np.round(num[idx].sum(1) / den[idx].sum(1), 6))
    assert rates <= {0.0, 0.5, 1.0}
    cell = analyze.cluster_rate(num, den, idx)
    assert cell["rate"] == 0.5 and cell["ci95"] == [0.0, 1.0]
    assert np.array_equal(analyze.boot_idx(2), idx)             # deterministic


def test_avoid_list_learns_only_other_seeds_and_conditions_on_v1_pass_v2_fail(tmp_path):
    comps = [(1, answer(1, "rare")), (1, answer(1, "bad")), (1, answer(1, "bad", ev=False)),
             (2, answer(2, "rare")), (2, answer(2, "bad")), (2, "garbage")]
    d = analyze.load_run(make_run(tmp_path, comps))
    for s in d["seeds"]:
        d["rows"][s] += [row(s, f"R{s}rare", "sync:rare", ok=s == 1),
                         row(s, f"R{s}bad", "sync:bad", ok=False, params_match_reference=False)]
    d["rows"][2][-1]["checks"]["v1"] = outcome()
    avoid = analyze.avoid_lists(d, "v1.2")
    assert avoid[1] == {"relaunch", "sync:rare", "sync:bad"}  # accepted ONLY on held-out seed: still avoided
    assert avoid[2] == {"relaunch", "sync:bad"}
    assert "sync:bad" not in analyze.avoid_lists(d, "v1")[1]  # chosen check controls acceptance
    idx = analyze.boot_idx(2)
    rep = analyze.agent_report(d, "v1.2", idx)["a"]["overall"]
    cell = rep["P(role_in_avoid_list|V1=1,V2_exploratory=0)"]
    assert cell["num"] == 2 and cell["den"] == 3 and cell["rate"] == pytest.approx(2 / 3)
    assert cell["ci95"] == [0.5, 1.0]  # one gap on seed 1, two on seed 2; resample whole seeds
    refs = analyze.reference_report(d, "v1.2", idx)
    cell = refs["uniform_location_excluding_avoid_list"]["overall"]["V2_exploratory"]
    assert cell["den"] == 2 and cell["rate"] == 0.75  # seed 1: canonical; seed 2: canonical or rare
    assert cell["ci95"] == [0.5, 1.0]


def test_avoid_list_empty_eligible_menu_retains_seed_denominator(tmp_path):
    d = analyze.load_run(make_run(tmp_path, [(1, answer(1, "good"))]))
    d["rows"][2][0]["checks"]["v1.2"]["ok"] = False
    rep = analyze.reference_report(d, "v1.2", analyze.boot_idx(2))
    cell = rep["uniform_location_excluding_avoid_list"]["overall"]["V2_exploratory"]
    assert cell["den"] == 2 and cell["rate"] == 0  # seed 1 abstains; seed 2's sole eligible repair fails
    rep = analyze.agent_report(d, "v1.2", analyze.boot_idx(2))["a"]["overall"]
    cell = rep["P(role_in_avoid_list|V1=1,V2_exploratory=0)"]
    assert cell["den"] == 0 and cell["rate"] is None and cell["ci95"] == [None, None]


def test_load_run_ignores_agents_without_outcome_tables(tmp_path):
    path = make_run(tmp_path, [(1, answer(1, "good")), (2, answer(2, "good"))])
    rows = analyze.read_jsonl(path / "outcomes.jsonl")
    (path / "outcomes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows if r["seed"] == 1))
    d = analyze.load_run(path)
    assert d["seeds"] == [1] and [r["seed"] for r in d["agents"]["a"]] == [1]
    assert analyze.analyze(d, "run", "v1.2")["shortcuts"]["learners"] == {}


def test_shortcut_with_unavailable_repair_gets_no_format_or_label_credit(tmp_path):
    d = analyze.load_run(make_run(tmp_path))
    d["rows"] = {1: [row(1, "R1good", "sync:canonical")], 2: [row(2, "R2good", "shard:canonical")]}
    rep = analyze.shortcut_report(d, "v1.2", analyze.boot_idx(2))["learners"]["menu_only"]["overall"]
    assert rep["valid_format"]["rate"] == rep["V0"]["rate"] == 0


def test_output_refuses_to_overwrite(tmp_path):
    d = make_run(tmp_path, [(1, answer(1, "good"))])
    out = tmp_path / "out"
    args = ["--run", "run", "--results", str(tmp_path), "--out", str(out), "--compare-historical"]
    assert analyze.main(args) == 0
    s = json.loads((out / "summary.json").read_text())
    tables = (out / "TABLES.md").read_text()
    assert s["agents"]["a"]["overall"]["V0"]["den"] == 1 and "not comparable" in tables
    assert s["avoid_list"]["roles_by_held_out_seed"] == {"1": ["relaunch"], "2": ["relaunch"]}
    assert "Avoid-list (leave-one-seed-out, menu-only)" in tables and "uniform_location_excluding_avoid_list" in tables
    before = (out / "summary.json").read_text()
    with pytest.raises(SystemExit):
        analyze.main(args)
    assert (out / "summary.json").read_text() == before


@pytest.mark.parametrize("mutation, message", [(None, None), ("duplicate", "duplicate outcome"),
                                               ("missing", "missing="), ("extra", "extra="),
                                               ("mixed", "exactly one exec_mode")])
def test_merged_outcomes_match_single_run_and_require_exact_menu(tmp_path, capsys, mutation, message):
    path = make_run(tmp_path, [(1, answer(1, "good")), (2, answer(2, "fail"))])
    baseline = analyze.analyze(analyze.load_run(path), "run", "v1.2")
    rows = analyze.read_jsonl(path / "outcomes.jsonl")
    parts = [rows[::2], rows[1::2]]
    if mutation == "duplicate":
        parts[1].append(parts[0][0])
    elif mutation == "missing":
        parts[0].pop()
    elif mutation == "extra":
        parts[0].append({**rows[0], "seed": 99})
    elif mutation == "mixed":
        parts[0][0]["exec_mode"] = "modal_sandbox"
    for name, part in zip(("split1", "split2"), parts):
        dest = tmp_path / name
        dest.mkdir()
        (dest / "outcomes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in part))
    (path / "outcomes.jsonl").unlink()  # only records/agents come from --run
    args = ["--run", "run", "--results", str(tmp_path), "--outcomes-runs", "split1,split2"]
    if message:
        with pytest.raises(SystemExit, match="2"):
            analyze.main(args)
        assert message in capsys.readouterr().err
        assert not (path / "analysis").exists()
        return
    assert analyze.main(args) == 0
    summary = json.loads((path / "analysis" / "summary.json").read_text())
    assert summary.pop("outcomes_runs") == ["split1", "split2"]
    baseline.pop("outcomes_runs")
    assert summary == json.loads(json.dumps(baseline))
    assert summary["exec_modes"] == ["local_unsandboxed_trusted"]
    assert "exec_mode: local_unsandboxed_trusted" in (path / "analysis" / "TABLES.md").read_text()


def test_modal_outcomes_headline_strict_v2_and_report_exploratory(tmp_path):
    path = make_run(tmp_path, [(1, answer(1, "good"))])
    rows = analyze.read_jsonl(path / "outcomes.jsonl")
    for r in rows:
        r["exec_mode"] = "modal_sandbox"
        for check in r["checks"].values():
            check.update(sandboxed=True, exec_mode="modal_sandbox")
    (path / "outcomes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert analyze.main(["--run", "run", "--results", str(tmp_path), "--outcomes-runs", "run"]) == 0
    summary = json.loads((path / "analysis" / "summary.json").read_text())
    assert summary["outcomes_runs"] == ["run"] and summary["exec_modes"] == ["modal_sandbox"]
    assert summary["headline_metric"] == "V2_strict"
    metrics = summary["agents"]["a"]["overall"]
    assert metrics["V2_strict"]["rate"] == metrics["V2_exploratory"]["rate"] == 1
    tables = (path / "analysis" / "TABLES.md").read_text()
    assert "exec_mode: modal_sandbox" in tables.splitlines()[0]
    assert "Headline metric: V2_strict" in tables and "V2_exploratory" in tables
