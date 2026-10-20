"""Menu v3 development experiment, step 3: score candidate tables and agent rollouts. Scoring calls the library
(tiera.parse_answer, verifiers.verdict / evidence_requirements, reexec.FixOutcome); nothing is reimplemented.

  PYTHONPATH=<library>:. python experiments/menu_v3_dev/analyze.py --run <id> [--check v1.2] [--out DIR] [--compare-historical]

Reads results/<run>/{outcomes.jsonl, records.jsonl, agent_*.jsonl, bundles.jsonl (config section, optional)}.
With --outcomes-runs RUN[,RUN...], only candidate outcomes come from the named runs, matching records' menus exactly.
Writes <out>/{summary.json, TABLES.md} (default results/<run>/analysis/); refuses to overwrite.

Outcomes with exec_mode local_unsandboxed_trusted are not sandboxed, so strict V2 is 0 for them. `v2_exploratory` (V2x)
re-judges the same stored outcome with sandboxed=True. It is exploratory and is never a verified result.
Rollouts of one seed are not independent: intervals are a cluster bootstrap over seeds (2000 resamples, RNG seed 0);
Wilson intervals on raw rollouts are given for reference only."""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from traindoctor_env.reexec import NO_RELAUNCH, FixOutcome
from traindoctor_env.scenario import Truth
from traindoctor_env.tiera import parse_answer
from traindoctor_env.verifiers import evidence_requirements, verdict

HERE = Path(__file__).resolve().parent
CHECKS = ("v1", "v1.1", "v1.2")
N_BOOT = 2000
INVARIANTS = ("sample_accounting", "optimizer_transition", "rank_state", "world_size", "lost_work", "history",
              "sampler_behaviour", "checkpoint_format")


# ---------------------------------------------------------------- loading
def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def load_run(d: Path, outcomes_runs: list[str] | None = None) -> dict:
    runs = outcomes_runs if outcomes_runs is not None else [d.name]
    outcomes = [r for run in runs for r in read_jsonl(d.parent / run / "outcomes.jsonl") if "cand_id" in r]
    recs = {r["seed"]: r for r in read_jsonl(d / "records.jsonl")}
    exec_modes = sorted({r.get("exec_mode", "unknown") for r in outcomes})
    if outcomes_runs is not None:
        counts = Counter((r["seed"], r["cand_id"]) for r in outcomes)
        duplicates = sorted(k for k, n in counts.items() if n != 1)
        if duplicates:
            raise ValueError(f"duplicate outcome (seed, cand_id): {duplicates}")
        expected = {(s, cid) for s, r in recs.items() for cid in r["menu_ids"]}
        actual = set(counts)
        if actual != expected:
            raise ValueError(f"outcome (seed, cand_id) set differs from records/menus: "
                             f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
        if len(exec_modes) != 1:
            raise ValueError(f"merged outcomes must have exactly one exec_mode; found {exec_modes}")
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for r in outcomes:
        by_seed[r["seed"]].append(r)
    seeds = sorted(s for s in by_seed if s in recs)
    agents = {}
    for f in sorted(glob.glob(str(d / "agent_*.jsonl"))):
        name = Path(f).stem[len("agent_"):]
        rows = read_jsonl(Path(f))
        for r in rows:
            r.setdefault("sample", 0)
        agents[name] = [r for r in rows if r["seed"] in seeds]
    configs = {}
    if (d / "bundles.jsonl").exists():
        for b in read_jsonl(d / "bundles.jsonl"):
            m = re.search(r"== job/config\.yaml ==\n(.*?)\n== records ==", b.get("prompt", ""), re.S)
            if m:
                configs[b["seed"]] = m.group(1)
    return dict(seeds=seeds, rows={s: by_seed[s] for s in seeds}, recs={s: recs[s] for s in seeds}, agents=agents,
                configs=configs, outcomes_runs=runs, exec_modes=exec_modes)


# ---------------------------------------------------------------- candidate classification
def outcome_of(row: dict, check: str) -> FixOutcome:
    c = (row.get("checks") or {}).get(check)
    if row.get("status") != "ran" or c is None:
        return dataclasses.replace(NO_RELAUNCH)
    return FixOutcome(**{**c, "tampered": tuple(c.get("tampered", ()))})


def semantic_valid(o: FixOutcome) -> bool:
    return bool(o.ckpt_valid and o.params_match_reference and o.samples_match_reference and o.ranks_consistent
                and "I6" not in o.detail)


def violated(o: FixOutcome) -> list[str]:
    v = []
    if not o.samples_match_reference:
        v.append("sample_accounting")
    if not o.transitions_consistent:
        v.append("optimizer_transition")
    if not o.ranks_consistent:
        v.append("rank_state")
    if "I6" in o.detail:
        v.append("world_size")
    if o.lost_steps > 4:
        v.append("lost_work")
    if not o.history_intact:
        v.append("history")
    if not o.sampler_behaviour:
        v.append("sampler_behaviour")
    if not v and not o.ok and not o.states_match_reference:
        v.append("checkpoint_format")
    return v


def classify(row: dict, check: str) -> dict:
    o = outcome_of(row, check)
    if row.get("status") != "ran" or (row.get("checks") or {}).get(check) is None:
        status, inv = "infrastructure_error", []
    elif not o.ckpt_valid:
        status, inv = "execution_failure", []
    elif o.ok:
        status, inv = "accepted", []
    else:
        status, inv = "invariant_rejection", violated(o)
    sem = semantic_valid(o) and status != "infrastructure_error"
    completed = status in ("accepted", "invariant_rejection")
    return dict(status=status, invariants=inv, semantic_valid=sem, ok=status == "accepted",
                completed=completed, completed_invalid=completed and not sem,
                false_accept=status == "accepted" and not sem,
                false_reject=sem and status != "accepted",
                false_reject_lost_only=sem and status != "accepted" and inv == ["lost_work"],
                false_reject_format_behaviour=sem and status != "accepted" and inv != ["lost_work"])


# ---------------------------------------------------------------- rollout scoring (library calls only)
def evidence_of(records: list[dict], truth: Truth) -> list[str]:
    return sorted({i for g in evidence_requirements(records, truth) if g for i in sorted(g[0], key=str)})


def score(truth: Truth, completion, records: list[dict], menu: dict[str, dict], check: str) -> dict:
    """V0/V1/V2/V2x for one completion. Invalid format -> all zero. `menu` maps cand_id -> outcomes row."""
    a = parse_answer(completion, set(menu)) if isinstance(completion, str) else None
    if a is None:
        return dict(valid=False, v0=False, v1=False, v2=False, v2x=False, role="invalid", row=None)
    return score_answer(truth, a.label, list(a.evidence), a.repair, a.claims_fixed, records, menu, check)


def score_answer(truth, label, evidence, repair_id, claims, records, menu, check) -> dict:
    row = menu.get(repair_id)
    o = outcome_of(row, check) if row else dataclasses.replace(NO_RELAUNCH)
    destructive = bool(row and row["destructive"])
    observed = {r["id"] for r in records}
    kw = dict(observed=observed, phase_records=[records], destructive=destructive, claimed_fixed=claims)
    v = verdict(truth, label, evidence, outcome=o, **kw)
    vx = verdict(truth, label, evidence, outcome=dataclasses.replace(o, sandboxed=True), **kw)
    return dict(valid=True, v0=v.v0, v1=v.v1, v2=v.v2, v2x=vx.v2, role=row["role"] if row else "unknown_repair", row=row)


def metrics(s: dict, check: str) -> dict:
    """One rollout -> {name: (numerator, denominator)}. Every reported rate is a ratio of sums of these."""
    c = classify(s["row"], check) if s["row"] else None
    v1, v2x = float(s["v1"]), float(s["v2x"])
    cinv = float(bool(c and c["completed_invalid"]))
    return {
        "n": (1.0, 1.0), "valid_format": (float(s["valid"]), 1.0),
        "V0": (float(s["v0"]), 1.0), "V1": (v1, 1.0), "V2_strict": (float(s["v2"]), 1.0),
        "V2_exploratory": (v2x, 1.0),
        "P(V1!=V2_strict)": (v1 - float(s["v2"]), 1.0),
        "P(V1!=V2_exploratory)": (v1 - v2x, 1.0),
        "P(V2_exploratory=0|V1=1)": (v1 - v2x, v1),
        "V1_credit_with_completed_invalid_repair": (v1 * cinv, v1),
        "repair_ok_under_check": (float(bool(c and c["ok"])), float(s["valid"])),
        "chosen_repair_completed_invalid": (cinv, float(s["valid"])),
    }


# ---------------------------------------------------------------- uncertainty
def wilson(k: float, n: float, z: float = 1.96):
    if n <= 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def boot_idx(n_seeds: int) -> np.ndarray:
    """Seed-level resample indices: all rollouts of a seed move together. Fixed RNG seed 0."""
    return np.random.default_rng(0).integers(0, n_seeds, size=(N_BOOT, n_seeds))


def cluster_rate(num: np.ndarray, den: np.ndarray, idx: np.ndarray) -> dict:
    """num/den are per-seed sums (aligned to the seed list). Returns rate, 95% percentile CI, denominators."""
    N, D = float(num.sum()), float(den.sum())
    out = dict(num=N, den=D, rate=N / D if D else None, ci95=[None, None])
    if D:
        bn, bd = num[idx].sum(1), den[idx].sum(1)
        ok = bd > 0
        if ok.any():
            r = bn[ok] / bd[ok]
            out["ci95"] = [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]
    return out


def aggregate(items: list[tuple[int, dict]], seeds: list[int], idx: np.ndarray, wilson_ci: bool = False) -> dict:
    """items: [(seed, metrics dict)]. Per-metric cluster-bootstrap rates."""
    pos = {s: i for i, s in enumerate(seeds)}
    names = sorted({k for _, m in items for k in m})
    out = {}
    for k in names:
        num, den = np.zeros(len(seeds)), np.zeros(len(seeds))
        for s, m in items:
            if k in m:
                num[pos[s]] += m[k][0]
                den[pos[s]] += m[k][1]
        out[k] = cluster_rate(num, den, idx)
        if wilson_ci and out[k]["den"]:
            out[k]["wilson95_rollouts"] = list(wilson(out[k]["num"], out[k]["den"]))
    out["n_rollouts"] = len(items)
    out["n_seeds"] = len({s for s, _ in items})
    return out


def group_report(items: list[tuple[int, dict, dict]], seeds, idx, wilson_ci=False) -> dict:
    """items: (seed, metrics, tags{kind, role}). Overall, by kind, by chosen role."""
    rep = dict(overall=aggregate([(s, m) for s, m, _ in items], seeds, idx, wilson_ci))
    for dim in ("kind", "role"):
        groups = defaultdict(list)
        for s, m, t in items:
            groups[t[dim]].append((s, m))
        rep[f"by_{dim}"] = {g: aggregate(v, seeds, idx, wilson_ci) for g, v in sorted(groups.items())}
    return rep


# ---------------------------------------------------------------- tables
def table_report(d: dict, check: str, idx: np.ndarray) -> dict:
    seeds = d["seeds"]
    flags = ["accepted", "execution_failure", "infrastructure_error", "invariant_rejection", "semantic_valid",
             "completed_invalid", "false_accept", "false_reject", "false_reject_lost_only",
             "false_reject_format_behaviour"]
    items = []
    for s in seeds:
        for r in d["rows"][s]:
            c = classify(r, check)
            m = {"n": (1.0, 1.0)}
            for f in flags:
                m[f] = (float(c[f] if f in c else c["status"] == f), 1.0)
            for inv in INVARIANTS:
                m["inv:" + inv] = (float(inv in c["invariants"]), 1.0)
            items.append((s, m, dict(kind=r["kind"], role=r["role"])))
    return group_report(items, seeds, idx)


# ---------------------------------------------------------------- policies
def seed_context(d: dict, s: int):
    t = d["recs"][s]
    truth = Truth(**t["truth"])
    menu = {r["cand_id"]: r for r in d["rows"][s]}
    return truth, t["records"], menu


def avoid_lists(d: dict, check: str) -> dict[int, set[str]]:
    """Menu-only, leave-one-seed-out: roles seen but never accepted in the OTHER seeds' tables."""
    out = {}
    for s in d["seeds"]:
        train = [r for t in d["seeds"] if t != s for r in d["rows"][t]]
        out[s] = {r["role"] for r in train} - {r["role"] for r in train if classify(r, check)["ok"]}
    return out


def agent_report(d: dict, check: str, idx: np.ndarray) -> dict:
    seeds, out = d["seeds"], {}
    avoid = avoid_lists(d, check)
    for name, rows in d["agents"].items():
        items, per_seed = [], defaultdict(list)
        for r in rows:
            truth, recs, menu = seed_context(d, r["seed"])
            sc = score(truth, r.get("completion"), recs, menu, check)
            m = metrics(sc, check)
            gap = float(sc["v1"] and not sc["v2x"])
            m["P(role_in_avoid_list|V1=1,V2_exploratory=0)"] = (gap * (sc["role"] in avoid[r["seed"]]), gap)
            items.append((r["seed"], m, dict(kind=truth.kind, role=sc["role"])))
            if sc["row"]:
                per_seed[r["seed"]].append(classify(sc["row"], check))
        rep = group_report(items, seeds, idx, wilson_ci=True)
        choice = Counter(t["role"] for _, _, t in items)
        multi = {s: v for s, v in per_seed.items() if len(v) >= 2}
        both = sum(1 for v in multi.values() if any(c["ok"] for c in v) and any(c["completed_invalid"] for c in v))
        rep["repair_choice_distribution"] = dict(sorted(choice.items()))
        rep["multi_sample"] = dict(seeds_with_multiple_valid_samples=len(multi), seeds_chose_both_accepted_and_completed_invalid=both)
        out[name] = rep
    return out


def perfect_evidence_item(truth, recs, menu, row, check, label=None) -> dict:
    ev = evidence_of(recs, truth)
    completion = json.dumps(dict(label=truth.label if label is None else label, evidence=ev,
                                 repair=row["cand_id"], claims_fixed=True))
    return score(truth, completion, recs, menu, check)


def location_items(truth: Truth, rows: list[dict]) -> list[dict]:
    if truth.kind == "B_OK":
        return [r for r in rows if not r["destructive"]]
    pre = {"A1": "sync" if truth.variant == 0 else "perrank", "A2": "shard"}.get(truth.kind, "ckpt")
    return [r for r in rows if r["role"].startswith(pre + ":")]


def canonical_item(truth: Truth, rows: list[dict]) -> list[dict]:
    if truth.kind == "B_OK":
        return [r for r in rows if r["role"] == "relaunch"]
    return [r for r in location_items(truth, rows) if r["role"].endswith(":canonical")]


def expected_metrics(sc_list: list[dict], check: str) -> dict:
    ms = [metrics(s, check) for s in sc_list]
    return {k: (sum(m[k][0] for m in ms) / len(ms), sum(m[k][1] for m in ms) / len(ms)) for k in ms[0]}


def reference_report(d: dict, check: str, idx: np.ndarray) -> dict:
    seeds = d["seeds"]
    avoid = avoid_lists(d, check)
    policies = {"uniform_all_items": lambda t, rows: rows, "uniform_location_items": location_items,
                "canonical_location_item": canonical_item,
                "uniform_location_excluding_avoid_list": lambda t, rows: [r for r in location_items(t, rows)
                                                                        if r["role"] not in avoid[t.seed]]}
    out = {}
    for pname, pick in policies.items():
        items = []
        for s in seeds:
            truth, recs, menu = seed_context(d, s)
            chosen = pick(truth, d["rows"][s])
            scs = [perfect_evidence_item(truth, recs, menu, r, check) for r in chosen]
            if not scs:   # empty menu: abstain, retaining the seed in the rate denominator
                scs = [score(truth, None, recs, menu, check)]
            items.append((s, expected_metrics(scs, check), dict(kind=truth.kind, role=pname)))
        out[pname] = group_report(items, seeds, idx)
    return out


def mask(text: str) -> str:
    return re.sub(r"\b[0-9a-fA-F]{6,}\b", "H", re.sub(r"\d+", "N", text))


def shortcut_report(d: dict, check: str, idx: np.ndarray) -> dict:
    """Leave-one-seed-out lookup learners predicting (label, repair role). Evidence is GRANTED perfectly when the
    predicted label is right, so V2_exploratory is an upper bound for a learner that must also cite evidence."""
    seeds = d["seeds"]
    if len(seeds) < 2:
        return dict(learners={}, notes=dict(leave_one_seed_out="skipped: requires at least two seeds"))
    feats = {"label_only": lambda s: "", "menu_only": lambda s: ""}
    notes = {}
    if len(d["configs"]) >= len(seeds):
        feats["config_only"] = lambda s: mask(d["configs"][s])
    else:
        notes["config_only"] = "skipped: bundles.jsonl prompt config sections not available for every seed"
    feats["status_only"] = lambda s: mask("\n".join(r["text"] for r in d["recs"][s]["records"] if r["kind"] == "job"))
    truths = {s: Truth(**d["recs"][s]["truth"]) for s in seeds}
    role_ok = {s: {r["role"]: float(classify(r, check)["ok"]) for r in d["rows"][s]} for s in seeds}
    out = {}
    for name, feat in feats.items():
        items = []
        for s in seeds:
            train = [t for t in seeds if t != s]
            key = feat(s)
            same = [t for t in train if feat(t) == key] or train
            label = Counter(truths[t].label for t in same).most_common()
            label = sorted(label, key=lambda x: (-x[1], x[0]))[0][0]
            truth, recs, menu = seed_context(d, s)
            if name == "label_only":     # label from majority; repair uniform over all items
                scs = [perfect_evidence_item(truth, recs, menu, r, check, label) for r in d["rows"][s]]
            else:                        # label as above; repair = role with best acceptance among `same`
                roles = defaultdict(list)
                for t in same:
                    for ro, v in role_ok[t].items():
                        roles[ro].append(v)
                best = sorted(roles, key=lambda ro: (-sum(roles[ro]) / len(roles[ro]), ro))[0]
                rows = [r for r in d["rows"][s] if r["role"] == best]
                scs = ([perfect_evidence_item(truth, recs, menu, rows[0], check, label)] if rows else
                       [score(truth, None, recs, menu, check)])
            items.append((s, expected_metrics(scs, check), dict(kind=truth.kind, role=name)))
        out[name] = group_report(items, seeds, idx)
    return dict(learners=out, notes=notes)


# ---------------------------------------------------------------- output
def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def fmt(cell: dict) -> str:
    if not cell or cell.get("rate") is None:
        return "n/a"
    lo, hi = cell["ci95"]
    return f"{pct(cell['rate'])} [{pct(lo)}, {pct(hi)}] ({cell['num']:g}/{cell['den']:g})"


def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] +
                     ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def render(summary: dict, compare: bool) -> str:
    L = [f"# Menu v3 analysis: run {summary['run']} (exec_mode: {', '.join(summary['exec_modes'])})", "",
         f"Headline metric: {summary['headline_metric']}.",
         f"Seeds: {summary['n_seeds']}. Headline check: {summary['headline_check']}. Intervals: 95% cluster bootstrap over "
         f"seeds ({N_BOOT} resamples, RNG seed 0); rollouts sharing a seed are not independent.",
         "V2_exploratory re-judges stored outcomes with sandboxed=True; it is exploratory and is NOT verified. "
         "Strict V2 is 0 for local_unsandboxed_trusted rows.", ""]
    for chk, rep in summary["tables"].items():
        L += [f"## Candidate table, check {chk}", ""]
        cols = ["accepted", "execution_failure", "infrastructure_error", "invariant_rejection", "semantic_valid",
                "completed_invalid", "false_accept", "false_reject", "false_reject_lost_only", "false_reject_format_behaviour"]
        for dim, title in (("overall", "all"), ("by_kind", "kind"), ("by_role", "role")):
            groups = {"all": rep["overall"]} if dim == "overall" else rep[dim]
            L += [md_table([title, "n"] + cols, [[g, int(v["n"]["num"])] + [fmt(v[c]) for c in cols] for g, v in groups.items()]), ""]
        inv = ["inv:" + i for i in INVARIANTS]
        L += ["Violated invariants (a rejection can violate several):", "",
              md_table(["kind"] + list(INVARIANTS), [[g] + [fmt(v[i]) for i in inv] for g, v in rep["by_kind"].items()]), ""]
    cols = ["valid_format", "V0", "V1", "V2_strict", "V2_exploratory", "P(V1!=V2_strict)", "P(V1!=V2_exploratory)",
            "P(V2_exploratory=0|V1=1)", "V1_credit_with_completed_invalid_repair", "repair_ok_under_check"]
    for title, key in (("Agents", "agents"), ("Reference policies (perfect label and certificate evidence)", "references")):
        L += [f"## {title} (check {summary['headline_check']})", ""]
        pols = summary[key]
        if not pols:
            L += ["none", ""]
        L += [md_table(["policy", "rollouts"] + cols, [[p, r["overall"]["n_rollouts"]] + [fmt(r["overall"].get(c)) for c in cols]
                                                       for p, r in pols.items()]), ""]
        for p, r in pols.items():
            L += [f"### {p} by kind", "", md_table(["kind", "rollouts"] + cols[:6], [[g, v["n_rollouts"]] + [fmt(v.get(c)) for c in cols[:6]]
                                                                                for g, v in r["by_kind"].items()]), ""]
            L += [f"### {p} by chosen role", "", md_table(["role", "rollouts"] + cols[:6], [[g, v["n_rollouts"]] + [fmt(v.get(c)) for c in cols[:6]]
                                                                                       for g, v in r["by_role"].items()]), ""]
            if "multi_sample" in r:
                L += [f"Multi-sample: {json.dumps(r['multi_sample'])}. Repair choice: {json.dumps(r['repair_choice_distribution'])}", ""]
    sc = summary["shortcuts"]
    L += ["## Avoid-list (leave-one-seed-out, menu-only)", "",
          "For each held-out seed, learn roles observed but never accepted under the chosen check in OTHER seeds' "
          "outcome tables only. The fraction below conditions on V1=1 and V2_exploratory=0.", "",
          md_table(["agent", "fraction of V1-pass/V2-fail rollouts in avoid-list"],
                   [[n, fmt(r["overall"].get("P(role_in_avoid_list|V1=1,V2_exploratory=0)"))]
                    for n, r in summary["agents"].items()]), "",
          "Reference: perfect V1 + uniform over items at the fault location excluding that seed's avoid-list. "
          "An empty eligible menu abstains (zero V2, seed retained in denominator).", "",
          md_table(["policy", "V2_exploratory"],
                   [["uniform_location_excluding_avoid_list",
                     fmt(summary["references"]["uniform_location_excluding_avoid_list"]["overall"].get("V2_exploratory"))]]), ""]
    L += ["## Shortcut baselines (leave-one-seed-out lookup; evidence granted, so V2_exploratory is an upper bound)", ""]
    L += [md_table(["learner", "V0", "V2_exploratory"], [[n, fmt(r["overall"]["V0"]), fmt(r["overall"]["V2_exploratory"])]
                                                        for n, r in sc["learners"].items()]), ""]
    for k, v in sc["notes"].items():
        L += [f"{k}: {v}", ""]
    if compare:
        L += ["## Historical V1!=V2 (not comparable distributions)", "",
              "- menu v1: 2.3% (3072 rollouts)", "- menu v2: 2.6%", "Different task distribution; shown for context only.", ""]
    return "\n".join(L)


def analyze(d: dict, run: str, check: str) -> dict:
    idx = boot_idx(len(d["seeds"]))
    return dict(run=run, outcomes_runs=d["outcomes_runs"], exec_modes=d["exec_modes"],
                headline_metric="V2_strict" if d["exec_modes"] == ["modal_sandbox"] else "V2_exploratory",
                n_seeds=len(d["seeds"]), headline_check=check, bootstrap=dict(resamples=N_BOOT, rng_seed=0, unit="seed"),
                tables={c: table_report(d, c, idx) for c in CHECKS},
                agents=agent_report(d, check, idx), references=reference_report(d, check, idx),
                avoid_list=dict(roles_by_held_out_seed={s: sorted(roles) for s, roles in avoid_lists(d, check).items()}),
                shortcuts=shortcut_report(d, check, idx))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--outcomes-runs", help="comma-separated runs supplying candidate outcomes instead of --run")
    ap.add_argument("--check", default="v1.2", choices=CHECKS)
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--out")
    ap.add_argument("--compare-historical", action="store_true")
    a = ap.parse_args(argv)
    rd = Path(a.results) / a.run
    out = Path(a.out) if a.out else rd / "analysis"
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    runs = [r.strip() for r in a.outcomes_runs.split(",")] if a.outcomes_runs is not None else None
    if runs is not None and not all(runs):
        ap.error("--outcomes-runs requires nonempty run names")
    try:
        summary = analyze(load_run(rd, runs), a.run, a.check)
    except ValueError as e:
        ap.error(str(e))
    out.mkdir(parents=True, exist_ok=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out / "TABLES.md").write_text(render(summary, a.compare_historical) + "\n")
    print(f"wrote {out}/summary.json and TABLES.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
