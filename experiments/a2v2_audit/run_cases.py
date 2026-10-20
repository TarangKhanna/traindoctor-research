"""Exploratory (trusted, unsandboxed, local): A2v2 red-team cases gold/S01/R1/S02 on scenario (101, A2v2, world 3).

Run from the LIBRARY root (it imports the library's scripts/redteam_a2v2.py for the case definitions):
  cd <traindoctor-env>; PYTHONPATH=. python <this file> <scratch dir> > outcomes.jsonl
Outcomes are exploratory (no OS isolation); repo-authored cases only."""
import os, sys, json, time
os.environ.setdefault("TD_KEY", "a2v2-audit-key")
from pathlib import Path
sys.path.insert(0, "scripts")
from traindoctor_env.scenario import Scenario
from traindoctor_env.snapshot import execute, load_truth
from traindoctor_env import reexec
from traindoctor_env.verifiers import edits_of
import redteam_a2v2 as ra
reexec.DEV_CONTAINER_MODE = True
snap = execute(Scenario(101, "A2v2", 0, 3, 31337), Path(sys.argv[1]))
truth = load_truth(snap)
cases = {c.id: c for c in ra.build_cases(snap.files, truth.details, edits_of(truth))}
rows = []
for name in ["null", "gold", "alt1", "alt2", "S01", "R1", "S02"]:
    for fc in ("v1", "v1.1"):
        t = time.time()
        o = reexec.verify_fix(snap, list(cases[name].edits), allow_unsandboxed=True, format_check=fc)
        rows.append(dict(case=name, format_check=fc, ok=o.ok, path=o.path, params=o.params_match_reference,
                         samples=o.samples_match_reference, trans=o.transitions_consistent, ranks=o.ranks_consistent,
                         lost=o.lost_steps, gaps=o.sample_gaps, dups=o.sample_dups, completed=o.ckpt_valid,
                         exec_mode="local_unsandboxed_trusted", secs=round(time.time() - t, 1)))
        print(json.dumps(rows[-1]), flush=True)
