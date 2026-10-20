"""Menu v3 development experiment, step 1b: the bundle records and the Truth needed to score answers.

build_tables.py stores each prompt but not the structured records or the injected Truth that the library's verdict()
needs. This step re-executes each incident (no candidates) with the library at its current commit, rebuilds the v3 menu
and bundle, and REQUIRES the menu ids and every candidate's executed edits (sha256) to equal what build_tables.py stored (scenarios and snapshots are deterministic for a given seed and TD_KEY; a mismatch aborts). So the
executed outcome table applies unchanged to the prompts written here. The prompt TEXT may differ from the stored one: the
v3 display of config items was corrected after dev1 started (library 80d1de0: config items show their key path only),
and a shorter menu can let the token-cap compaction keep more records (recorded per seed). records.jsonl holds the
corrected prompt and its records, which every agent answer is collected and scored on. Refuses to overwrite.

  TD_KEY=<same key> PYTHONPATH=<library>:. python experiments/menu_v3_dev/bundle_records.py --run dev1"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import subprocess
import json
import sys
import tempfile
from pathlib import Path

from traindoctor_env import reexec
from traindoctor_env.bundle import make_bundle, make_menu
from traindoctor_env.scenario import make_scenario
from traindoctor_env.snapshot import execute, load_truth
from traindoctor_env.tiera import prompt_text

HERE = Path(__file__).resolve().parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    d = HERE / "results" / a.run
    out = d / "records.jsonl"
    if out.exists():
        sys.exit(f"{out} exists; results are never overwritten")
    stored = [json.loads(x) for x in (d / "bundles.jsonl").read_text().splitlines()]
    edits_sha = {(r["seed"], r["cand_id"]): r["edits_sha256"] for r in map(json.loads, (d / "outcomes.jsonl").read_text().splitlines())}
    lib = Path(reexec.__file__).resolve().parent.parent
    lib_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=lib, capture_output=True, text=True).stdout.strip()
    rows = []
    for b in stored:
        if "prompt" not in b:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            snap = execute(make_scenario(b["seed"]), Path(tmp) / "snap")
            truth = load_truth(snap)
            menu = make_menu(snap, None, "v3")
            bundle = make_bundle(snap, menu, truth)
            sha = {c.id: hashlib.sha256(json.dumps([dataclasses.asdict(e) for e in c.edits]).encode()).hexdigest() for c in menu}
            same = [c.id for c in menu] == [m["id"] for m in b["menu"]] and all(edits_sha[(b["seed"], i)] == h for i, h in sha.items())
            if not same:
                sys.exit(f"seed {b['seed']}: rebuilt menu or edits differ from the stored ones (not deterministic)")
            prompt = prompt_text(bundle, "v2")
            rows.append(dict(seed=b["seed"], truth=dataclasses.asdict(truth), records=[dict(r) for r in bundle.records],
                             menu_ids=[c.id for c in menu], prompt=prompt, tokens_est=bundle.tokens,
                             prompt_identical_to_stored=prompt == b["prompt"],
                             record_ids_identical_to_stored=[r["id"] for r in bundle.records] == b["record_ids"],
                             library_commit=lib_sha))
        print(b["seed"], "ok", flush=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
