"""Menu v3 development experiment, step 1: execute every menu-v3 candidate on fresh development incidents.

For each seed: execute the incident (real Gloo job), build the v3 menu (traindoctor_env.menu_v3) and the prompt bundle,
then re-execute every candidate ONCE and judge that run under verifier format checks v1, v1.1 and v1.2
(reexec.verify_fix_checks). Nothing about a candidate's outcome is assigned: the role is recorded only to group results.

Execution mode is recorded per row. `local_unsandboxed_trusted` means the job ran on this machine without OS isolation:
allowed only because every candidate is repo-authored code (traindoctor_env/menu_v3.py), never agent output. Such
outcomes are EXPLORATORY and never count as verified (the library's strict V2 requires a sandboxed run); the verified
rerun is the Modal plan in RUN_PLAN_MODAL.md.

  PYTHONPATH=<library>:. python experiments/menu_v3_dev/build_tables.py --run <id> --per-kind 12 [--unsandboxed-trusted]
Writes experiments/menu_v3_dev/results/<run>/{config.json, outcomes.jsonl, bundles.jsonl}; refuses to overwrite."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from traindoctor_env import reexec
from traindoctor_env.bundle import make_bundle, make_menu
from traindoctor_env.pilot import DEV_SEEDS
from traindoctor_env.scenario import make_scenario
from traindoctor_env.snapshot import execute, load_truth
from traindoctor_env.tiera import prompt_text

HERE = Path(__file__).resolve().parent
SEED_START = 50_000          # disjoint from DEV (10000-10399), TRAIN (20000-20399), the sealed TEST pool and BURNED (7001-7008)
RESERVED = (DEV_SEEDS, range(20_000, 20_400), range(30_000, 30_400), range(7001, 7009))
CHECKS = ("v1", "v1.1", "v1.2")
KINDS = ("A1", "A2", "B_SKIP", "B_DUP", "B_OK")


def seeds(per_kind: int) -> list[int]:
    """The first `per_kind` seeds of each kind from SEED_START upward, in seed order (fixed before any outcome is seen)."""
    got, s = {k: [] for k in KINDS}, SEED_START
    while any(len(v) < per_kind for v in got.values()):
        assert not any(s in r for r in RESERVED)
        k = make_scenario(s).kind
        if len(got[k]) < per_kind:
            got[k].append(s)
        s += 1
    return sorted(x for v in got.values() for x in v)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def judge(snap, cand, mode: str) -> dict:
    t = time.time()
    try:
        outs = reexec.verify_fix_checks(snap, list(cand.edits), CHECKS, allow_unsandboxed=(mode == "local_unsandboxed_trusted"),
                                        backend="modal" if mode == "modal_sandbox" else None)
    except Exception as e:  # noqa: BLE001  an infrastructure failure is recorded, never scored as a verdict
        return dict(status="infrastructure_error", error=f"{type(e).__name__}: {e}"[:500],
                    trace=traceback.format_exc()[-1500:], secs=round(time.time() - t, 1))
    return dict(status="ran", secs=round(time.time() - t, 1),
                checks={fc: dataclasses.asdict(o) for fc, o in outs.items()})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--per-kind", type=int, default=12)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seeds", help="comma-separated explicit seeds (a recheck of an earlier run), instead of --per-kind")
    mx = ap.add_mutually_exclusive_group()
    mx.add_argument("--unsandboxed-trusted", action="store_true",
                    help="run repo-authored candidates without OS isolation (exploratory; never verified)")
    mx.add_argument("--modal", action="store_true",
                    help="re-execute every candidate in a Modal gVisor sandbox (verified path; costs money: RUN_PLAN_MODAL.md)")
    a = ap.parse_args(argv)
    out = HERE / "results" / a.run
    if out.exists():
        sys.exit(f"{out} exists; results are never overwritten")
    mode = "local_unsandboxed_trusted" if a.unsandboxed_trusted else "modal_sandbox" if a.modal else "sandboxed"
    reexec.DEV_CONTAINER_MODE = a.unsandboxed_trusted      # keeps the raw outcome of an unsandboxed run, labelled unverified
    lib = Path(reexec.__file__).resolve().parent.parent
    lib_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=lib, capture_output=True, text=True).stdout.strip()
    lib_dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "traindoctor_env", "jobs"], cwd=lib,
                                    capture_output=True, text=True).stdout.strip())
    out.mkdir(parents=True)
    sd = [int(x) for x in a.seeds.split(",")] if a.seeds else seeds(a.per_kind)
    assert not any(s in r for s in sd for r in RESERVED)
    (out / "config.json").write_text(json.dumps(dict(
        run=a.run, seeds=sd, per_kind=a.per_kind, menu_version="v3", prompt_version="v2", checks=CHECKS, exec_mode=mode,
        library_commit=lib_sha, library_dirty=lib_dirty, td_key_sha256=sha(os.environ.get("TD_KEY", "")),
        python=sys.version.split()[0], platform=platform.platform(), started=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        workers=a.workers, source="executed"), indent=2) + "\n")
    with (out / "outcomes.jsonl").open("a") as fo, (out / "bundles.jsonl").open("a") as fb:
        for seed in sd:
            sc = make_scenario(seed)
            with tempfile.TemporaryDirectory() as tmp:
                t0 = time.time()
                snap = execute(sc, Path(tmp) / "snap")
                truth = load_truth(snap)
                menu = make_menu(snap, None, "v3")
                try:
                    bundle = make_bundle(snap, menu, truth)
                    fb.write(json.dumps(dict(seed=seed, kind=sc.kind, variant=sc.variant, world=sc.world_size, label=truth.label,
                                             prompt=prompt_text(bundle, "v2"), tokens_est=bundle.tokens,
                                             record_ids=[r["id"] for r in bundle.records],
                                             menu=[dict(id=c.id, role=c.role, destructive=c.destructive) for c in menu])) + "\n")
                except Exception as e:  # noqa: BLE001
                    fb.write(json.dumps(dict(seed=seed, kind=sc.kind, bundle_error=f"{type(e).__name__}: {e}")) + "\n")
                fb.flush()
                with ThreadPoolExecutor(a.workers) as ex:
                    res = list(ex.map(lambda c: judge(snap, c, mode), menu))
                for c, r in zip(menu, res):
                    fo.write(json.dumps(dict(seed=seed, kind=sc.kind, variant=sc.variant, world=sc.world_size, label=truth.label,
                                             cand_id=c.id, role=c.role, destructive=c.destructive,
                                             edits_sha256=sha(json.dumps([dataclasses.asdict(e) for e in c.edits])),
                                             n_edits=len(c.edits), exec_mode=mode, source="executed", **r)) + "\n")
                fo.flush()
                print(f"seed {seed} {sc.kind} done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
