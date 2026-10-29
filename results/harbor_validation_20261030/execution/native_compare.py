"""Three authored native-macOS controls matched to the single Harbor task; no model."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
import psutil
from jobs import sandbox
from traindoctor_env.reexec import Replay
from traindoctor_env.scenario import Scenario
from traindoctor_env.snapshot import execute


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def memory_guard():
    available, swap = psutil.virtual_memory().available, psutil.swap_memory().used
    if available < 8 * 1024**3:
        raise RuntimeError("native comparison requires at least 8 GiB available RAM")
    return {"available_bytes": available, "swap_used_bytes": swap}


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError("authored control target must match exactly once")
    return source.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.environ["TD_KEY"] = "harbor-local-smoke-key"
    if platform.system() != "Darwin" or not sandbox.available():
        raise RuntimeError("native macOS sandbox unavailable; refusing unsandboxed comparison")
    initial = memory_guard()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__import__("traindoctor_env").__file__).resolve().parents[1]
    scenario = Scenario(seed=777, kind="B_DUP", variant=0, world_size=2, job_seed=777)
    config = {"source": "executed", "scenario": asdict(scenario), "platform": platform.platform(),
              "backend": "local_os_sandbox", "format_check": "v1.2", "allow_unsandboxed": False,
              "environment_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
              "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "psutil")},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "memory_before": initial, "timings_exclude_imports": True,
              "provenance": "authored controls; shared fixture, dependent outcomes; not model trajectories"}
    write(args.out / "config.json", config)
    began = time.perf_counter()
    snapshot = execute(scenario, args.out / "snapshot")
    baseline_wall = time.perf_counter() - began
    original = snapshot.files["job/ckpt.py"]
    oracle = replace_once(original, 'state.update(step=torch.tensor([step]))',
                          'state.update(sampler=sampler.state(), step=torch.tensor([step]))')
    invalid = replace_once(oracle, '    opt.load_state_dict(sd)\n',
                           '    sd["state"] = {}\n    opt.load_state_dict(sd)\n')
    sources = {"oracle": oracle, "nop": original, "invalid": invalid}
    assert len({hashlib.sha256(s.encode()).hexdigest() for s in sources.values()}) == 3
    rows = []
    for name, source in sources.items():
        before, started = memory_guard(), time.perf_counter()
        case = args.out / name
        case.mkdir()
        (case / "submitted_ckpt.py").write_text(source)
        replay = Replay(snapshot, backend=None, allow_unsandboxed=False, format_check="v1.2")
        try:
            if not replay.write_file("job/ckpt.py", source):
                raise RuntimeError("declared source edit was rejected")
            outcome = replay.outcome()
        finally:
            replay.close()
        row = {"source": "executed", "case": name, "wall_s": time.perf_counter() - started,
               "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
               "memory_before": before, "memory_after": {"available_bytes": psutil.virtual_memory().available,
               "swap_used_bytes": psutil.swap_memory().used}, "outcome": asdict(outcome)}
        write(case / "result.json", row)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if not outcome.sandboxed or outcome.ok != (name == "oracle") or not outcome.ckpt_valid:
            raise RuntimeError("native control did not satisfy its declared expected result")
    write(args.out / "summary.json", {"source": "executed", "trusted_fixture_wall_s": baseline_wall,
          "cases": rows, "expected_rewards": {"oracle": 1, "nop": 0, "invalid": 0}, "status": "pass"})
    print("PASS: native sandbox controls oracle=1, nop=0, completing-invalid=0", flush=True)


if __name__ == "__main__":
    main()
