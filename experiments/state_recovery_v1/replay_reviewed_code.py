"""Re-execute one exact, reviewed model proposal; never accept arbitrary edits.

The Qwen proposal below merely repeats the existing enclosing condition. Its
calls and effects are those of the repository's original restore code. This is
a fixed trusted control, not a generated-code executor or security validator.
"""
import argparse
import dataclasses
import json
from pathlib import Path

from jobs.build import Edit
from traindoctor_env.reexec import verify_fix
from traindoctor_env.scenario import Scenario
from traindoctor_env.snapshot import execute

OLD = ('    if "sampler" in state:  # checkpoints from before the sampler was tracked have no position\n'
       '        sampler.load_state(state["sampler"])\n    return int(state["step"])')
NEW = ('    if "sampler" in state:  # checkpoints from before the sampler was tracked have no position\n'
       '        sampler.load_state(state["sampler"]) if "sampler" in state else None\n'
       '    return int(state["step"])')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("refusing to overwrite evidence")
    edit = json.loads((args.profile / "code_edit.json").read_text())
    if edit != {"path": "job/ckpt.py", "old": OLD, "new": NEW}:
        ap.error("this command accepts ONLY the exact reviewed redundant-condition proposal")
    args.out.mkdir(parents=True)
    snap = execute(Scenario(202, "B_DUP", 0, 2, 777), args.out / "incident")
    outcome = verify_fix(snap, [Edit(**edit)], format_check="v1.2")
    receipt = {"source": "executed", "model": "Qwen3-4B-Instruct-2507-4bit",
               "control": "one exact reviewed free-form proposal on existing B_DUP",
               "scope": "no general code execution; no capability population estimate",
               "isolation_backend": "macOS sandbox-exec", "outcome": dataclasses.asdict(outcome)}
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (args.out / "config.json").write_text(json.dumps({"source": "executed", "seed": 202,
        "world_size": 2, "job_seed": 777, "format_check": "v1.2", "edit": edit}, indent=2) + "\n")
    assert outcome.sandboxed and not outcome.ok
    print("Reviewed Qwen code proposal: sandboxed execution completed; repair rejected, sampler still not saved.")


if __name__ == "__main__":
    main()
