"""Execute one verbatim model fragment as a supervised diagnostic, not agent credit."""
import argparse
import json
import re
import textwrap
from pathlib import Path

from experiments.state_recovery_v1.reference_free_audit import write
from . import task


def candidate(trajectory):
    raw = trajectory["rows"][-1]["raw"]
    blocks = re.findall(r"```python\n(.*?)\n```", raw, re.S)
    if len(blocks) != 1:
        raise ValueError("require exactly one complete Python fragment; never complete truncated code")
    fragment = blocks[0]
    original = task.source()
    if original.count(task.MASKED) != 1:
        raise ValueError("source replacement is not unique")
    # Preserve every fragment character, including comments and the proposed arithmetic.
    edited = original.replace(task.MASKED, textwrap.indent(fragment, " " * 8) + "\n")
    compile(edited, "model_diagnostic_train.py", "exec")
    return fragment, edited


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--approve-hash")
    args = ap.parse_args()
    t = json.loads(args.trajectory.read_text())
    fragment, edited = candidate(t)
    sha = task.source_sha(edited)
    if args.approve_hash is None:
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "pending.py").write_text(edited)
        write(args.out / "config.json", {"source": "executed", "policy": "supervised_diagnostic",
            "not_autonomous_agent_execution": True, "model_reward_unchanged": t["result"]["terminal_reward"],
            "trajectory_sha256": task.source_sha(args.trajectory.read_text()),
            "fragment_sha256": task.source_sha(fragment), "candidate_sha256": sha,
            "transformation": "indent the single complete code block into the existing loop; no semantic edits"})
        print("PENDING exact-source security review", sha)
        return
    cfg = json.loads((args.out / "config.json").read_text())
    if args.approve_hash != sha or cfg["candidate_sha256"] != sha or task.source_sha((args.out / "pending.py").read_text()) != sha:
        raise ValueError("exact reviewed source hash required")
    import psutil
    if psutil.virtual_memory().available < 8 * 1024**3:
        raise RuntimeError("local resource guard")
    swap0 = psutil.swap_memory().used
    result = task.execute_case(args.out / "execution", edited)
    cfg["resources"] = {"available_bytes_after": psutil.virtual_memory().available,
                        "swap_growth_bytes": psutil.swap_memory().used - swap0}
    write(args.out / "receipt.json", {**cfg, "result": result})
    print(json.dumps({"policy": "supervised_diagnostic", "completed": result["completed"],
          "error_class": result["error_class"], "semantic_valid": result["semantic_valid"],
          "model_reward_unchanged": cfg["model_reward_unchanged"]}))


if __name__ == "__main__":
    main()
