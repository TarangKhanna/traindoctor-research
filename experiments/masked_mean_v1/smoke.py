"""Explicitly scripted tool-path control, never a model trajectory."""
import argparse
import os
import time
from pathlib import Path

from experiments.state_recovery_v1.reference_free_audit import write
from .controls import controls
from . import task
from .task import RepairEpisode, source_sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists() or os.environ.get("TD_KEY") != "state-recovery-demo-key":
        ap.error("require fresh output directory and public demo key")
    import psutil
    swap0 = psutil.swap_memory().used
    resources = []

    def guard():
        row = {"available_bytes": psutil.virtual_memory().available,
               "swap_growth_bytes": psutil.swap_memory().used - swap0}
        resources.append(row)
        if row["available_bytes"] < 8 * 1024**3 or row["swap_growth_bytes"] > 128 * 1024**2:
            raise RuntimeError("local resource guard")

    guard()
    gold = controls()["correct_loss"]
    episode = RepairEpisode(args.fixture, args.out, [source_sha(gold)])
    episode.pre_execution_guard = guard
    rows = []
    for action in [
        {"tool": "read_file", "args": {"path": "job/train.py"}},
        {"tool": "read_file", "args": {"path": "job/config.yaml"}},
        {"tool": "write_file", "args": {"path": "job/train.py", "content": gold}},
        {"tool": "relaunch", "args": {"resume_from": "none"}},
        {"tool": "submit_diagnosis", "args": {"label": "global_loss_normalization", "evidence_ids": [],
                                               "claimed_fixed": True, "fix_summary": "global valid-target mean"}},
        {"tool": "finish", "args": {}},
    ]:
        started = time.perf_counter()
        observation, done = episode.step(action)
        rows.append({"action": action, "observation": observation, "done": done,
                     "wall_s": time.perf_counter() - started})
        print(action["tool"], observation[:180], flush=True)
    assert episode.result["terminal_reward"] == 1
    assert len(episode.last_execution["per_step"]) == 4
    write(args.out / "receipt.json", {"source": "executed", "policy": "scripted_positive_control",
          "not_model_evidence": True, "rows": rows, "result": episode.result, "resources": resources,
          "config": {"task": "masked_mean_v1", "approved_source_sha256": source_sha(gold),
                     "task_sha256": source_sha(Path(task.__file__).read_text()),
                     "smoke_sha256": source_sha(Path(__file__).read_text())}})
    print("PASS: scripted inspect/edit/relaunch/observe/finish and hidden regressions")


if __name__ == "__main__":
    main()
