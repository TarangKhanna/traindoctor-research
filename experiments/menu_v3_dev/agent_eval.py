"""Menu v3 development experiment, step 2: strong-agent answers on the v3 prompts.

Each prompt in results/<run>/records.jsonl (prompt version v2 + the tiera answer format, the same text a trained model
sees) is sent once to Codex CLI through ~/.claude/tools/codex_bridge.sh, from an EMPTY temporary directory with a
read-only sandbox, so the agent sees the prompt only (no repository, no truth, no outcome table). The raw final message
is stored; parsing and scoring happen in analyze.py.

  python experiments/menu_v3_dev/agent_eval.py --run dev1 --agent codex-medium [--samples K] [--limit N] [--workers 4]
Appends to results/<run>/agent_<agent>.jsonl; a (seed, sample) already answered is skipped (resumable, never overwritten)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from traindoctor_env.tiera import FORMAT

HERE = Path(__file__).resolve().parent
BRIDGE = Path.home() / ".claude/tools/codex_bridge.sh"
AGENTS = {"codex-medium": ["--tier", "medium"], "codex-mini": ["--tier", "mini"]}


def ask(agent: str, prompt: str, timeout: int) -> dict:
    t = time.time()
    with tempfile.TemporaryDirectory() as empty:
        p = subprocess.run([str(BRIDGE), *AGENTS[agent], "--sandbox", "read-only", "--no-discipline", "--cd", empty,
                            "--timeout", str(timeout), "--", prompt + "\n\n" + FORMAT],
                           capture_output=True, text=True, timeout=timeout + 60, env={**os.environ})
    path = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    text = Path(path).read_text() if path and Path(path).is_file() else None
    return dict(completion=text, bridge_exit=p.returncode, secs=round(time.time() - t, 1),
                error=None if text is not None else (p.stderr or "")[-400:])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--agent", choices=sorted(AGENTS), default="codex-medium")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--samples", type=int, default=1, help="independent answers per prompt")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=240)
    a = ap.parse_args(argv)
    d = HERE / "results" / a.run
    out = d / f"agent_{a.agent}.jsonl"
    rows = [json.loads(x) for x in out.read_text().splitlines()] if out.exists() else []
    done = {(r["seed"], r.get("sample", 0)) for r in rows}
    bundles = [b for b in map(json.loads, (d / "records.jsonl").read_text().splitlines()) if "prompt" in b][:a.limit]
    todo = [(b, k) for k in range(a.samples) for b in bundles if (b["seed"], k) not in done]

    def one(job):
        b, k = job
        return dict(seed=b["seed"], sample=k, agent=a.agent, **ask(a.agent, b["prompt"], a.timeout))

    with ThreadPoolExecutor(a.workers) as ex, out.open("a") as f:
        for r in ex.map(one, todo):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(r["seed"], r["bridge_exit"], r["secs"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
