"""Turn a modal_train eval result (run_eval on the menu-v3 cache built by make_episodes.py) into results/<run>/agent_<name>.jsonl.

trainlib.run_eval emits g completions per episode, in the episode order of the uploaded cache, without seeds; the seed
of row i is the seed of episode i // g of modal_train/<cache>/dev_eps.pkl, and every row's kind must match that
episode's kind (checked). Only the raw completion text is kept: scoring is done by analyze.py, never by the eval's own
summary. The cache is the local pickle this repository built (make_episodes.py), not external data.

  PYTHONPATH=<library>:. python experiments/menu_v3_dev/import_modal_eval.py --run dev1 --eval <v3dev_x_eval.json> --name qwen3-4b-csft"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--cache", default=str(HERE.parent.parent / "modal_train" / "cache_v3dev" / "dev_eps.pkl"))
    a = ap.parse_args(argv)
    out = HERE / "results" / a.run / f"agent_{a.name}.jsonl"
    if out.exists():
        sys.exit(f"{out} exists; results are never overwritten")
    ev = json.loads(Path(a.eval).read_text())
    eps = pickle.loads(Path(a.cache).read_bytes())
    g, rows = ev["g"], ev["completions"]
    if len(rows) != g * len(eps):
        sys.exit(f"{len(rows)} completions for {len(eps)} episodes x g={g}")
    lines = []
    for i, r in enumerate(rows):
        truth = eps[i // g][0]
        if r["kind"] != truth.kind:
            sys.exit(f"row {i}: kind {r['kind']} != episode kind {truth.kind}")
        lines.append(json.dumps(dict(seed=truth.seed, sample=i % g, agent=a.name, completion=r["text"],
                                     source=f"modal_train eval {ev.get('run_id')} adapter={ev.get('adapter')}")))
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
