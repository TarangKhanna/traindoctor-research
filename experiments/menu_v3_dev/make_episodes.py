"""Convert stored dev1 records and v1.2 outcomes to the existing Modal eval cache.

No execution, training, tokenizer downloads or Modal calls. Candidate edits are
empty: this cache is for table-scored evaluation, never for live repair replay.
The stored prompt body is unchanged; eval adds its usual FORMAT/chat wrapper.

  PYTHONPATH=<library>:. TD_KEY=menu-v3-dev-1 python experiments/menu_v3_dev/make_episodes.py
"""
from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

from modal_train.core import read_jsonl, render_prompt
from modal_train.trainlib import load_eps
from traindoctor_env.bundle import REPAIRS_V3, Bundle, Candidate
from traindoctor_env.reexec import FixOutcome
from traindoctor_env.scenario import Truth
from traindoctor_env.tiera import FORMAT, NO_FAULT_DEF, apply_qwen3, prompt_text

HERE = Path(__file__).resolve().parent
DATA = HERE / "results" / "dev1"
CACHE = HERE.parents[1] / "modal_train" / "cache_v3dev"


def episode(row: dict, outcomes: dict):
    seed, prompt, ids = row["seed"], row["prompt"], row["menu_ids"]
    assert len(ids) == len(set(ids)) == 12, seed
    assert set(outcomes) == set(ids), seed
    truth = Truth(**row["truth"])
    assert truth.seed == seed

    # rl.render applies the v2 framing to a v1 Bundle. Undo only that framing,
    # then require its library-produced round trip to equal the stored prompt.
    head, labels, no_fault, rest = prompt.split("\n", 3)
    intro = "Determine whether this job has a fault. Answer with:"
    assert head.startswith(intro) and no_fault == NO_FAULT_DEF, seed
    text = "\n".join(["A distributed training job failed or misbehaved. Answer with:" + head[len(intro):], labels, rest])
    _, tail = text.split("\n" + REPAIRS_V3 + "\n", 1)
    pairs, menu, table = [], [], {}
    for i, cid in enumerate(ids):
        prefix = cid + ":\n"
        assert tail.startswith(prefix), (seed, cid)
        diff = tail[len(prefix):]
        if i + 1 < len(ids):
            diff, sep, tail = diff.partition("\n" + ids[i + 1] + ":\n")
            assert sep, (seed, cid)
            tail = ids[i + 1] + ":\n" + tail
        r = outcomes[cid]
        assert r["status"] == "ran" and r["source"] == "executed", (seed, cid)
        assert r["kind"] == truth.kind and r["label"] == truth.label, (seed, cid)
        stored = r["checks"]["v1.2"]
        assert not stored["sandboxed"] and r["exec_mode"] == "local_unsandboxed_trusted", (seed, cid)
        # The outer label describes the actual run. The nested dev_container
        # label is historical bookkeeping, not sandbox verification.
        table[cid] = FixOutcome(**{**stored, "tampered": tuple(stored["tampered"]), "exec_mode": r["exec_mode"]})
        pairs.append((cid, diff))
        menu.append(Candidate(cid, r["role"], (), r["destructive"], display=diff))
    config = text.split("== job/config.yaml ==\n", 1)[1].split("\n\n== records ==", 1)[0]
    bundle = Bundle(seed, tuple(row["records"]), {"job/config.yaml": config}, tuple(pairs), text, row["tokens_est"])
    assert prompt_text(bundle, "v2") == prompt, seed
    assert render_prompt("Qwen/Qwen3-4B", None, bundle, "v2") == apply_qwen3([
        dict(role="user", content=prompt + "\n\n" + FORMAT)]), seed
    return truth, bundle, menu, table


def main() -> int:
    records = read_jsonl(DATA / "records.jsonl")
    rows = read_jsonl(DATA / "outcomes.jsonl")
    seeds = [r["seed"] for r in records]
    assert len(seeds) == len(set(seeds)) == 60
    by_seed = {s: {} for s in seeds}
    for r in rows:
        group = by_seed[r["seed"]]
        assert r["cand_id"] not in group, (r["seed"], r["cand_id"])
        group[r["cand_id"]] = r
    eps = [episode(r, by_seed[r["seed"]]) for r in records]
    modes = Counter(o.exec_mode for _, _, _, table in eps for o in table.values())
    stats = dict(source="executed", run="dev1", menu_version="v3", prompt_version="v2", format_check="v1.2",
                 episodes=len(eps), candidates=sum(modes.values()), by_exec_mode=dict(modes),
                 stored_outcome_exec_modes=dict(Counter(r["checks"]["v1.2"]["exec_mode"] for r in rows)),
                 sandboxed=sum(o.sandboxed for _, _, _, table in eps for o in table.values()),
                 verified=False, usage="table-scored eval only; candidate edits omitted")
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "dev_eps.pkl").write_bytes(pickle.dumps(eps))
    (CACHE / "menu_version").write_text("v3\n")
    (CACHE / "prompt_version").write_text("v2\n")
    (CACHE / "dev_exec_mode.json").write_text(json.dumps(stats, indent=1, sort_keys=True) + "\n")
    assert len(load_eps(str(CACHE), "dev")) == len(eps)
    print("wrote", CACHE, json.dumps(stats), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
