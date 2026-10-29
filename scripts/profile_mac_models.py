"""Offline, serialized MLX inference profiles and tiny capability controls.

Generated code is saved as text only. A separate sandboxed validation command
may execute patches only after a strict AST equivalence safety gate.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import json
import platform
import statistics
import time
from pathlib import Path

import psutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--demo", type=Path, required=True)
    ap.add_argument("--env", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--thinking", choices=("off", "default"), default="off")
    args = ap.parse_args()
    if args.out.exists() or not args.model.is_dir():
        ap.error("require an existing local snapshot and a new output directory")
    swap0 = psutil.swap_memory().used

    def guard():
        if psutil.virtual_memory().available < 8 * 1024**3:
            raise RuntimeError("resource guard: less than 8 GiB available")
        if psutil.swap_memory().used - swap0 > 128 * 1024**2:
            raise RuntimeError("resource guard: swap growth exceeds 128 MiB")

    guard()
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    mx.set_default_device(mx.gpu)
    args.out.mkdir(parents=True)
    config = {"source": "executed", "model": args.model_id, "revision": args.model.name,
              "platform": platform.platform(), "machine": platform.machine(),
              "backend": "MLX Metal", "trust_remote_code": False, "offline": True,
              "device": str(mx.default_device()), "device_info": mx.device_info(),
              "warmups_per_shape": 1, "repetitions": args.reps, "max_decode_tokens": 128,
              "sampler": "greedy", "prefix_reuse": False,
              "thinking_mode": args.thinking,
              "versions": {n: importlib.metadata.version(n) for n in ("mlx", "mlx-lm", "transformers")},
              "memory_available_before": psutil.virtual_memory().available,
              "swap_used_before": swap0, "min_available_gib": 8, "max_swap_growth_mib": 128}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    t0 = time.perf_counter()
    model, tok = load(str(args.model), tokenizer_config={"trust_remote_code": False, "local_files_only": True})
    mx.eval(model.parameters())
    mx.synchronize()
    load_s = time.perf_counter() - t0
    rows = []

    def generate(messages, limit=128, prefix_cache=None):
        guard()
        template_kw = {"enable_thinking": False} if args.thinking == "off" else {}
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kw)
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        mx.reset_peak_memory()
        mx.random.seed(0)
        started = time.perf_counter()
        first, chunks, last = None, [], None
        cache_kw = {"prompt_cache": copy.deepcopy(prefix_cache)} if prefix_cache is not None else {}
        supplied_ids = prompt_ids[-1:] if prefix_cache is not None else prompt_ids
        for response in stream_generate(model, tok, supplied_ids, max_tokens=limit,
                                        sampler=make_sampler(temp=0), **cache_kw):
            if first is None:
                first = time.perf_counter() - started
            chunks.append(response.text)
            last = response
        mx.synchronize()
        wall = time.perf_counter() - started
        guard()
        row = {"source": "executed", "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
               "prompt_tokens": len(prompt_ids), "generation_tokens": last.generation_tokens,
               "ttft_s": first, "wall_s": wall, "prefill_tokens_s": last.prompt_tps,
               "decode_tokens_s": last.generation_tps, "peak_mlx_gb": last.peak_memory,
               "rss_bytes": psutil.Process().memory_info().rss,
               "available_bytes": psutil.virtual_memory().available,
               "swap_growth_bytes": psutil.swap_memory().used - swap0,
               "finish_reason": last.finish_reason, "text": "".join(chunks)}
        mx.clear_cache()
        return row

    # Same byte-level trace vocabulary; native token lengths are recorded rather
    # than treating different tokenizers' tokens as interchangeable units.
    trace = "rank=0 step=8 checkpoint=ck0008 committed; sampler_epoch=0 sampler_index=8\n"
    instruction = "Explain the checks needed before resuming distributed training. Give at least 250 words.\n"
    profile_messages = {}
    for target in (2048, 8192):
        repeats = max(1, target // len(tok.encode(trace)))
        messages = [{"role": "user", "content": instruction + trace * repeats}]
        profile_messages[target] = messages
        for rep in range(-1, args.reps):
            row = generate(messages)
            row.update(kind="throughput", target_prompt_tokens=target, repetition=rep, warmup=rep == -1)
            rows.append(row)
            print(args.model_id, target, rep, "tokens/s", round(row["decode_tokens_s"], 2), flush=True)
            (args.out / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")

    # A supported-runtime prefix-cache experiment, not a training optimization.
    # Cache only this exact prompt, clone per answer, and check output parity.
    messages = profile_messages[2048]
    template_kw = {"enable_thinking": False} if args.thinking == "off" else {}
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kw)
    ids = tok.encode(prompt, add_special_tokens=False)
    prefix = make_prompt_cache(model)
    t0 = time.perf_counter()
    for start in range(0, len(ids) - 1, 2048):
        model(mx.array([ids[start:min(start + 2048, len(ids) - 1)]]), cache=prefix)
        mx.eval([c.state for c in prefix])
    mx.synchronize()
    prefix_setup_s = time.perf_counter() - t0
    expected_text = next(r["text"] for r in rows if r["kind"] == "throughput"
                         and r["target_prompt_tokens"] == 2048 and not r["warmup"])
    for rep in range(args.reps):
        row = generate(messages, prefix_cache=prefix)
        row.update(kind="prefix_cache", repetition=rep, text_matches_uncached=row["text"] == expected_text)
        rows.append(row)
        print(args.model_id, "prefix cache", rep, "parity", row["text_matches_uncached"], flush=True)
    del prefix
    mx.clear_cache()

    tool_intro = (
        'Investigate this training recovery. Output exactly one JSON object, no markdown: '
        '{"tool":"NAME","args":{}}. Read tools: job_status, inspect_checkpoint, committed_history. '
        'Recovery tool: {"tool":"recover","args":{"action":"ACTION"}}. '
        'Actions: resume, restore_sampler, restore_optimizer, restore_full, older, scratch. '
        'Restore actions use the genuine committed step-8 archive. Correctness is required; then minimize redone '
        'optimizer steps. Full restore is allowed. Source code must stay unchanged. '
        'Use inspection if needed. Tool results will be supplied as user messages.')
    capability = {}
    for name in ("case0", "case1"):
        obs = json.loads((args.demo / name / "observations.json").read_text())
        messages = [{"role": "user", "content": tool_intro + "\nInitial status:\n" + json.dumps(obs["status"])}]
        calls, selected, error = [], None, None
        for turn in range(6):
            row = generate(messages, 256)
            row.update(kind="multi_turn", case=name, turn=turn)
            rows.append(row)
            try:
                call = json.loads(row["text"])
                if not isinstance(call, dict):
                    raise ValueError("tool call is not an object")
                name_tool, tool_args = call["tool"], call.get("args", {})
                if not isinstance(tool_args, dict):
                    raise ValueError("args is not an object")
                if name_tool == "recover":
                    selected = tool_args["action"]
                    if selected not in obs["status"]["actions"]:
                        raise ValueError("unknown recovery action")
                    calls.append(call)
                    break
                response = {"job_status": obs["status"], "inspect_checkpoint": obs["checkpoint"],
                            "committed_history": obs["history"]}[name_tool]
                calls.append(call)
                messages += [{"role": "assistant", "content": row["text"]},
                             {"role": "user", "content": "Tool result:\n" + json.dumps(response)}]
            except (ValueError, KeyError, TypeError) as exc:
                error = str(exc)
                selected = None
                break
        scored = json.loads((args.demo / name / selected / "receipt.json").read_text()) if selected else None
        capability[name] = {"calls": calls, "selected": selected, "format_error": error,
                            "semantic_valid_from_executed_table": scored["semantic_valid"] if scored else None,
                            "cost_score_from_executed_table": scored["cost_score"] if scored else None,
                            "evaluation": "one greedy trajectory, cached actual controls; not a live-agent benchmark"}

    pristine = (args.env / "jobs/template/ckpt.py").read_text()
    clean_line = "        state.update(sampler=sampler.state(), step=torch.tensor([step]))"
    buggy = pristine.replace(clean_line, "        state.update(step=torch.tensor([step]))")
    assert buggy != pristine
    question = ('Fix this checkpoint implementation. After restart, training repeats data because the sampler cursor '
                'is not saved. Return one source edit as strict JSON {"path":"job/ckpt.py","old":"...","new":"..."}. '
                'The old text must occur exactly once. Preserve optimizer state and existing behavior.\n' + buggy)
    code = generate([{"role": "user", "content": question}], 512)
    code.update(kind="code_repair")
    rows.append(code)
    code_result = {"format_valid": False, "applies": False, "ast_equivalent_to_pristine": False,
                   "executed": False, "note": "text only here; separate sandboxed validation required"}
    try:
        edit = json.loads(code["text"])
        code_result["format_valid"] = isinstance(edit, dict) and set(edit) == {"path", "old", "new"}
        if code_result["format_valid"] and edit["path"] == "job/ckpt.py" and buggy.count(edit["old"]) == 1:
            patched = buggy.replace(edit["old"], edit["new"])
            code_result["applies"] = True
            code_result["ast_equivalent_to_pristine"] = ast.dump(ast.parse(patched)) == ast.dump(ast.parse(pristine))
            (args.out / "code_edit.json").write_text(json.dumps(edit, indent=2) + "\n")
    except (ValueError, TypeError, KeyError, SyntaxError):
        pass
    (args.out / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    measured = [r for r in rows if r["kind"] == "throughput" and not r["warmup"]]
    summary = {"source": "executed", "model": args.model_id, "load_s": load_s,
               "multi_turn": capability, "code_repair": code_result,
               "throughput": {str(n): {
                   k: statistics.median(r[k] for r in measured if r["target_prompt_tokens"] == n)
                   for k in ("ttft_s", "wall_s", "prefill_tokens_s", "decode_tokens_s", "peak_mlx_gb", "prompt_tokens")}
                   for n in (2048, 8192)},
               "swap_growth_bytes": psutil.swap_memory().used - swap0}
    cached = [r for r in rows if r["kind"] == "prefix_cache"]
    uncached = [r for r in measured if r["target_prompt_tokens"] == 2048]
    summary["prefix_cache"] = {"setup_s": prefix_setup_s, "n": len(cached),
        "all_outputs_match_uncached": all(r["text_matches_uncached"] for r in cached),
        "uncached_total_s": sum(r["wall_s"] for r in uncached),
        "cached_total_including_setup_s": prefix_setup_s + sum(r["wall_s"] for r in cached),
        "cached_median_ttft_s": statistics.median(r["ttft_s"] for r in cached)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
