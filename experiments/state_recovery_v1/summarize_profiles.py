"""Recount saved local profiles and optionally sanitize historical kernel rows."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", type=Path, nargs="+", required=True)
    ap.add_argument("--historical-bench", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("refusing to overwrite evidence")
    result = {"source": "executed", "profiles": []}
    for directory in args.profiles:
        summary = json.loads((directory / "summary.json").read_text())
        rows = json.loads((directory / "rows.json").read_text())
        for target, metrics in summary["throughput"].items():
            selected = [r for r in rows if r["kind"] == "throughput" and not r["warmup"]
                        and r["target_prompt_tokens"] == int(target)]
            assert len(selected) == 5
            for key, value in metrics.items():
                assert value == statistics.median(r[key] for r in selected)
        profile = {"directory": directory.name, "summary": summary,
                   "minimum_available_gib": min(r["available_bytes"] for r in rows) / 1024**3,
                   "max_swap_growth_bytes": max(r["swap_growth_bytes"] for r in rows)}
        if "prefix_cache" in summary:
            p = summary["prefix_cache"]
            profile["cache_group_speedup_including_setup"] = p["uncached_total_s"] / p["cached_total_including_setup_s"]
            assert p["all_outputs_match_uncached"]
        result["profiles"].append(profile)
    if args.historical_bench:
        raw = args.historical_bench.read_bytes()
        rows = [json.loads(line) for line in raw.decode().splitlines()]
        keep = ("job", "pass", "variant", "model", "G", "median_step_s", "step_s", "max_peak_gb",
                "swap_growth_mb", "mlx", "mlx_lm", "commit")
        selected = [{k: r[k] for k in keep if k in r} for r in rows if "0.6B" in r["model"]]
        stats = []
        for group in (1, 2):
            for variant in ("stock", "patched"):
                cells = [r for r in selected if r["G"] == group and r["variant"] == variant]
                stats.append({"group": group, "variant": variant, "runs": len(cells),
                              "median_of_run_median_s": statistics.median(r["median_step_s"] for r in cells),
                              "max_peak_mlx_gb": max(r["max_peak_gb"] for r in cells)})
        result["historical_kernel_bench"] = {"file": args.historical_bench.name,
            "sha256": hashlib.sha256(raw).hexdigest(), "sanitized_rows": selected, "statistics": stats,
            "scope": "historical 0.6B plumbing microbenchmarks, not a new training result or end-to-end RL speed claim"}
    args.out.mkdir(parents=True)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "config.json").write_text(json.dumps({"source": "executed", "operation": "read-only saved-profile recount",
        "profile_directories": [p.name for p in args.profiles], "historical_bench_included": bool(args.historical_bench)}, indent=2) + "\n")
    for profile in result["profiles"]:
        print(profile["directory"], "minimum_available_GiB", profile["minimum_available_gib"],
              "max_swap_growth", profile["max_swap_growth_bytes"],
              "cache_speedup", profile.get("cache_group_speedup_including_setup", "not measured"))


if __name__ == "__main__":
    main()
