# modal_train: CUDA port of SFT / GRPO / eval

Core packages never import Modal. The macOS MLX path (`rl/*.py`) is separate. This directory imports `rl.render`, `rl.first_update` (classify, strip_special) and `traindoctor_env.tiera.reward` (strict gate, V0/V2); prompts are produced only through `rl.render` (HF Qwen3 ids alias to the MLX ids so the same thinking-off template is used).

Pins: modal client 1.3.5, torch 2.14.1, transformers 5.20.1, trl 1.16.0, peft 0.21.2, accelerate 1.15.0, Python 3.12 `debian_slim` image. GPU strings `A100-80GB` (default) / `H100` (`TD_GPU=H100`).

## Files
`app.py` (image, volumes, functions, entrypoints) · `trainlib.py` (TRL/PEFT code) · `core.py` (pure glue) · `prepare.py` (local episodes) · `cpu_app.py` (CPU scoring and sandbox replay) · `sandbox_backend.py` (Modal Sandbox replay backend) · `budget.py` + `rates.json` (cost guard) · `run.sh`, `run_cpu.sh` (launchers).

## Flow
1. Local, CPU, no Modal: `python -m modal_train.prepare --dev 48 --train 64` (real re-execution; DEV and TRAIN seeds only, asserted disjoint from the sealed TEST seeds). Writes `modal_train/cache/` (gitignored).
2. `modal_train/run.sh 300 upload`, then `run_smoke`, `run_sft "--run-id sft1"`, `run_grpo "--run-id g_v2 --arm V2 --sft-run sft1"`, `run_eval "--run-id e1 --adapter-run g_v2"`, then `modal run modal_train/app.py::download`.

Evaluation uses HF `generate` (sampling T=1.0, top_p 0.95), not the MLX sampler, so numbers are not directly comparable to MLX runs. `run.sh` forwards extra arguments as one string; quote as shown. The training and evaluation runs behind the published tables were executed on Modal; the run scripts are included, the per-run logs are not.

## Library source in the image

`app.py` adds `traindoctor_env`, `jobs`, `rl` and `modal_train` with `add_local_python_source` (Python files only). `rl` and `modal_train` are this repository. `traindoctor_env` and `jobs` are not in this repository since the split; they come from the library commit pinned in `requirements.txt` (`0b0a36b`), so run `pip install -r requirements.txt` in the environment that launches `modal run`.

- Historical: the published runs were built before the split, when both packages sat in the same checkout as `modal_train/`.
- Supported now: the pinned-library install above. Static check, no Modal build: both packages exist at that commit, every file under them is a `.py` file (so the extension filter in `_exclude` drops nothing), and every library module imported by this repository's `rl/`, `modal_train/`, `scripts/` and `tests/` exists there.
- Unverified (needs a paid Modal build): that the image builds and the functions import the library inside the container. The path-based exclusion filter was not exercised on an installed-package location.
