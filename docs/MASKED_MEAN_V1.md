# One masked-loss repair demonstration

Owner-authorized bounded follow-up, 2026-10-30. This supersedes the earlier stop
on task expansion only for this incident and the frozen local episodes. No
training, dependencies, cloud runs, publishing or framework work is authorized.

The task's objective is global mean squared error over valid scalar targets.
Two CPU/Gloo ranks process fixed physical batches with deterministic validity
patterns. The unchanged implementation averages local means. Source repair is
required; data, configuration, model, SGD momentum and checkpoint behavior remain
fixed. This is an authored reproducer, not a documented production incident.

The fixture and independent NumPy update recipe live in
`experiments/masked_mean_v1/task.py`. Existing external captures record every
optimizer step using `ckpt_every=1`. Compare rank0 parameters, momentum and sampler
position against the independent recipe with absolute tolerance 1e-5 and zero
relative tolerance. Hidden cases test balanced, reversed, zero-local-count,
changing and all-valid patterns. The global count is always positive. Balanced
and all-valid cases are regression gates, not evidence that unchanged code fails.

Captures are job-serialized snapshots, not independent samples of live optimizer
or device memory. Fidelity rests on exact-source security review, not runtime
attestation. Served inputs show delivery; consumption is inferred from the oracle
match. Non-rank0 state divergence is unobserved. No hardware/kernel attestation is
claimed. Review rejects checkpoint/protocol forgery, bypasses, and unrelated file,
network or process access. Only reviewed training source executes in the existing
fail-closed sandbox. A source hash must bind approval, execution and final grading.

Terminal repair reward is binary: current-source execution plus semantic success
on visible and hidden cases. Model-selected relaunch and received feedback are
reported separately. Diagnosis, formatting, citations and cost earn no partial
reward. Infrastructure/resource interruptions are null; genuine failure or budget
exhaustion is zero. Finish never manufactures a missing agent relaunch.

Freeze Qwen and Gemma at temperature 0.2, seed 0, thinking disabled, sixteen
responses, 512 output tokens/response, 8192 total context with output reserved,
two relaunches and fifteen minutes of active inference. Security-review pauses
are separate. Qwen full then its reduced arm only if its full run relaunches and
observes; then Gemma under the same per-model gate. At most four episodes, no
automatic replacement. Reduced arms hide runtime count/gradient telemetry across
all tool routes but retain source/configuration. Source-only diagnosis remains
possible. These episodes are not repair-rate, causal-effect or RL-group estimates.

Controls and an explicitly scripted smoke trajectory precede models. No model
runs if either gate fails. Retain at least 8 GiB available memory, limit swap
growth to 128 MiB, serialize inference/heavy execution and retain raw receipts.
Stop after this demonstration. Benchmark generalization and learning remain
unestablished; future training needs structural heldouts and genuine policy data.

## Reproduction

Use the existing prepared Python 3.12 environment and cached model snapshots; no
installation or network access is performed by these commands. Set `TD_ENV` and
`TD_RESEARCH` to the environment/research checkouts and `TD_PYTHON` to that
environment's interpreter. Invoke from the research checkout:

```bash
export PYTHONPATH="$TD_ENV:$TD_RESEARCH"
export UV_CACHE_DIR=/tmp/td-masked-uv
export TD_KEY=state-recovery-demo-key
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
uv run --no-project --offline "$TD_PYTHON" -m pytest -q tests/test_masked_mean_v1.py tests/test_local_agent_v1.py tests/test_state_recovery_oracle.py
uv run --no-project --offline "$TD_PYTHON" -m experiments.masked_mean_v1.controls --out results/new-masked-controls
uv run --no-project --offline "$TD_PYTHON" -m experiments.masked_mean_v1.smoke --fixture results/new-masked-controls/fixture --out results/new-masked-smoke
caffeinate -s uv run --no-project --offline "$TD_PYTHON" -m experiments.local_agent_v1.run --task masked_mean_v1 --fixture results/new-masked-controls/fixture --model "$TD_MODEL" --out results/new-masked-episode --temperature 0.2 --seed 0 --turns 16
```

`TD_KEY` above is the public demonstration recipe, not a private credential.
`TD_MODEL` must point to an already cached snapshot. A staged patch exits pending
review; inspect `pending.py` and `patch.diff`, then repeat the same episode command
with `--approve-hash` and the exact approved source hash. Approval is security-only,
not repair coaching. An infrastructure-terminated episode cannot be resumed.

The saved episodes used frozen source `e14a0f5`. Later protocol fixes reject a
clipped final response or standalone tool-role delimiter instead of accepting it
as a natural final. Those fixes were regression-tested without rerunning either
model. Use the frozen commit to inspect historical controller behavior; running
the updated controller is a fresh experiment, not reproduction of its outcome.

The final report and raw receipts are in
`results/masked_mean_v1_final_20261030/`. Their archives preserve the original
control matrix, corrected overfit-control recheck, scripted smoke, actual model
episodes and separately labelled supervised diagnostic. Input and archive hashes
permit recounting without inference. The earlier fixed-count baseline was not a
true visible-overfit control; its receipts remain intact, while the final matrix
uses the subsequently executed visible-schedule-hardcoded control.
These fresh full-path receipts remain local and Git-ignored under the repository's
public-hygiene rule. The scripts reproduce the task without them. No evidence
export or publication is performed by this milestone.
