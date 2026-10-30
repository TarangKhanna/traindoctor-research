# Bounded corrected-controller feasibility trials

Owner-authorized 2026-10-30; no training or publication. This tests live agent
behavior after the incomplete-final correction, not learning or generalization.

Freeze the existing masked-loss task, semantic grader, hidden cases, model
snapshots and prompts. Reuse the existing controls and scripted smoke first.
Run one full-evidence Qwen episode, then one full-evidence Gemma episode.
Each uses temperature 0.2, seed 0, sixteen responses, 512 output tokens per
response, the existing 8192-token context rule, two relaunches and 900 seconds
of active inference. Security-only approval pauses are recorded separately.
The existing time check runs between responses, not as a hard interruption of
an in-progress generation; report measured inference time separately.
No correctness coaching or manually applied repair earns autonomous credit.

Only if Qwen hits its output cap while attempting an edit, permit one fresh,
separately labelled diagnostic with a 1024-token response cap. It is a changed
budget condition, not an identical repeat. Do not retry infrastructure stops.
The task/reward and cached model revisions remain unchanged in that condition.
The larger output reservation also reduces prompt room within the unchanged
context limit. A narrow `--max-tokens` option retains the historical 512 default
and disallows changing this setting while resuming an existing episode.

Retain the existing RAM floor, swap-growth limit, loopback communication and
fail-closed executable-source approval. Serialize models and stop an episode
on resource or isolation failure. Preserve raw model outputs, tools, patches,
execution feedback, source identities, tokens and resource measurements locally.

Report protocol validity, successful inspections, staged/applied edits,
model-requested execution, feedback delivery and subsequent model action,
completion and independent semantic correctness separately. A zero from an
exhausted generation budget does not isolate reasoning inability. Resource
failures remain null, not semantic failures. No repair-rate or RL signal estimate.

Machine-readable configuration and receipts are script-written under the ignored
`results/masked_mean_v1_protocol_corrected_20261030/` directory. Historical
receipts remain unchanged. Update status after the bounded trials and stop.

## Commands

Use the existing prepared interpreter and cached snapshot. From this checkout,
set `PYTHONPATH` to the environment and research checkouts, `TD_PYTHON` to that
interpreter, `TD_MODEL` to the frozen cached model, and `TD_OUT` to a fresh path.
The demo key below is public. Keep raw receipts local and ignored.

```bash
export TD_KEY=state-recovery-demo-key HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
uv run --no-project --offline "$TD_PYTHON" -m experiments.masked_mean_v1.controls --out "$TD_OUT/controls"
uv run --no-project --offline "$TD_PYTHON" -m experiments.masked_mean_v1.smoke --fixture "$TD_OUT/controls/fixture" --out "$TD_OUT/scripted_smoke"
caffeinate -s uv run --no-project --offline "$TD_PYTHON" -m experiments.local_agent_v1.run --task masked_mean_v1 --fixture "$TD_OUT/controls/fixture" --model "$TD_MODEL" --out "$TD_OUT/qwen_full" --temperature 0.2 --seed 0 --turns 16
```

For Gemma, select its frozen snapshot and a new `gemma_full` output. A pending
source pauses the same episode: review the exact `pending.py` and `patch.diff`
for execution safety, then resume the identical command with `--approve-hash`
and its exact approved hash. Do not correct the model's source. Only the eligible
fresh `qwen_output1024` diagnostic adds `--max-tokens 1024`; retain that argument
on any security-approval resume of the diagnostic. Never resume a terminated run.

## Subsequent resume-integrity correction

New paused episodes bind the canonical fixture path, its full plain-file tree
(including the trusted prefix history), controller source and selected task
source before episode reconstruction or patch approval. A different path, changed
tree or changed recorded code is rejected. Per-load metadata names the selected
task module. This is reproducibility checking, not malicious-owner attestation
or a hash of every transitive dependency/model-weight file.

Older paused trajectories without this binding fail closed and require a new
episode. Completed historical trajectories are left unchanged and are not
restarted. The bounded trials above predate this correction; their frozen source
identities and exact-source approvals were separately checked by the critic.
No model was rerun to test this subsequent code change.
