# Coding-agent evidence recount, 2026-11-02

source: executed. This is a new analysis of historical executions, not new model inference.

README Table B separates three studies: checkpoint-format agent-v2, frozen original
masked-loss episodes, and the later protocol-corrected masked-loss controller.
The [checkpoint report](AGENT_V2_REPORT.md) and [original portable packet](../evidence/masked_mean_v1/REPORT.md)
remain unchanged. Corrected-controller inputs are now preserved as
[deidentified complete trajectories, frozen controller source and configurations](../results/agent_recount_20261102/).
[Recounted metrics](../results/agent_recount_20261102/summary.json) and
[original/derivative hashes](../results/agent_recount_20261102/manifest.json) make the new rows inspectable.
Only host paths were replaced. Original local files were not modified. A derivative
hash does not authenticate the original run or establish independent reproduction.

## What the later runs establish

| Condition | Responses | Native valid calls | Applied edits | Agent relaunches | Final result |
|---|---:|---:|---:|---:|---|
| Qwen 512 | 16 | 14 | 1 | 1 | IndentationError; reward 0 |
| Gemma 512 | 9 | 8 | 1 | 0 | Context budget exhausted; reward 0, execution untested |
| Qwen 1024 | 16 | 16 | 1 | 2 | AttributeError; reward 0 |

The model snapshots are Qwen3-4B-Instruct-2507-4bit revision
`50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b` and Gemma-4-E2B-it-4bit revision
`238767527555cb75a05732a84dff5d6ba0dd6809`. Both use temperature 0.2, seed 0,
thinking disabled, 16 response turns, 8192 context, two relaunches, a 900-second
active-inference limit checked between responses, 8 GiB minimum available RAM and
128 MiB maximum swap growth. MLX/MLX-LM versions are retained in each trajectory.
The frozen configuration identifies research `ed262d0` and environment `221096d`;
controller/task hashes record the actual uncommitted trial source, rather than
incorrectly attributing it to those base commits. Frozen controller bytes are
included; task source is unchanged in this checkout and its hash is checked.

Qwen512 executed its own edit, received failure feedback, and selected another
inspection. Gemma's first literal replacement was inapplicable; its later line edit
was applied, but there was no relaunch. Qwen1024 applied a native edit, executed it,
received feedback, attempted a stale-hash edit that was rejected, and reran unchanged
code. Neither Qwen candidate completed; neither is a completing-but-semantically-wrong
model repair. Their serialized execution records classify `PYTHON_ERROR` separately
from semantic rejection and infrastructure interruption. Exact exception names are
visible in the saved log observations. Later terminal budget handling can omit
`actual_continuation`; recount uses `last_execution` as well.

The larger-output condition is a separate budget diagnostic. The first ten Qwen
responses have identical token IDs. At the first capped response, the prompt and
first 512 tokens match, and the larger budget permits a complete edit call. Later
states diverge; reserving 1024 tokens also reduces prompt room. This supports a
specific serialization-cap explanation, not a repair-success or latency comparison.
No same-policy reward groups or inspection ablation were sampled.

Applied edits require an applicable edit, a recorded tool observation and safety
approval. No reviewer supplied repaired code. Relaunch counts come from the agent's
native calls and feedback, not the separately executed original/normalized proposals
in agent-v2. Checkpoint Qwen retest remains **null/unscored** after a resource guard;
masked-loss budget exhaustion remains **zero** under its frozen contract.

## Task gate and reproduction

The existing global-count loss patch and gradient-normalization patch each pass
all six saved schedules. Unchanged code completes but fails unequal-count cases;
a visible-schedule hardcode passes visible and fails hidden cases. The portable
packet recomputes all 36 control outcomes and 144 captured states. These controls
validate the semantic oracle, not task difficulty. The same global-count patch is
correct for every schedule: source and the declared objective suffice, so useful
inspection benefit or conditional-choice reward variation is unproven. No further multi-turn model or training run follows. The separate
[source-only capability checks](../LEARNING_SIGNAL_AUDIT.md#capability-floor-masked-loss)
fail the formula/patch floor before tools. The scripted successful edit/relaunch/finish smoke
is an infrastructure control only.

With the environment checkout importable, from the research root:

```bash
uv run --no-project python scripts/recount_coding_agents.py --check results/agent_recount_20261102
TD_KEY=state-recovery-demo-key uv run --no-project python -m experiments.masked_mean_v1.public_packet --check evidence/masked_mean_v1
uv run --no-project python scripts/audit_learning_signal.py --check results/learning_signal_20261102_verified
```

Use a prepared interpreter/`PYTHONPATH` when the library is not installed. Commands
perform no model inference, worker launch or network access. They re-score saved
evidence, not live-memory attestation; rank0 capture fidelity remains conditional
on exact-source review. Fresh distributed execution was not repeated in this audit.
