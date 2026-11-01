# Historical DEV learning-signal recount

source: executed. Saved DEV evaluation groups, not training-time rollouts. V2 is exploratory dev_container scoring.

| Reward | Positive completions | Zero-variance groups / 384 | Zero-variance fraction |
|---|---:|---:|---:|
| V0 | 1973 / 3072 | 163 | 42.45% |
| V1 | 767 / 3072 | 174 | 45.31% |
| V2 | 697 / 3072 | 179 | 46.61% |

V1 != V2: 70/3072 (2.28%).
Centered and normalized advantages differ in 20/384 groups (5.21%).
Repair acceptance conditional on V1: 697/767 (90.87%).

| Failure mechanism / repair role | V1-positive V2-negative |
|---|---:|
| A1 | decoy_shard | no_completed_relaunch | 30 |
| A1 | decoy_sync | no_completed_relaunch | 30 |
| B_SKIP | decoy_perrank | no_completed_relaunch | 1 |
| A1 | decoy_ckpt | no_completed_relaunch | 6 |
| A1 | noop | no_completed_relaunch | 2 |
| A2 | decoy_ckpt | no_completed_relaunch | 1 |

Eight contiguous completions per cached episode, separately for each policy; no groups pooled across policies. No completion reaches the 400-token cap. Centering and sample-standard-deviation normalization with epsilon 1e-4 give the same changed-group count; population-std sensitivity agrees.

Historical training configured TRL 1.16.0 with DAPO, beta=0 and eight generations. Exact training tensors, EOS/padding masks, logprobs and sampled groups were not retained here. These evaluation-group calculations are not measured gradient differences. The local inspected TRL source is a later 1.17 development snapshot; the normalized values are an explicit formula reconstruction, not an authenticated historical trainer replay.

The complete grading fixtures and per-completion scores are exported for re-scoring. Original cache/completion hashes are in config.json. No fresh sandbox or model was run.
