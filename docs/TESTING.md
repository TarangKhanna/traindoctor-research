# What the tests establish

Test counts are not a correctness guarantee. The useful question is whether a
check detects a concrete failure, uses an appropriate expected result, and runs
at the boundary where that failure can occur.

| Layer | Covered failures | Limit |
|---|---|---|
| Protocol and edit contracts | Truncated/multiple calls, schema errors, invalid edits, stale source hashes, changed decoding settings | Does not validate a real model's native chat template |
| Deterministic controller wiring | Scripted runtime/episode doubles exercise protocol recovery, inspection, approval pause, rejected approval, resume, relaunch feedback and finish through `run.execute` | No model inference, real worker, or semantic repair is performed by this test |
| Independent numerical checks | NumPy gradients checked against finite differences; masked updates compared with PyTorch global-objective calculations, including momentum and zero-valid-rank cases | Shares the declared initialization/data recipe; only the stated MLP/optimizer contract is checked |
| Executed semantic controls | Correct loss-side/gradient-side fixes, completing-but-wrong code, skip-update and visible-overfit controls; hidden validity schedules | Author-designed cases within one mechanism, not unseen-mechanism generalization |
| Evidence integrity | Recomputed state checks plus altered-claim, omitted-model, symlink, host-path and diagnostic-source tampering tests | Recounting saved captures does not authenticate the original execution or attest live memory |
| Statistical analysis | Controlled analysis inputs and grouped-result/accounting checks | Does not remove historical task shortcuts, missing comparison arms or selection bias |

The environment repository separately has contract tests, a configuration
serialization property test, and explicit `executed`/`sweep` selections. Its
[testing record](https://github.com/TarangKhanna/traindoctor-env/blob/main/docs/TESTING.md)
preserves native-backend failures and the distinction between unit CI, fresh Gloo
execution, and sandboxed verification. Do not call all of these property-tested
distributed semantics merely because Hypothesis appears in the dependencies.

## Run the relevant checks

From this checkout with the declared CPU dependencies and library available:

```bash
PYTHONPATH=. python -m pytest -q tests/test_agent_controller_flow.py tests/test_local_agent_v1.py tests/test_masked_mean_v1.py tests/test_masked_mean_public.py tests/test_state_recovery_oracle.py
TD_KEY=state-recovery-demo-key PYTHONPATH=. python -m experiments.masked_mean_v1.public_packet --check evidence/masked_mean_v1
python scripts/check_readme.py
python scripts/check_readme_numbers.py
python experiments/menu_v3_dev/accounting.py --check
```

Required CI checks unit/analysis behavior and the portable packet. The recorded
live controls and scripted tool smoke are distinct evidence, not a fresh worker
execution on every push. Runtime/verifier changes need the narrow supported-backend
live smoke and positive/negative controls as well as unit checks. The optional
native executed suite has documented limitations; no clean full-suite claim is
made here.

## Still unproven

- A successful live model-applied edit/relaunch/feedback trajectory after the protocol fixes.
- Compatibility of those fixes with both actual native model templates under complete episodes.
- A mandatory, reliable fresh distributed-execution gate on each change.
- Independent security review, arbitrary-program correctness, or production NCCL/FSDP coverage.
- Useful same-policy reward variation, unseen-mechanism generalization, or an RL advantage over continued SFT.

The controller wiring test closes an orchestration gap. It must never be reported
as a successful model episode. Preserve the published negative trajectories and
version any future live-model experiment separately.
