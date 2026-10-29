# Reference-free v1: bounded experimental specification

Written after the prior-art NO-GO and before prototype implementation, 2026-10-29.
Purpose: validate/falsify local reference independence, not claim novel replay.

## Inputs and sufficiency

Use the existing loopback-verified raw archive, SHA256
`0579811fa306765a6a5a5b1c279efaca1acdbf4e123235dbd336af4f80bc1671`.
Extract only configuration, original failed-prefix job/captures and the twelve
continuation job/capture trees. Exclude uninterrupted reference, observations,
receipts, oracle flags and summary. Outcomes are compared with old receipts only
after the checker returns. No twelve-job rerun is needed.

Available: genuine checkpoints 4 and 8; final checkpoint 12; per-rank served
IDs/positions/window numbers, acknowledgements and capture hashes; fixed job source
and configuration. The demo's specified public key and fixed salt recipe regenerate
input values. The ledger does **not** independently commit those input bytes/salt,
and has no scheduler or post-initialization stochastic model state. Mark these
boundaries explicitly; no metadata will be fabricated. No runtime extension is
required for this conditional local calculation.

An inspected catch: legacy `resume_sha256` selects a previous archived capture by
checkpoint ID, not the delivered bytes. It still names the clean step-8 capture for
the mixed resumes. Do not treat that field as loaded-byte attestation. Inspect the
new run's captured checkpoint and actual input history, with pristine-source and
single-writer assumptions; actual per-rank load-time attestation remains absent.

## Contract

A: compare observed continuation-run checkpoint components at overlapping steps
4/8 to the original committed state at that step; no original step-12 state exists.
Step-12 expectations belong only to Method B. Check step/sampler and explicit history. This is consistency
relative to trusted captured state, not intrinsic provenance from tensor values.

B: trust original checkpoint 4, replay 4→8 under the specified MLP/SGD rule, then
8→12 using the existing NumPy update implementation. Validate required per-rank
ordered history against the intended schedule; compare each continuation's
applicable captured checkpoints to that anchored computation. Accept only complete,
finite, schema-compatible, hash-valid evidence with no invariant failure.

Scratch is compared at checkpoint 4 and afterward; its initialization→4 segment is
not independently replayed. Earlier coherent rollback is valid. Neither cost nor
the existence of a robust action causes semantic rejection. Scheduler/RNG checks
are not applicable to this fixed-LR, deterministic-after-initialization job.

No reference run, verifier `ok` flag or correctness receipt is an input. Reuse of
the same NumPy kernel means agreement with the old NumPy oracle is a consistency
comparison, not another independent mathematical implementation. PyTorch execution
versus NumPy arithmetic remains the independent calculation boundary.

## Measurement and tests

- Produce a twelve-row detection matrix: completion, hashes, component/transition
  decision, failed invariants, absolute/relative discrepancies, evidence limits and
  measured checker wall time. Preserve old exact errors separately.
- Keep existing absolute tolerance `1e-5`; no fitted threshold. Relative diagnostics:
  maximum coordinate error divided by `max(abs(expected), 1e-12)`, plus normalized
  infinity error. Neither relative metric changes acceptance. Reject NaN/Inf.
- Repeat two correct CPU/Gloo continuations from the original committed step 8;
  compare both with the existing matching continuation and NumPy. Record wall time,
  backend and configuration. This establishes only local repeatability.
- Falsification 1: perturb trusted checkpoint 4, replay coherent 8/12. Relative-only
  replay should accept the poisoned chain; pinning the original anchor should
  reject. This is a false acceptance of the stronger original-trajectory claim,
  not a violation of the conditional contract.
- Falsification 2: remove or falsify a required input-history record with checkpoint
  bytes unchanged. Missing records must yield insufficient evidence; conflicting
  records must fail history consistency. Do not claim adversarial security.
- Focused tests enforce no reads of reference/receipt/verdict files, missing evidence,
  nonfinite rejection, numerical comparison semantics and the anchor counterexample.

Stop after report/prototype. No training/inference, task-family expansion, new
dependencies, integration proposal, publishing or outreach. Preserve v1.2 unchanged.
