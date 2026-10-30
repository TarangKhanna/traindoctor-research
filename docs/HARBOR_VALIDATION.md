# Harbor interoperability validation — 2026-10-30

**GO for one local interoperability smoke task. NO-GO for a benchmark, Harbor Hub release or RL training.**

Actual Harbor Docker execution accepts the authored correct repair and rejects no-op, completing-invalid and forged-evidence controls. These are scripted controls, not autonomous model repairs or independent human validation.

## Provenance

Environment: `demo/harbor-v1`, baseline `cc69e14` (public base `c5a8a9b`); research: `demo/agent-v2`, baseline `308ab3d` (public head observed `4eea735`; not its branch base). Final executed source hashes are in `source-proof-hardened.json`; file/archive manifests identify the executed runtime diff independently of later closure commits. Core `jobs/` and `traindoctor_env/` are unchanged by this milestone.

Harbor 0.25.0 from official commit `509b119db0d97b141dcd9c80f4f7ab861c47c2d6`, schema 1.4. Docker 29.5.3 / Docker Desktop 4.79.0 (230596), Linux arm64. Container Python 3.12.16, Torch 2.14.1+cpu, NumPy 2.5.4. Exact installed packages and image configuration/digests are saved; images are local and unpublished.

## Executed results

| Control | Completes | Valid checkpoint hashes | Reward | Native verdict agrees |
| --- | --- | --- | ---: | --- |
| oracle-hardened-proof | yes | True | 1 | True |
| nop-hardened | yes | True | 0 | True |
| invalid-hardened | yes | True | 0 | True |
| tamper-hardened | yes | True | 0 | True |

The no-op and tamper controls have 32 sample gaps and 32 duplicates. The completing-invalid repair fixes sample persistence (zero gaps/duplicates) but discards optimizer momentum; transition/state checks reject it. Thus completing execution and valid checksums are insufficient for semantic acceptance. All four hardened controls ran against the final stale-reward cleanup code.

Native Mac controls use the exact same scenario and source bytes. Every shared v1.2 verdict field agrees. This is cross-backend consistency of the same verifier, **not independent mathematical validation**. Prior NumPy/state-recovery evidence remains separate.

## Isolation and failure handling

- Actual Docker configuration: separate agent/verifier containers and disjoint named volumes, read-only roots, network mode `none`, 4 GiB and 256 PID limits, no privileged containers or host/Docker-socket mounts.
- Agent uid 1000; submitted verifier workers have all UID/GID slots 65534, empty groups, zero capabilities and no-new-privileges. Root-owned private ledger access is denied to workers; launcher preemption is tested. Only loopback is UP; dormant kernel interfaces are allowed, with no NET_ADMIN and an external-connect denial.
- Forging agent-side reward/ledger files does not affect scoring. Attempting to write installed scoring code is denied. Only bounded regular `ckpt.py` is consumed; supplied evidence is not trusted. This is an author-generated adversarial smoke test, not a general security proof.
- A real initial interface-preflight failure is retained as `oracle-volume-fix`: infrastructure error, null reward and neither reward format emitted. CLI exit 0 alone is not sufficient; recount checks actual trial/result files. Unit regression additionally seeds both stale success formats and proves both are removed before an infrastructure error.
- Default Harbor lint retains one `compose-additional-capabilities` finding. Narrow reviewed lint ignoring only that category returns `[]`. Root capabilities are required for privilege dropping and preemption; workers have none. No blanket lint-pass claim.
- The provider-level network policy says `public` because this Docker backend rejects native `no-network`. Actual confinement comes from both Compose files plus the runtime proof, not Harbor policy enforcement.

## Performance and resource accounting

One final run per control, warm build cache, shared laptop with other research running. These timings are not independent repeats or a speed comparison. End-to-end includes Harbor/container setup and artifact transfer; verifier time includes preflight, recovery, fault-free reference and checking.

| Control | Total s | Setup s | Verifier s | Recovery s | Reference s | Check s | Peak MiB | Host min GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle-hardened-proof | 66.212 | 14.727 | 43.411 | 17.399 | 8.633 | 0.07468 | 595.62 | 9.85 |
| nop-hardened | 219.760 | 84.283 | 130.023 | 22.545 | 8.698 | 0.07492 | 595.55 | 8.44 |
| invalid-hardened | 58.136 | 15.707 | 34.938 | 13.432 | 8.364 | 0.03439 | 596.01 | 9.15 |
| tamper-hardened | 75.240 | 19.246 | 49.127 | 18.634 | 11.247 | 0.05605 | 595.49 | 9.38 |

Maximum recorded host swap growth: 0 bytes. All runs retained at least 8 GiB available. Full one-second host samples and container cgroup counters are saved. Native oracle/no-op/invalid wall times are recorded in `native/summary.json`; different platform/load/cache conditions prevent attributing their difference solely to Harbor.

## Reproduction and evidence

From the research repository, recount the saved final packet without Docker or model inference:

```bash
python3 -m experiments.harbor_smoke.recount \
  --run results/harbor_validation_20261030 --series hardened \
  --out /tmp/traindoctor-harbor-recount.json
```

Live setup/run commands are in the environment repository’s `docs/HARBOR.md`. For completing-invalid and tamper trials, install this research repository on `PYTHONPATH` in the isolated Harbor environment and pass `-a experiments.harbor_smoke.controls:CompletingInvalidAgent` or `-a experiments.harbor_smoke.controls:TamperNopAgent`. Use fresh output directories and serialize trials. `execution/run_control.py` and `execution/native_compare.py` preserve the exact commands/configuration used here; their absolute paths identify the originating worktrees and must be adjusted for another machine.

Checks: 20 adapter tests pass (`tests.xml`); reviewed native lint returns no remaining findings (`lint-reviewed-final.json`); recount validates four controls, native source/verdict agreement, isolation, infrastructure-null handling and 16 committed capture hashes. Exact terminal logs, trial JSON, submitted patches, trusted execution evidence, image IDs/descriptors, transport discriminator and cleanup receipts are retained in `results/harbor_validation_20261030/`.

## Reproduced fixes and limits

Runtime validation fixed: unsupported provider network declaration; read-only upload directories; Docker Desktop tmpfs/docker-cp transport mismatch (replaced transported paths with scoped named volumes); false rejection of dormant kernel interfaces; and stale `reward.txt` fallback after infrastructure failure. Original verifier v1.2 semantics and historical receipts remain unchanged. Earlier failed/final runs remain preserved rather than overwritten.

The task still reconstructs a fresh diagnostic preemption to evaluate submitted source, uses a fault-free reference, and exposes a shallow source-mutation repair. It does not grade an agent’s original recovery trajectory or establish a discriminative RL task. The validation run used no model inference or training, added no task, and performed no push, publication or Hub upload. External-machine reproduction and stronger third-party security review remain outstanding. Local image identities and package versions are recorded; this is not a portable bit-reproducible image claim.

**Highest-value next action:** have one independent engineer reproduce these four local Harbor controls using the saved task and receipts before expanding scope. No outreach was sent.
