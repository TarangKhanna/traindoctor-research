# Reference-free gate — completed before experimental implementation

2026-10-29. Verdict: **NO-GO for novelty of checkpoint-to-checkpoint replay.**
Proceed only with a bounded reproduction/falsification of the existing captures.
No new experimental implementation preceded this decision. Public source files
were inspected, not executed; revision/content hashes are recorded separately.

## Prior-art matrix

| System / evidence | What is established | Boundary relevant to TrainDoctor |
|---|---|---|
| [PyTorch DCP documentation](https://docs.pytorch.org/docs/main/distributed/checkpoint.html) | Distributed save/load, resharding, canonical model/optimizer parameter names and structural checks. Callers supply stateful objects. | Structural/key consistency does not establish the historical step at which arbitrary momentum tensors originated. Application completeness still matters. This is a scope distinction, not evidence that every DCP application lacks semantic checks. |
| [TorchTitan checkpointer](https://github.com/pytorch/torchtitan/blob/2db55136735f35ea77d629115890c7fa40decb02/torchtitan/components/checkpointer/base.py), [DCP implementation](https://github.com/pytorch/torchtitan/blob/2db55136735f35ea77d629115890c7fa40decb02/torchtitan/components/checkpointer/dcp.py) | Bundles model, optimizer, LR scheduler and supplied extra states; supports intentional partial loads. Waits for asynchronous staging before mutating the source again; discovers published DCP metadata. | Bundling, staging and loadability address different risks from independently recomputing an optimizer transition. They already prevent some inconsistency classes. Do not equate metadata publication with a proof of training semantics. |
| [TorchTitan checkpoint compatibility test](https://github.com/pytorch/torchtitan/blob/2db55136735f35ea77d629115890c7fa40decb02/scripts/checkpoint_compat_test.py), [integration configurations](https://github.com/pytorch/torchtitan/blob/2db55136735f35ea77d629115890c7fa40decb02/tests/integration_tests/features.py) | Runs a reference trajectory, saves/resumes, compares extracted losses; save/load integration tests exist. | Uninterrupted-vs-resumed checking is already established test practice. These tests are not evidence of automatic full-state production validation. |
| [TorchTitan SDC replay implementation](https://github.com/pytorch/torchtitan/blob/2db55136735f35ea77d629115890c7fa40decb02/torchtitan/observability/sdc_replayer.py), [CPU tests](https://github.com/pytorch/torchtitan/blob/2db55136735f35ea77d629115890c7fa40decb02/tests/unit_tests/cpu/observability/test_sdc_replayer.py) | Optional in-training deterministic replay compares gradients, buffers and RNG effects; tests inject gradient/buffer/RNG divergence. | It checks reproducibility of forward/backward execution, not provenance of preexisting optimizer momentum. A consistently wrong restore can reproduce itself. Source inspection supports this narrow boundary; deployment prevalence is unknown. |
| [torchdata StatefulDataLoader](https://github.com/meta-pytorch/data/blob/ee79a399d3a53676544e0d3f0e826ffa58d1fec9/torchdata/stateful_dataloader/README.md), [state tests](https://github.com/meta-pytorch/data/blob/ee79a399d3a53676544e0d3f0e826ffa58d1fec9/test/stateful_dataloader/test_state_dict.py) | Mid-epoch state, sampler/dataset hooks, worker/RNG restoration and interrupted iteration tests. Worker aggregation is not rank aggregation. | Gives application components resumable state; does not independently attest its relationship to model/optimizer state. Cross-rank coordination remains the application's responsibility in this documented API. |
| [DeepSpeed engine](https://github.com/deepspeedai/DeepSpeed/blob/12ffb990341f021ea9847d86a98ba91c81f77fdb/deepspeed/runtime/engine.py), [checkpoint comparison helpers](https://github.com/deepspeedai/DeepSpeed/blob/12ffb990341f021ea9847d86a98ba91c81f77fdb/tests/unit/checkpoint/common.py), [scheduler tests](https://github.com/deepspeedai/DeepSpeed/blob/12ffb990341f021ea9847d86a98ba91c81f77fdb/tests/unit/checkpoint/test_lr_scheduler.py) | Saves/restores partitioned model/optimizer, scheduler and progress; client state is extensible. Tests compare tensors, optimizer dictionaries and counters; tag validation coordinates rank labels. | Checks exceed file integrity. A matching rank tag does not prove numerical lineage. Ordinary user data/RNG completeness depends on supplied client state and integration. No claim of a repository-wide absence of replay is made. |
| [Megatron checkpointing](https://github.com/NVIDIA/Megatron-LM/blob/5de526f232464e1b1ff1d6645fe2c4aaf01bc0d8/megatron/training/checkpointing.py), [rerun engine](https://github.com/NVIDIA/Megatron-LM/blob/5de526f232464e1b1ff1d6645fe2c4aaf01bc0d8/megatron/core/rerun_state_machine.py), [rerun checkpoint tests](https://github.com/NVIDIA/Megatron-LM/blob/5de526f232464e1b1ff1d6645fe2c4aaf01bc0d8/tests/unit_tests/dist_checkpointing/test_rerun_state_machine_ckpt.py) | Saves optimizer, scheduler, iteration/consumed samples, Python/NumPy/torch/CUDA RNG and rerun state. Runtime reruns investigate reproducibility of selected results, including checkpoint-based restarts. | There are already in-training diagnostics, not just CI. Such diagnostics are not a general proof that a coherent but wrong algorithm is intended. Exact production enablement/use was not established. |
| [OLMo-core loader](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/olmo_core/data/data_loader.py), [checkpoint implementation](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/olmo_core/train/checkpoint.py) | Loader checks dataset fingerprint/seed and tracks processed tokens; checkpoint code manages model/optimizer and trainer state. | Runtime metadata checks already detect some incompatible resumes. They cannot infer missing tensor lineage from a step label alone. |
| [Proof-of-Learning, Jia et al., IEEE S&P 2021](https://arxiv.org/html/2103.05633), [author implementation verify.py](https://github.com/cleverhans-lab/Proof-of-Learning/blob/main/verify.py) | `verify_all` loads earlier checkpoints, replays logged batch indices and compares the next checkpoint; sampled verification is also implemented. No uninterrupted reference run is required. | **Direct prior art for the central proposed mechanism.** Recovery diagnosis and optimizer/data checks may be useful applications; they do not make transition replay a new method. This audit does not endorse the paper's security claims. |
| [Oobleck, SOSP 2023](https://arxiv.org/abs/2309.08125) | Uses replicated training state and pipeline templates for failure recovery. | Recovery availability/throughput is an established systems problem; it is distinct from an independent semantic checker. No claim that TrainDoctor improves Oobleck's recovery performance. |

### Documented failures, not invented production incidents

[OLMo-core #929](https://github.com/allenai/OLMo-core/issues/929) is an open
contributor report with a local reproducer: changing batch size floors a consumed-token
cursor and repeats data. The inspected loader computes `tokens_processed //
global_batch_size`. [OLMo-core #908](https://github.com/allenai/OLMo-core/issues/908)
reports fresh Adam state advancing during step-0 serialization and links upstream
[PyTorch #164929](https://github.com/pytorch/pytorch/issues/164929). These are reported
software failures with valid serialization, not independently reproduced production
outages in this session. TrainDoctor's external component mixtures are controlled
artifacts, not evidence that non-atomic publication happened in a real job.

## One-page novelty boundary

**A. Established:** complete-state checkpointing, data/RNG resume, structural
validation, reference-based regression tests, runtime reruns and checkpoint-interval
replay all predate this work.

**B. Demonstrated locally:** one correct-source CPU/Gloo job, two externally assembled
component mixtures, twelve completed recoveries, explicit trusted capture/history
records and a separate NumPy continuation calculation. Full restore is optimal in
both states. No independent human validation exists.

**C. Candidate:** a narrowly scoped recovery audit that explains which component or
input-history invariant fails using trusted artifacts without an uninterrupted run.
This is an engineering application/reproduction until differentiation is demonstrated.

**D. Strongest objection:** Proof-of-Learning already implements the proposed interval
replay; frameworks already supply extensive checkpoint/reproducibility machinery.
A second handwritten MLP replay does not establish a new verification method.

**E. Adoption value:** potentially a small regression/debugging fixture for resume
bugs. It requires trustworthy anchors, reproducible inputs and a specified update
rule. The need for custom capture/data instrumentation may outweigh the benefit.
No measured real-team adoption value is established.

**F. Discriminating experiment:** remove access to the uninterrupted reference tree,
replay only original committed anchors and actual continuation records, compare all
twelve outcomes, and falsify the trusted-anchor assumption. A successful result
establishes local feasibility, not novelty. A later maintainer assessment would need
to identify a concrete unmet resume regression before integration is proposed.

**G. Decision:** NO-GO for a novelty/paper claim based on replay alone. GO for one
bounded reference-free validation and trust-boundary counterexample, reusing saved
artifacts. No new task, training run or infrastructure expansion is justified.
