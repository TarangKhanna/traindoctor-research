# Local milestone status

Testing clarification: the deterministic controller wiring regression now covers
protocol recovery through approval/resume, relaunch feedback and finish using
explicit test doubles. It is not a live model repair. [Test coverage and
remaining gaps](TESTING.md) separates this check from numerical replay and fresh
distributed execution. A native critic reviewed the final scope; no new model
episode or training run was performed.

Current hiring release: a working multi-turn model harness and executable
masked-loss repair task are implemented. The [public report and portable
packet](../evidence/masked_mean_v1/REPORT.md) recount captured states and model
actions without a model or workers. Model-selected investigation is demonstrated;
successful autonomous source repair/relaunch/feedback is not. The frozen episodes
remain unchanged, and the later protocol fixes were not model-rerun. The earlier
Harbor preflight status in AGENT_V2_REPORT is now explicitly marked superseded by
the executed validation. Release preparation starts no new inference or training.

Current: owner-authorized `demo/masked-mean-v1` follow-up is bounded and complete.
Read [the task contract and reproduction commands](MASKED_MEAN_V1.md) and the
script-generated `results/masked_mean_v1_final_20261030/REPORT.md`. The new task's
controls and scripted tool smoke pass. Neither actual local model applied a patch
or requested relaunch. The separately executed Qwen text fragment fails Python;
it receives no autonomous credit. No reduced arm, model retry or training followed.
Historical milestones below remain unchanged.

2026-10-29: one state-only recovery pair implemented on isolated `demo/state-recovery-v1` branches. All twelve continuations complete and agree with an independent NumPy continuation oracle. Full restore is optimal in both states; scratch is correct but repeats more work. Historical verifier v1.2 and evidence remain unchanged.

Cached Qwen 4B and Gemma E2B were profiled locally with no swap growth. A parity-checked Gemma prefix-cache probe improved the five-request total by 1.47x including setup. No new training was run. See [the report](../experiments/state_recovery_v1/REPORT.md) for exact evidence, model-control limitations and reproduction commands.

Verdict: engineering demonstration GO; conditional-selection RL benchmark NO-GO. Stop after this worked example. Independent reproduction remains outstanding. No cloud compute, dependency changes, paid actions, pushes, merges or publication occurred.

Frozen source commits: environment d568f69, research 6e158ce. Fresh local clones pass 36 focused tests, archive checksum and README checks. A fresh sandboxed execution and independent recount again pass all twelve continuations. The prepared local dependency environment was reused; external-machine reproduction remains outstanding.

Final transport hardening: environment 65e4bea fixes a live-observed wildcard TCPStore bind with a launcher-owned loopback store. Live lsof regression and 241 focused tests pass; all twelve continuations and their recount pass again with unchanged observations and outcomes. Final evidence is results/state_recovery_v1_loopback_verified; earlier packets are preserved.

Local follow-up: [scoped agent instructions](../AGENTS.md) and [append-only learning notes](STATE_RECOVERY_LEARNINGS.md) capture the scientific corrections, discarded unsafe approach and measured performance limits. They require reuse before new infrastructure and preserve the current stop decision. This is a documentation-only update.

2026-10-29 reference-free follow-up, isolated `audit/reference-free-v1`: primary-source gate finds direct interval-replay prior art in Proof-of-Learning (2021). A bounded checker audits the existing twelve continuations without fault-free/reference/verdict inputs; eight accept and four reject, agreeing with historical labels. Two new CPU/Gloo continuations are bitwise repeatable. A coherent poisoned-anchor chain demonstrates why the claim must remain conditional. GO for local engineering validation; NO-GO for replay novelty, paper/release now or RL. See [the report](REFERENCE_FREE_REPORT.md). No environment code, v1.2 or historical receipts changed; no experimental model training/inference or external publication/outreach occurred.

2026-10-29 owner-authorized release preparation supersedes the earlier publication hold for the bounded engineering artifact. README now leads with the completed demo, conditional reference-free audit and saved-evidence reproduction. Historical numerical results/pins remain unchanged. The incomplete local-agent milestone stays on its separate local branch and is excluded from this release; no model-agent success is claimed. Editable sibling-source installation, saved recount, README checks and focused CPU tests pass.

2026-10-29 focused README compression: concise recovery-first landing page; prior full content retained in docs/PROJECT_DETAIL.md. Relative links, code fences, numerical provenance and diff checks pass; read-only critic found no blocker. No new experimental claim.

2026-10-29 owner-authorized local-agent milestone, isolated `demo/local-agent-v1`: one checkpoint-code migration incident and positive/negative/scratch/skip-work controls execute correctly. Cached Gemma made three real native inspection calls, then hit the RAM/swap guard before proposing a patch. It is unscored infrastructure failure, not model repair failure. Memory remained below the 8 GiB floor with our model stopped, so Qwen and the reduced-evidence conditions remain unrun. Actual model-generated executable repair is **not yet demonstrated**. See [the bounded report](LOCAL_AGENT_V1_REPORT.md); no training or multi-agent experiment was started.

2026-10-29 Qwen follow-up: one full-evidence invocation stopped by the pre-load resource guard after a previously adequate host preflight. Zero loads/tokens/calls; unscored infrastructure error. No retry or process cleanup. Receipt preserved in results/local_agent_v1_qwen_full_20261029; 10 focused harness tests pass. Model repair remains unestablished.

2026-10-29 post-wake Qwen attempt: model loaded and made seven schema-valid inspection calls; a repeated-escape malformed patch hit the output cap. Next generation hit the resource guard. No patch/recovery executed; terminal reward null. Full raw transcript/summary saved under results/local_agent_v1_qwen_full_20261029_attempt2. Local training is owner-authorized but not started because task/evaluation and stable execution gates remain unresolved. No conditions retried after this interruption.

2026-10-29 owner-requested Qwen retry: controller 895f0df adds failing-sample telemetry only; 12 focused tests pass. Seven schema-valid inspections, two malformed capped patch blocks, no applied patch/recovery. All eight comparable responses match the previous greedy episode. RAM-floor guard fired at 7.7312469482421875 GiB available while swap decreased 40 MiB. Unscored infrastructure stop; no training or additional full episode. Receipts: results/local_agent_v1_qwen_full_20261029_attempt3.

2026-10-30 demo/agent-v2 feasibility: publicheads c5a8a9b/4eea735; source6e0045a. Native Gemma completed12turns with no applicable edit; sampledQwen applied a model-generated line edit at turn12 but had no relaunch budget. Original proposal executed in sandbox and failedPython; editor-boundary-normalized diagnostic control completed but independentNumPy state check rejected it (maxabs0.24375582367959417, hashes/historyvalid). A fresh16turn retest resource-interrupted; nullreward. Fixed-source baseline passes current/legacy and fullrestore solves bothstatecases. No G4/G8 groups sampled. STOP training/benchmark expansion oncurrentcases; autonomous model-requested execution remains unproven. 39research+18Harborunit tests pass. See AGENT_V2_REPORT.md. Harbor static task committed envcc69e14; operational controls pending isolatedsetup approval. No pushes/training/cloud/newmodels.

2026-10-30 Harbor runtime validation supersedes the pending setup above: owner-approved isolated CLI and Docker Desktop execute the existing single B_DUP smoke task. Final controls oracle1/no-op0/completing-invalid0/tamper0 all complete, validate checkpoint hashes and agree with native source/v1.2 verdict controls. The incorrect optimizer repair completes with correct sample history but fails semantic checks. Real infrastructure failure emits no reward; forgery and stale-text-reward regressions are checked. 20 adapter tests pass; default lint has one narrowly reviewed root-capability finding. Exact costs, image identities, runtime fixes and isolation limits are in HARBOR_VALIDATION.md and results/harbor_validation_20261030. Own trial resources cleaned; no other research terminated. GO local interoperability smoke only; existing STOP on learning/benchmark expansion remains. No models/training/cloud/push/publication occurred.

2026-10-30 owner-authorized public release: retain the concise published README while integrating agent feasibility receipts and executed Harbor controls. Focused release checks caught an undeclared optional psutil import on offline/CI paths; imports now work without the profiler, while live inference fails closed without its memory guard. No dependency or historical evidence changed. Reviewed, tested source and concise README updates are approved for normal fast-forward pushes to both repositories.

Release verification: 20 Harbor adapter tests, 22 agent tests (also against the historical CI pin without psutil), README numerical/link checks and the four-control saved-evidence recount pass. Read-only subagent review found no remaining blocker; outgoing history/archive credential-pattern checks were clear within their stated limits.
