# Mutation pilot (procedural task generation with trusted-replay validation)

Eight TrainDoctor tasks made by seeded patch templates (four operators, two tasks each) over the known-good Gloo training job:
O1 collective type/shape divergence on one rank, O2 checkpoint sampler state dropped or saved a batch behind,
O3 non-atomic checkpoint-manifest write with the writer killed mid-write, O4 off-by-one in the resume step.
Each task has a broken job, a gold fix, an alternative fix where the verifier admits one, and a decoy repair that lets the job
complete but fails the recovery invariants. No language model writes any mutant.

Like SWE-smith, tasks come from procedural mutations of known-good programs and are kept only if execution shows the broken version fails and the gold fix passes. Unlike SWE-smith, the check is a re-executed multi-rank training run graded on recovery invariants, not a unit-test suite, and every task includes a decoy repair that completes the job but fails those invariants.

## Result (honest)
Verdict: NO-GO (see results/FINAL_VERDICT.md). All 8 tasks passed validation, but Codex medium verified 24/24 attempts, so the tasks
do not discriminate. Supported claim: 8 procedurally mutated tasks passed trusted re-execution validation. Nothing about ranking,
generalisation or RL.

## Layout
- generator/mutate.py: operators, gold/alt/decoy patches. pilot/: snapshot execution, fingerprints, validation, freeze, agent protocol and scoring, vendored patch parser.
- tasks.json (frozen definitions + freeze hash), results/ (evaluation, table, validation summary, verdict). The per-task validation receipts and the agent prompt were archived privately and are not published.
- tests/: 31 offline tests (determinism, gold restores pristine bytes, arms valid and distinct, seed range). Seeds used: 40001-40008.

## Reproduce
The generator and pilot read `jobs/` and `traindoctor_env/` from a checkout of the library. `TD_REPO` is the path to a checkout at the pinned commit (the one in the top-level `requirements.txt`, 0b0a36b).
Offline tests, from a fresh clone (no Modal, no API calls; 31 passed in 0.3 s on Python 3.12):
```
git clone https://github.com/TarangKhanna/traindoctor-research && git clone https://github.com/TarangKhanna/traindoctor-env
(cd traindoctor-env && git checkout 0b0a36b6b2a34a620e959c665a59f8a92b7413c2)
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r traindoctor-research/requirements.txt
export TD_REPO=$PWD/traindoctor-env
cd traindoctor-research/experiments/mutation_pilot && python -m pytest -q
```
Validation replays additionally need a Modal account (client importable, or `TD_MODAL_SITE` pointing at its site-packages) and cost about $0.15 of CPU:
```
export TD_KEY=$(python -c "import secrets;print(secrets.token_hex(16))")   # env var only, never stored
python -m pilot.validate run                # 8 tasks x noop/gold/decoy/alt trusted replays
python -m pilot.validate summary            # table + fingerprint dedup
```
Agent lane: `python -m pilot.agent build`, then give an agent PROMPT.md plus a workspace under `$TD_WS_ROOT`, save its final message as
`out/answers/<agent>/<task>_k<n>.json`, `python -m pilot.agent score-all <agent>`, `table`, `evaluate`.
Modal spend is recorded to the canonical `modal_train` ledger with run ids prefixed `mutpilot-`; the cap used was $2.20.
