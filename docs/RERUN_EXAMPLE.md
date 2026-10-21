# Rerunning one failure, a repair and a fake repair (CPU only)

One fixed-seed example from the library, [traindoctor-env](https://github.com/TarangKhanna/traindoctor-env) at commit 0b0a36b, the library commit this example was recorded against (`requirements.txt` now pins a later commit). No GPU, API key or Modal account is needed. It uses the library's own `scripts/quickstart.py`; nothing in this repository wraps it.

```bash
git clone https://github.com/TarangKhanna/traindoctor-env && cd traindoctor-env
git checkout 0b0a36b6b2a34a620e959c665a59f8a92b7413c2
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements-cpu.txt
python scripts/quickstart.py --fake F11 --out quickstart_out
```

Seed 123 builds an A1 job (rank 2 calls `broadcast` while ranks 0, 1 and 3 call `all_reduce`) that fails with a collective timeout. The script then re-runs the job with the reference repair (a source diff) and with a fake repair (`world_size: 1`, F11). Output of a run on macOS, Python 3.12, torch 2.14.1 (about 30 s):

```text
INVALID verification: re-execution ran without OS isolation; ok forced False
INVALID verification: re-execution ran without OS isolation; ok forced False
== 1. failed job from seed 123: kind=A1 variant=0 world_size=4
   124 public records (source=executed). What the agent sees:
   E004893 job       [run0] resume_from=none started_at_step=0 -> failed last_step=5 error=COLLECTIVE_TIMEOUT
   Edde9ac process   [run0] rank 0 failed exit=1 last_step=5 COLLECTIVE_TIMEOUT
   Ef147be process   [run0] rank 1 failed exit=1 last_step=5 COLLECTIVE_TIMEOUT
   Ec625f1 process   [run0] rank 2 failed exit=1 last_step=5 COLLECTIVE_TIMEOUT
   Ebd3f6e process   [run0] rank 3 failed exit=1 last_step=5 COLLECTIVE_TIMEOUT
   E5d239e scheduler submitted: run0 resume_from=none
   E666c1f scheduler failed: run0 COLLECTIVE_TIMEOUT
   E3b6981 trace     [run0] rank 0 step=6 seq=12 all_reduce bucket=0 numel=80 pending
   Ef4a4f6 trace     [run0] rank 1 step=6 seq=12 all_reduce bucket=0 numel=80 pending
   E077217 trace     [run0] rank 2 step=6 seq=12 broadcast bucket=0 numel=80 pending
   E5d9a14 trace     [run0] rank 3 step=6 seq=12 all_reduce bucket=0 numel=80 pending
   hidden ground truth (never shown to an agent): label=COLLECTIVE_DIVERGENCE {'bad_op': 'broadcast', 'rank': 2, 'step': 6}

== 2. gold repair (oracle: label, cited evidence ids, source edit)
   diagnosis=COLLECTIVE_DIVERGENCE evidence=['E077217', 'E3b6981']
--- a/job/train.py
+++ b/job/train.py
@@ -77,10 +77,7 @@
             comm.broadcast(bucket, step, b)
             bucket.div_(comm.world)
         else:
-            if rank == 2 and step >= 6 and b == 0:
-                comm.broadcast(bucket, step, b)
-            else:
-                comm.all_reduce(bucket, step, b)
+            comm.all_reduce(bucket, step, b)
 
 
 def main() -> None:

== 3. re-run the repaired job LOCALLY (UNVERIFIED: not sandboxed)
  [gold, local] exec_mode=local_unsandboxed sandboxed=False ok=False steps_completed=12 lost_steps=2 sample_gaps=0 sample_dups=0
   invariants: params_match_reference=True, samples_match_reference=True, restores_consistent=True, no_duplicate_updates=True, ranks_consistent=True, ckpt_valid=True, states_match_reference=True
   detail: INVALID: unsandboxed re-execution
   -> the frozen verifier refuses to count this: re-executing agent-edited code without OS/gVisor isolation is not
      verification, so ok is forced False (INVALID) although all invariants above hold. Use --modal for the verified path.

   fake repair F11: world_size:1 collapse
  [F11, local] exec_mode=local_unsandboxed sandboxed=False ok=False steps_completed=12 lost_steps=2 sample_gaps=96 sample_dups=0
   invariants: params_match_reference=False, samples_match_reference=False, restores_consistent=True, no_duplicate_updates=True, ranks_consistent=True, ckpt_valid=True, states_match_reference=False
   detail: INVALID: unsandboxed re-execution; I6 world size differs from the reference
```

How to read it:

- Failure: all four ranks time out; the trace records (E077217 versus E3b6981) show rank 2 in a different collective.
- Legitimate repair: the diff restores a common `all_reduce`. All invariants hold (parameters, samples, restores, ranks, checkpoints, state).
- Fake repair: the hang goes away and the job finishes 12 steps, but the world size differs from the reference, 96 samples are missing and the parameters differ.
- Verifier result: this local run is labelled unsandboxed, so the verifier forces `ok=False` for both cases by design (`INVALID: unsandboxed re-execution`). The accepted/rejected verdict (gold accepted, F11 rejected) needs the Modal Sandbox path, which I did not run for this page; a recorded result for the A2 task is in [evidence/results/quickstart_verified_replay.txt](../evidence/results/quickstart_verified_replay.txt). The invariant lines above are what differ between the two cases.
- Sample counts depend on the task instance: the 96 missing samples here are for seed 123, while the red-team table reports 72 missing and 8 duplicated for F11 on a different instance. They are not comparable.
