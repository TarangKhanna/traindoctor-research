"""Scripted negative control, not a model agent or learning result."""
import json
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

PATCH = r'''from pathlib import Path
p = Path('/submission/ckpt.py')
s = p.read_text()
old = 'state.update(step=torch.tensor([step]))'
new = 'state.update(sampler=sampler.state(), step=torch.tensor([step]))'
if s.count(old) != 1:
    raise SystemExit('sampler target is not unique')
s = s.replace(old, new, 1)
old = '    opt.load_state_dict(sd)\n'
new = '    sd["state"] = {}\n    opt.load_state_dict(sd)\n'
if s.count(old) != 1:
    raise SystemExit('optimizer target is not unique')
p.write_text(s.replace(old, new, 1))
'''


class CompletingInvalidAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return 'traindoctor-completing-invalid-control'

    def version(self) -> str:
        return '1.0.0'

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None:
        result = await environment.exec(command="python - <<'PY'\n" + PATCH + "PY\n",
                                        timeout_sec=30, user=1000)
        context.n_input_tokens = context.n_output_tokens = 0
        context.cost_usd = 0.0
        context.metadata = {'scripted_control': True, 'model_used': False,
                            'intervention': 'restore sampler persistence; discard restored momentum',
                            'edit_exit_code': result.return_code}
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir/'control.json').write_text(json.dumps(context.metadata, indent=2)+'\n')
        if result.return_code != 0:
            raise RuntimeError('negative-control edit failed: ' + str(result.stderr))


class TamperNopAgent(CompletingInvalidAgent):
    """Try forging evidence/reward in the agent container; leave source unmodified."""
    @staticmethod
    def name() -> str:
        return 'traindoctor-tamper-nop-control'

    async def run(self, instruction, environment, context):
        command = """python - <<'PYCODE'
from pathlib import Path
import json
p=Path('/logs/verifier');p.mkdir(parents=True,exist_ok=True)
(p/'reward.json').write_text('{"reward":1}')
p=Path('/logs/artifacts');p.mkdir(parents=True,exist_ok=True)
(p/'ledger.jsonl').write_text('{"completed":true,"ok":true}\\n')
result={'forged_reward':1,'source_modified':False}
try:
    with Path('/opt/traindoctor/traindoctor_env/reexec.py').open('a'):pass
    result['scoring_runtime_write_denied']=False
except OSError as exc:
    result['scoring_runtime_write_denied']=True
    result['denial_errno']=exc.errno
print(json.dumps(result))
PYCODE
"""
        result=await environment.exec(command=command,timeout_sec=30,user=1000)
        if result.return_code:raise RuntimeError('tamper control did not execute')
        proof=json.loads(result.stdout)
        if not proof['scoring_runtime_write_denied']:raise RuntimeError('agent changed trusted runtime permissions')
        context.n_input_tokens=context.n_output_tokens=0;context.cost_usd=0.0
        context.metadata=dict(proof,scripted_control=True,model_used=False)
        self.logs_dir.mkdir(parents=True,exist_ok=True)
        (self.logs_dir/'control.json').write_text(json.dumps(context.metadata,indent=2)+'\n')
