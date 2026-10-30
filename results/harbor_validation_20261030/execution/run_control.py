import json,os,signal,subprocess,sys,time
from pathlib import Path
import psutil
name,agent=sys.argv[1:3]
root=Path('<LOCAL_HOME>/Workspace/rl_envs/.wt/agent-v2-research/results/harbor_validation_20261030')
out=root/name;out.mkdir(exist_ok=False)
cmd=['/private/tmp/traindoctor-harbor-venv-20261030/bin/harbor','run','-p','<LOCAL_HOME>/Workspace/rl_envs/.wt/harbor-v1-env/harbor/tasks/training-repair-01','-a',agent,'-e','docker','-n','1','-k','1','-r','0','-o',str(out/'jobs'),'--job-name',name,'--no-delete','--ek','keep_containers=true']
env=os.environ.copy()
for key in ('OPENAI_API_KEY','ANTHROPIC_API_KEY','GEMINI_API_KEY','GOOGLE_API_KEY','HF_TOKEN','HUGGING_FACE_HUB_TOKEN'):env.pop(key,None)
env.update(PYTHON_DOTENV_DISABLED='1',HARBOR_TELEMETRY='0',PYTHONPATH='<LOCAL_HOME>/Workspace/rl_envs/.wt/agent-v2-research',NO_COLOR='1')
start=time.monotonic();swap0=psutil.swap_memory().used
samples=[];stop=None
with (out/'console.txt').open('w') as log:
 p=subprocess.Popen(cmd,cwd='/private/tmp/td-harbor-cwd-20261030',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 while p.poll() is None:
  sample={'elapsed_s':time.monotonic()-start,'available_bytes':psutil.virtual_memory().available,'swap_used_bytes':psutil.swap_memory().used}
  samples.append(sample)
  if sample['available_bytes']<8*1024**3 or sample['swap_used_bytes']-swap0>128*1024**2 or sample['elapsed_s']>1200:
   stop=sample
   os.killpg(p.pid,signal.SIGINT)
   try:p.wait(timeout=30)
   except subprocess.TimeoutExpired:
    os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10)
   break
  time.sleep(1)
 p.wait()
result={'source':'executed','control':name,'scripted_control':True,'model_used':False,'command':cmd,'wall_s':time.monotonic()-start,'returncode':p.returncode,'guard_stop':stop,'swap_baseline_bytes':swap0,'minimum_available_bytes':min((x['available_bytes'] for x in samples),default=None),'maximum_swap_growth_bytes':max((x['swap_used_bytes']-swap0 for x in samples),default=None),'samples':samples,'harbor_source_commit':'509b119db0d97b141dcd9c80f4f7ab861c47c2d6','dotenv_disabled':True,'telemetry_disabled':True}
(out/'resources.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))
raise SystemExit(p.returncode or bool(stop))
