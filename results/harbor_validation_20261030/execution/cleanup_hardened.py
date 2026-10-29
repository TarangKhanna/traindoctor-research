import json,subprocess
from pathlib import Path
root=Path('/Users/tarangkhanna/Workspace/rl_envs/.wt/agent-v2-research/results/harbor_validation_20261030')
ids=set();volumes=set()
for control in ('oracle-hardened-proof','nop-hardened','invalid-hardened','tamper-hardened'):
 for x in json.loads((root/control/'images.json').read_text())['containers']:
  ids.add(x['container_id']);volumes.update(m['Name'] for m in x['mounts'] if m['Type']=='volume')
removed=[]
for cid in sorted(ids):
 r=subprocess.run(['docker','inspect',cid],text=True,capture_output=True,check=True)
 x=json.loads(r.stdout)[0];assert x['Name'].lstrip('/').startswith('training-repair-01__') and not x['State']['Running']
 subprocess.run(['docker','rm',cid],check=True,capture_output=True);removed.append(cid)
for name in sorted(volumes):
 assert name.startswith('training-repair-01__')
 subprocess.run(['docker','volume','rm',name],check=True,capture_output=True)
(root/'cleanup-hardened.json').write_text(json.dumps({'source':'executed','removed_container_ids':removed,'removed_volume_names':sorted(volumes),'images_retained':True,'docker_desktop_left_running':True},indent=2)+'\n')
print('PASS: removed',len(removed),'own stopped containers and',len(volumes),'own volumes; images/Desktop retained')
