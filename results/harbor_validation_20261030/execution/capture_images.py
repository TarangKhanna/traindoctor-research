import json,subprocess
from pathlib import Path
root=Path('<LOCAL_HOME>/Workspace/rl_envs/.wt/agent-v2-research/results/harbor_validation_20261030')
all_ids=subprocess.check_output(['docker','ps','-aq'],text=True).split()
inspect=json.loads(subprocess.check_output(['docker','inspect',*all_ids],text=True)) if all_ids else []
for control in root.iterdir():
 if not control.is_dir() or not (control/'jobs').exists():continue
 names=[p.name.lower() for p in (control/'jobs').glob('*/*') if p.is_dir() and p.name.startswith('training-repair-01__')]
 selected=[]
 for x in inspect:
  if not any(x['Name'].lstrip('/').startswith(n+'__') for n in names):continue
  h=x['HostConfig'];c=x['Config'];im=json.loads(subprocess.check_output(['docker','image','inspect',x['Image']],text=True))[0]
  selected.append({'container_id':x['Id'],'name':x['Name'],'image_id':x['Image'],
   'image':{k:im.get(k) for k in ('Id','RepoDigests','RepoTags','Architecture','Os','Created','Descriptor')},
   'runtime':{'user':c.get('User'),'entrypoint':c.get('Entrypoint'),'cmd':c.get('Cmd'),
    **{k:h.get(k) for k in ('ReadonlyRootfs','NetworkMode','CapAdd','CapDrop','SecurityOpt','PidsLimit','Memory','NanoCpus','PidMode','Privileged')}},
   'state':{k:x['State'].get(k) for k in ('ExitCode','OOMKilled','Status')},
   'mounts':[{k:m.get(k) for k in ('Type','Name','Destination','RW')} for m in x['Mounts']]})
 if selected:(control/'images.json').write_text(json.dumps({'source':'executed','containers':selected},indent=2)+'\n')
 print(control.name,'inspected',len(selected))
