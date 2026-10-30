import hashlib,json,subprocess,tarfile
from pathlib import Path
root=Path('<LOCAL_HOME>/Workspace/rl_envs/.wt/agent-v2-research/results/harbor_validation_20261030')
repo=Path('<LOCAL_HOME>/Workspace/rl_envs/.wt/harbor-v1-env')
images=json.loads((root/'oracle-hardened-proof/images.json').read_text())['containers']
v=next(x for x in images if '__verifier__' in x['name'])
sources=[]
for src,rel in [('/tests/worker_bridge.py','harbor/tasks/training-repair-01/tests/worker_bridge.py'),('/tests/grade.py','harbor/tasks/training-repair-01/tests/grade.py'),('/opt/traindoctor/traindoctor_env/reexec.py','traindoctor_env/reexec.py')]:
 dest=Path('/private/tmp/td-harbor-cwd-20261030')/('sealed-'+Path(src).name)
 subprocess.run(['docker','cp',v['container_id']+':'+src,str(dest)],check=True,capture_output=True)
 actual=hashlib.sha256(dest.read_bytes()).hexdigest();assert actual==hashlib.sha256((repo/rel).read_bytes()).hexdigest()
 sources.append({'container_path':src,'repository_path':rel,'sha256':actual,'matches':True})
(root/'source-proof-hardened.json').write_text(json.dumps({'source':'executed','image':v['image_id'],'files':sources},indent=2)+'\n')
paths=subprocess.check_output(['git','-C',str(repo),'ls-files','jobs','traindoctor_env','harbor','pyproject.toml','uv.lock','tests/test_harbor_grader.py'],text=True).splitlines()
records=[]
with tarfile.open(root/'environment-source.tar.gz','w:gz') as archive:
 for rel in paths:
  p=Path(rel)
  assert not p.is_absolute() and '..' not in p.parts and not any(x in ('secrets','.git','__pycache__') or x=='.env' or x.startswith('.env.') for x in p.parts)
  f=repo/p;assert f.is_file() and not f.is_symlink()
  data=f.read_bytes();records.append({'path':rel,'sha256':hashlib.sha256(data).hexdigest()})
  archive.add(f,arcname=rel,recursive=False)
(root/'environment-files.json').write_text(json.dumps({'source':'executed','files':records},indent=2)+'\n')
(root/'environment-source.patch').write_bytes(subprocess.check_output(['git','-C',str(repo),'diff','HEAD','--','harbor','tests/test_harbor_grader.py']))
print('PASS: executed source bytes match; archived',len(records),'allowlisted source files')
