"""Export deidentified saved trajectories and recount model actions; no inference."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from experiments.local_agent_v1.audit_v2 import trajectory_metrics

CONDITIONS=('qwen_full','gemma_full','qwen_output1024')

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def clean(x):
    if isinstance(x,str):
        x=re.sub(r'/Users/[^/\s]+','<home>',x)
        x=re.sub(r'/private/var/folders/[^\s\"\']+','<temporary-path>',x)
        x=x.replace('/private/tmp/','<tmp>/').replace('/tmp/','<tmp>/')
        assert not re.search(r'/Users/|/home/|/var/folders/',x), 'unmapped private path'
        return x
    if isinstance(x,list):return [clean(v) for v in x]
    if isinstance(x,dict):return {clean(k):clean(v) for k,v in x.items()}
    return x

def metrics(root):
    frozen=json.loads((root/'frozen_config.json').read_text())
    diagnostic=json.loads((root/'diagnostic_config.json').read_text())
    task_path=Path(__file__).resolve().parents[1]/'experiments/masked_mean_v1/task.py'
    assert digest(task_path)==frozen['files']['experiments/masked_mean_v1/task.py'], 'task source differs from frozen trial'
    for filename,want in (('controller_512.py',frozen['files']['experiments/local_agent_v1/run.py']),
                          ('controller_1024.py',diagnostic['controller_sha256'])):
        assert digest(root/filename)==want, 'frozen controller bytes differ from trial'

    rows=[]
    for n in CONDITIONS:
        p=root/n/'trajectory.json';t=json.loads(p.read_text());m=trajectory_metrics(p)
        model='gemma' if n.startswith('gemma') else 'qwen'
        assert t['model_path']==frozen['models'][model]['path'] and t['revision']==frozen['models'][model]['revision']
        assert t['task_sha256']==frozen['files']['experiments/masked_mean_v1/task.py']
        assert t['controller_sha256']==(diagnostic['controller_sha256'] if n.endswith('1024') else frozen['files']['experiments/local_agent_v1/run.py'])
        assert t['generation_config']==dict(max_tokens=1024 if n.endswith('1024') else 512,temperature=0.2,seed=0)
        edits=[r for r in t['rows'] if r.get('tool') in ('apply_patch','edit_lines') and 'security_review' in r]
        assert all(r.get('patch_applicable') and r.get('observation')=='recorded' for r in edits)
        calls=[r for r in t['rows'] if r.get('protocol_valid') and r.get('tool')=='relaunch']
        assert len(calls)==t['relaunches']
        assert all('"execution_status": "failed"' in r['observation'] for r in calls)
        m['condition']=n;m['path']=f'{n}/trajectory.json'
        m['controller_sha256']=t['controller_sha256'];m['task_sha256']=t['task_sha256']
        m['valid_native_calls']=sum(bool(r.get('protocol_valid')) and r.get('parse_mode')=='native' for r in t['rows'])
        actual=t.get('last_execution') or t['result'].get('actual_continuation')
        m['actual_completed']=actual['completed'] if actual else None
        m['actual_semantic_valid']=actual['semantic_valid'] if actual else None
        m['failure_stage']='execution' if calls and actual and not actual['completed'] else m['failure_stage']
        m['feedback_followed_by_action']=any(r.get('tool')=='relaunch' and any(x.get('protocol_valid') for x in t['rows'][i+1:]) for i,r in enumerate(t['rows']))
        rows.append(m)
    return clean(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--check',type=Path);a=ap.parse_args()
    if a.check:
        assert metrics(a.check)==json.loads((a.check/'summary.json').read_text())
        manifest=json.loads((a.check/'manifest.json').read_text())
        assert all(digest(a.check/p)==v['exported_sha256'] for p,v in manifest['files'].items())
        print('PASS: corrected-controller identities, budgets, actions, rewards and derivative hashes')
        return
    if not a.raw or not a.out:ap.error('--raw and --out required')
    # First recount originals; originals are read only.
    rows=metrics(a.raw);a.out.mkdir(parents=True,exist_ok=False)
    manifest=dict(source='executed',analysis='historical receipt derivative; no new model execution',
       deidentification='home and temporary path strings replaced; no actions or outputs normalized',files={})
    paths=['frozen_config.json','diagnostic_config.json']+[f'{n}/trajectory.json' for n in CONDITIONS]
    for rel in paths:
        src=a.raw/rel;dst=a.out/rel;dst.parent.mkdir(exist_ok=True,parents=True)
        dst.write_text(json.dumps(clean(json.loads(src.read_text())),indent=2,sort_keys=True)+'\n')
        manifest['files'][rel]=dict(original_sha256=digest(src),exported_sha256=digest(dst))
    # Frozen controllers are source evidence, not executed by this script.
    for rel in ('controller_512.py','controller_1024.py'):
        src=a.raw/rel;dst=a.out/rel
        assert '/Users/' not in src.read_text()
        dst.write_bytes(src.read_bytes())
        assert digest(src)==(rows[2] if '1024' in rel else rows[0])['controller_sha256']
        manifest['files'][rel]=dict(original_sha256=digest(src),exported_sha256=digest(dst))
    (a.out/'summary.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
    assert metrics(a.out)==rows
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps([{k:r[k] for k in ('condition','response_turns','valid_native_calls','applied_edits','execution_attempts','terminal_reward','failure_stage')} for r in rows],indent=2))

if __name__=='__main__':main()
