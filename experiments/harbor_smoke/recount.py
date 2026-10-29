"""Recount one local Harbor smoke task. Scripted controls are not model rollouts."""
from __future__ import annotations
import argparse
import hashlib
import json
import tarfile
from datetime import datetime
from pathlib import Path


def read(path):return json.loads(path.read_text())

def phase_seconds(phase):
    if not phase or not phase.get('started_at') or not phase.get('finished_at'):return None
    return (datetime.fromisoformat(phase['finished_at'])-datetime.fromisoformat(phase['started_at'])).total_seconds()


def recount(root, series="final"):
    native={x['case']:x for x in read(root/'native/summary.json')['cases']}
    rows=[];checked=0;checksums=set()
    oracle = 'oracle-hardened-proof' if series == 'hardened' else 'oracle-final'
    for control,case,reward in [(oracle,'oracle',1),(f'nop-{series}','nop',0),
                                (f'invalid-{series}','invalid',0),(f'tamper-{series}','nop',0)]:
        base=root/control
        trials=list((base/'jobs'/control).glob('training*/result.json'))
        if len(trials)!=1:raise ValueError('exactly one trial required')
        path=trials[0].parent;t=read(trials[0]);g=read(path/'verifier/grade_result.json')
        checksums.add(t['task_checksum'])
        resources=read(base/'resources.json');images=read(base/'images.json')['containers']
        evidence=path/'verifier/execution_evidence';replay=read(evidence/'replay.json')
        assert t['exception_info'] is None and t['verifier_environment_mode']=='separate'
        assert t['verifier_result']['rewards']['reward']==g['terminal_reward']==reward
        assert g['status']=='graded' and g['execution_performed']
        v=g['verdict'];p=v['worker_isolation']
        assert v['format_check']=='v1.2' and v['sandboxed'] and v['exec_mode']=='harbor_docker_uid'
        assert p['uids']==p['gids']==[65534]*4 and p['groups']==[] and p['no_new_privs']=='1'
        assert all(int(x,16)==0 for x in p['caps'].values())
        assert all(not(int(x,16)&(1<<12)) for x in p['root_capabilities'].values())
        assert p['private_read_denied'] and p['private_write_denied'] and p['launcher_can_kill_worker']
        assert {x['name'] for x in p['network_interfaces'] if x['up']}=={'lo'}
        assert p['external_connect_errno'] in (1,13,101,113)
        assert len(images)==2 and all(x['runtime']['NetworkMode']=='none' and x['runtime']['ReadonlyRootfs']
            and not x['runtime']['Privileged'] for x in images)
        assert all(m['Type']=='volume' for x in images for m in x['mounts'])
        assert all(not x['state']['OOMKilled'] for x in images)
        mounts=[{m['Name'] for m in x['mounts'] if m['Type']=='volume'} for x in images]
        assert not mounts[0]&mounts[1]
        source=(path/'artifacts/ckpt.py').read_bytes()
        assert hashlib.sha256(source).hexdigest()==g['submitted_source_sha256']==native[case]['source_sha256']
        shared=set(native[case]['outcome'])-{'exec_mode'}
        assert all(native[case]['outcome'][k]==v[k] for k in shared), 'cross-backend verdict mismatch'
        ledger=[json.loads(x) for x in (evidence/'protocol.trusted/ledger.jsonl').read_text().splitlines() if x]
        assert ledger[-1]['completed']
        for record in ledger:
            for cap in record['captures']:
                rel=Path(cap['file'])
                if rel.is_absolute() or '..' in rel.parts:raise ValueError('invalid capture path')
                f=evidence/'protocol.trusted'/rel
                assert f.resolve().is_relative_to((evidence/'protocol.trusted').resolve()) and not f.is_symlink()
                assert hashlib.sha256(f.read_bytes()).hexdigest()==cap['sha256'];checked+=1
        assert resources['guard_stop'] is None and resources['maximum_swap_growth_bytes']<=128*1024**2
        assert resources['minimum_available_bytes']>=8*1024**3
        if control==f'tamper-{series}':
            assert t['agent_result']['metadata']['forged_reward']==1
            assert t['agent_result']['metadata']['scoring_runtime_write_denied']
        rows.append(dict(control=control,expected_reward=reward,reward=reward,completed=True,
            source_sha256=g['submitted_source_sha256'],native_verdict_agrees=True,ckpt_valid=v['ckpt_valid'],
            sample_gaps=v['sample_gaps'],sample_dups=v['sample_dups'],transitions_consistent=v['transitions_consistent'],
            wall_s=resources['wall_s'],environment_setup_s=phase_seconds(t['environment_setup']),
            agent_setup_s=phase_seconds(t['agent_setup']),agent_execution_s=phase_seconds(t['agent_execution']),
            verifier_s=phase_seconds(t['verifier']),**v['timings'],
            container_peak_memory_bytes=replay['cgroup_memory_bytes']['memory.peak'],
            container_memory_limit_bytes=replay['cgroup_memory_bytes']['memory.max'],
            host_minimum_available_bytes=resources['minimum_available_bytes'],
            maximum_swap_growth_bytes=resources['maximum_swap_growth_bytes'],
            images=[x['image_id'] for x in images]))
    assert len(checksums)==1, 'controls used different task bytes'
    if series=='hardened':
        proof=read(root/'source-proof-hardened.json')
        with tarfile.open(root/'environment-source.tar.gz','r:gz') as archive:
            for record in proof['files']:
                assert record['matches']
                f=archive.extractfile(record['repository_path']);assert f is not None
                assert hashlib.sha256(f.read()).hexdigest()==record['sha256']
    infra=list((root/'oracle-volume-fix/jobs/oracle-volume-fix').glob('training*/verifier/grade_result.json'))
    assert len(infra)==1
    i=read(infra[0]);assert i['status']=='infrastructure_error' and i['terminal_reward'] is None
    assert not (infra[0].parent/'reward.json').exists() and not (infra[0].parent/'reward.txt').exists()
    result=dict(source='executed',protocol='one-task-local-harbor-v1',model_used=False,
        controls=rows,checked_capture_hashes=checked,task_checksum=next(iter(checksums)),infra_no_reward=True,
        isolation_checks_pass=True,verdict='GO: local interoperability smoke test only',
        limitations=['same v1.2 verifier across backends; not independent mathematical validation',
            'one authored smoke task; no benchmark or model-learning claim',
            'one local run/control with warm build cache, not a performance comparison',
            'default lint has one reviewed capability finding; narrow exception recorded'])
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--series',choices=('final','hardened'),default='hardened')
    a=p.parse_args()
    if a.out.exists():p.error('require fresh output file')
    result=recount(a.run,a.series);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: four real Harbor controls, native verdict/source agreement, isolation, infrastructure-null and',result['checked_capture_hashes'],'capture hashes')


if __name__=='__main__':main()
