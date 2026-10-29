"""Bounded two-incident feasibility recount; no model or training framework."""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path


def group_information(groups, size):
    """Only explicitly assigned same-task groups count; infrastructure is unscored."""
    if any(len(g) != size for g in groups):
        raise ValueError('incorrect prespecified group size')
    scored = [g for g in groups if all(r is not None for r in g)]
    if any(r not in (0, 1) for g in scored for r in g):
        raise ValueError('terminal semantic rewards must be binary or null')
    mixed = sum(min(g) != max(g) for g in scored)
    return dict(groups=len(groups), infrastructure_groups=len(groups)-len(scored),
                scored_groups=len(scored), mixed_groups=mixed,
                mixed_fraction=mixed/len(scored) if scored else None,
                zero_advantage_fraction=1-mixed/len(scored) if scored else None,
                reward_std=[statistics.pstdev(g) for g in scored])


def trajectory_metrics(path):
    t = json.loads(path.read_text())
    if t.get('result') is None:
        raise ValueError('cannot finalize a running trajectory')
    rows, result = t['rows'], t['result']
    edits = [r for r in rows if r.get('tool') in ('apply_patch', 'edit_lines')]
    actual = result.get('actual_continuation')
    if result.get('terminal_reward') is None:
        failure = 'infrastructure'
    elif result.get('terminal_reward') == 1:
        failure = None
    elif not any('security_review' in r for r in edits):
        failure = 'patch_applicability' if edits else 'patch_generation'
    elif actual is None:
        failure = 'no_execution'
    elif not actual.get('completed'):
        failure = 'execution'
    elif not actual.get('semantic_valid'):
        failure = 'semantic_corruption'
    else:
        failure = 'hidden_regression'
    partial = t.get('partial_generation', {})
    return dict(path=str(path), provenance='live_model', model_revision=t['revision'],
        configuration=t.get('generation_config', {'temperature':0, 'seed':0, 'max_tokens':512}),
        status=t['status'], terminal_reward=result.get('terminal_reward'), failure_stage=failure,
        response_turns=len(rows), protocol_valid=sum(bool(r.get('protocol_valid')) for r in rows),
        protocol_errors=sum(bool(r.get('protocol_error')) for r in rows),
        edit_attempts=len(edits), applicable_edits=sum(bool(r.get('patch_applicable')) for r in edits),
        applied_edits=sum('security_review' in r for r in edits),
        execution_attempts=t.get('relaunches', 0), actual_completed=actual.get('completed') if actual else None,
        actual_semantic_valid=actual.get('semantic_valid') if actual else None,
        diagnosis_recorded=bool(t.get('episode', {}).get('diagnosis')), diagnosis_accuracy=None,
        generated_tokens=sum(r['generation_tokens'] for r in rows),
        partial_generated_tokens=len(partial.get('token_ids', [])),
        load_s=sum(r['wall_s'] for r in t.get('loads', [])),
        inference_s=sum(r['wall_s'] for r in rows), partial_inference_s=partial.get('wall_s', 0),
        prefill_s=sum(r.get('prefill_s') or 0 for r in rows),
        decode_s=sum(r.get('decode_s') or 0 for r in rows),
        tool_s=sum(r.get('tool_wall_s', 0)+r.get('approval_apply_wall_s', 0) for r in rows),
        terminal_verification_s=t.get('terminal_verification_wall_s', 0),
        measured_active_s=t.get('measured_active_s'), elapsed_including_review_s=t.get('elapsed_including_review_s'),
        peak_mlx_gb=max((r['peak_mlx_gb'] for r in rows), default=None),
        minimum_completed_turn_available_bytes=min((r['available_bytes'] for r in rows), default=None),
        resource_guard_failure=t.get('resource_guard_failure'),
        group_id=t.get('rollout_group_id'),
        warning='Single trajectories are not rates. No diagnosis taxonomy or independent human validation.')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--trajectory', type=Path, action='append', default=[])
    ap.add_argument('--fixture', type=Path)
    ap.add_argument('--replay-controls', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    a=ap.parse_args()
    if a.out.exists(): ap.error('require a fresh result directory')
    root=Path(__file__).resolve().parents[2]
    a.out.mkdir(parents=True)
    state_path=root/'results/state_recovery_v1_loopback_verified/recount.json'
    control_path=root/'results/local_agent_v1_fixture_20261029/controls.json'
    state=json.loads(state_path.read_text()); controls=json.loads(control_path.read_text())
    metrics=[trajectory_metrics(p) for p in a.trajectory]
    full=[r for r in state['rows'] if r['action']=='restore_full']
    baselines={'checkpoint_format': {'provenance':'cached_executed_controls',
        'unchanged_completes':controls['baseline']['completed'],
        'unchanged_semantic_valid':controls['baseline']['semantic_valid'],
        'fixed_patch_current_valid':controls['gold_current']['semantic_valid'],
        'fixed_patch_legacy_valid':controls['gold_legacy']['semantic_valid'],
        'completing_invalid_skip_rejected':not controls['skip_work_exploit']['semantic_valid'],
        'scratch_semantic_valid':controls['scratch_control']['actual_continuation']['semantic_valid'],
        'scratch_durable_code_reward':controls['scratch_control']['terminal_reward']},
        'state_recovery': {'provenance':'cached_executed_recount', 'fixed_full_restore_valid':[r['semantic_valid'] for r in full],
            'common_optimal_actions':state['common_optimal_actions'],
            'requires_inspection_for_safe_recovery':False}}
    if a.replay_controls:
        if not a.fixture: ap.error('--fixture required for live replay')
        import psutil
        from .task import continuation, read_source, LEGACY_RESTORE, GOLD_RESTORE
        source=read_source(a.fixture/'prefix')
        if source.count(LEGACY_RESTORE)!=1: raise ValueError('frozen baseline source pattern changed')
        fixed=source.replace(LEGACY_RESTORE,GOLD_RESTORE)
        live={}
        for name, candidate, legacy in [('unchanged',source,False),('fixed_current',fixed,False),('fixed_legacy',fixed,True)]:
            if psutil.virtual_memory().available<8*1024**3: raise RuntimeError('baseline replay RAM guard')
            live[name]=continuation(a.fixture,a.out/name,candidate,legacy)
        baselines['checkpoint_format']['new_live_replay']=live
    groups={}
    for m in metrics:
        if m['group_id'] is not None:
            key=(m['model_revision'],str(m['group_id']))
            groups.setdefault(key,[]).append(m['terminal_reward'])
    simple_baselines_pass = (baselines['checkpoint_format']['fixed_patch_current_valid']
        and baselines['checkpoint_format']['fixed_patch_legacy_valid']
        and all(baselines['state_recovery']['fixed_full_restore_valid'])
        and 'restore_full' in baselines['state_recovery']['common_optimal_actions'])
    if a.replay_controls:
        assert not live['unchanged']['semantic_valid']
        assert live['fixed_current']['semantic_valid'] and live['fixed_legacy']['semantic_valid']
    diagnostics={str(n):group_information([g for g in groups.values() if len(g)==n],n) for n in (4,8)}
    out={'source':'executed','analysis':'saved/model receipts; live controls only when explicitly requested',
        'research_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'baselines':baselines,'trajectories':metrics,'rollout_groups':diagnostics,
        'group_note':'No group IDs means no sampled G=4/G=8 estimate; no grouping across models or dependent retries.',
        'gate2':('NO-GO: these existing cases admit fixed source/full-restore baselines.'
                 if simple_baselines_pass else 'UNRESOLVED: inspect baseline outcomes'),
        'inputs_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [state_path,control_path,*a.trajectory]}}
    (a.out/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'trajectories':len(metrics),'gate2':out['gate2'],'rollout_groups':diagnostics},indent=2))


if __name__=='__main__':main()
