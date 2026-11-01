"""Re-score saved DEV groups; export a portable arithmetic audit, never train.

--archive points to the historical modal_train directory (author-created pickle).
The portable packet retains grading fixtures and identical completions. --check
re-scores every reward and recounts advantages without the private pickle cache.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import dataclasses
from types import SimpleNamespace
import numpy as np

RUNS = ['esft_full', 'ecsft', 'eg_v0', 'eg_v2', 'eg_v0_s1', 'eg_v2_s1', 'eg_v0_s2', 'eg_v2_s2']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def advantages(values, normalize=True):
    r = np.asarray(values, dtype=float)
    a = r - r.mean()
    return a / (r.std(ddof=1) + 1e-4) if normalize else a


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r['policy'], r['episode'])].append(r)
    assert all(len(g) == 8 and [r['sample'] for r in g] == list(range(8)) for g in groups.values())
    details = []
    for (policy, ep), g in groups.items():
        rewards = {v: [r[v] for r in g] for v in ('v0', 'v1', 'v2')}
        centered = {v: advantages(x, False).tolist() for v, x in rewards.items()}
        normalized = {v: advantages(x).tolist() for v, x in rewards.items()}
        details.append(dict(policy=policy, episode=ep, centered=centered, normalized=normalized,
                            centered_different=not np.allclose(centered['v1'], centered['v2'], atol=1e-12, rtol=0),
                            normalized_different=not np.allclose(normalized['v1'], normalized['v2'], atol=1e-12, rtol=0),
                            zero_variance={v: len(set(x)) == 1 for v, x in rewards.items()}))
    gap = [r for r in rows if r['v1'] and not r['v2']]
    def counts(rs):
        return dict(Counter(f"{r['kind']} | {r['role']} | {r['failure']}" for r in rs))
    def one(rs, gs):
        pos = [r for r in rs if r['v1']]
        return dict(n=len(rs), groups=len(gs), rewards={v:sum(r[v] for r in rs) for v in ('v0','v1','v2')},
                    disagreement=sum(r['v1'] != r['v2'] for r in rs),
                    centered_different=sum(g['centered_different'] for g in gs),
                    normalized_different=sum(g['normalized_different'] for g in gs),
                    zero_variance={v:sum(g['zero_variance'][v] for g in gs) for v in ('v0','v1','v2')},
                    v1_positive=len(pos), repair_ok_given_v1=sum(r['repair_ok'] for r in pos),
                    v1_positive_v2_negative=counts([r for r in pos if not r['v2']]))
    summary = one(rows, details)
    summary['per_policy'] = {p:one([r for r in rows if r['policy']==p], [g for g in details if g['policy']==p]) for p in RUNS}
    summary['gap_by_repair_role'] = dict(Counter(r['role'] for r in gap))
    # Identity of changed groups is insensitive to population/sample std here.
    for g in groups.values():
        x,y=([r[v] for r in g] for v in ('v1','v2'))
        pop=lambda z:(np.asarray(z)-np.mean(z))/(np.std(z)+1e-4)
        assert (not np.allclose(pop(x),pop(y),atol=1e-12,rtol=0)) == (not np.allclose(advantages(x),advantages(y),atol=1e-12,rtol=0))
    return summary, details


def extract(archive):
    import pickle
    from modal_train.core import dev_episode, score
    from rl.first_update import strip_special
    from traindoctor_env.tiera import parse_answer
    eps_path = archive/'cache/dev_eps.pkl'
    eps = pickle.loads(eps_path.read_bytes())
    hashes = {'cache/dev_eps.pkl':sha(eps_path)}
    rows = []
    for name in RUNS:
        p=archive/'results'/f'{name}_eval.json'
        d=json.loads(p.read_text()); hashes[f'results/{p.name}']=sha(p)
        assert d['g']==8 and len(d['completions'])==8*len(eps)
        for i,c in enumerate(d['completions']):
            e=eps[i//8]; assert c['kind']==e[0].kind
            de=dev_episode(e)
            scored=[score(c['text'], de, v) for v in ('V0','V1','V2')]
            a=parse_answer(strip_special(c['text']), {x.id for x in e[2]})
            candidate=next((x for x in e[2] if a and x.id==a.repair), None)
            o=e[3].get(a.repair) if a else None
            failure=('invalid_answer' if not a else 'accepted' if o and o.ok else
                     'no_completed_relaunch' if not o or not o.ckpt_valid else 'semantic_rejection')
            rows.append(dict(policy=name, episode=i//8, sample=i%8, kind=c['kind'],
                  completion=c['text'], ntok=c['ntok'], capped=c['ntok']>=400,
                  v0=float(scored[0].reward),v1=float(scored[1].reward),v2=float(scored[2].reward),
                  valid=scored[2].valid, unsafe=scored[2].unsafe,
                  role=candidate.role if candidate else 'invalid', failure=failure,
                  repair_ok=bool(o and o.ok),
                  outcome=None if not o else {k:getattr(o,k) for k in ('ok','ckpt_valid','steps_completed','params_match_reference','samples_match_reference','transitions_consistent','restores_consistent','sandboxed','exec_mode','detail')}))
    portable = []
    for truth, bundle, menu, table in eps:
        portable.append(dict(truth=dataclasses.asdict(truth), bundle=dataclasses.asdict(bundle),
            menu=[dict(id=c.id,role=c.role,destructive=c.destructive,diff=c.diff(),
                       edits=[dataclasses.asdict(e) for e in c.edits]) for c in menu],
            table={k:{**dataclasses.asdict(o), 'path':''} for k,o in table.items()}))
    assert '/Users/' not in json.dumps(portable), 'private path in portable episode'
    return rows, hashes, portable


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--check',type=Path)
    a=ap.parse_args()
    if a.check:
        rows=[json.loads(x) for x in (a.check/'rows.jsonl').read_text().splitlines()]
        from modal_train.core import score, dev_episode
        from traindoctor_env.scenario import Truth
        from traindoctor_env.bundle import Candidate
        from traindoctor_env.reexec import FixOutcome
        episodes=[]
        for e in json.loads((a.check/'episodes.json').read_text()):
            episodes.append((Truth(**e['truth']),SimpleNamespace(**e['bundle']),
                [Candidate(c['id'], c['role'], (), destructive=c['destructive']) for c in e['menu']],
                {k:FixOutcome(**v) for k,v in e['table'].items()}))
        for r in rows:
            e=dev_episode(episodes[r['episode']])
            assert [score(r['completion'],e,v).reward for v in ('V0','V1','V2')]==[r[v] for v in ('v0','v1','v2')]
        s,g=summarize(rows)
        assert s==json.loads((a.check/'summary.json').read_text())
        assert g==json.loads((a.check/'groups.json').read_text())
        print('PASS: identical-completion rewards and all eight-completion group arithmetic recounted')
        return
    if not a.archive or not a.out:ap.error('--archive and --out required')
    rows, hashes, portable=extract(a.archive)
    summary,groups=summarize(rows)
    assert summary['n']==3072 and summary['disagreement']==70, 'does not reproduce original DEV claim'
    a.out.mkdir(parents=True,exist_ok=False)
    config=dict(source='executed',analysis='historical DEV evaluation re-score; approximation to training groups',
       inputs=hashes,group_size=8,ordering='run_eval: episode-major, eight contiguous samples per episode',
       normalization='sample std (ddof=1), epsilon 1e-4; conventional TRL group normalization, not recovered training tensors',
       masking='no rows dropped; historical config does not enable mask_truncated_completions; cap=400',
       limits=['not training-time rollouts; no token masks/logprobs/gradients recovered',
              'V2 uses transient dev_episode sandboxed=True over dev_container outcomes; not fresh sandbox verification',
              'portable grading fixtures reproduce V0/V1/V2; original cache hash records provenance'],
       capped_completions=sum(r['capped'] for r in rows),versions={'numpy':np.__version__},
       scorer_sha256={p:sha(Path(p)) for p in ('modal_train/core.py','scripts/audit_learning_signal.py')})
    for name,obj in [('config.json',config),('summary.json',summary),('groups.json',groups),('episodes.json',portable)]:
        (a.out/name).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    report=['# Historical DEV learning-signal recount', '',
        'source: executed. Saved DEV evaluation groups, not training-time rollouts. V2 is exploratory dev_container scoring.', '',
        '| Reward | Positive completions | Zero-variance groups / 384 | Zero-variance fraction |',
        '|---|---:|---:|---:|']
    for v in ('v0','v1','v2'):
        report.append(f'| {v.upper()} | {summary["rewards"][v]:.0f} / {summary["n"]} | {summary["zero_variance"][v]} | {summary["zero_variance"][v]/summary["groups"]:.2%} |')
    report += ['', f'V1 != V2: {summary["disagreement"]}/{summary["n"]} ({summary["disagreement"]/summary["n"]:.2%}).',
        f'Centered and normalized advantages differ in {summary["normalized_different"]}/{summary["groups"]} groups ({summary["normalized_different"]/summary["groups"]:.2%}).',
        f'Repair acceptance conditional on V1: {summary["repair_ok_given_v1"]}/{summary["v1_positive"]} ({summary["repair_ok_given_v1"]/summary["v1_positive"]:.2%}).', '',
        '| Failure mechanism / repair role | V1-positive V2-negative |', '|---|---:|']
    report += [f'| {k} | {v} |' for k,v in summary['v1_positive_v2_negative'].items()]
    report += ['', 'Eight contiguous completions per cached episode, separately for each policy; no groups pooled across policies. No completion reaches the 400-token cap. Centering and sample-standard-deviation normalization with epsilon 1e-4 give the same changed-group count; population-std sensitivity agrees.', '',
        'Historical training configured TRL 1.16.0 with DAPO, beta=0 and eight generations. Exact training tensors, EOS/padding masks, logprobs and sampled groups were not retained here. These evaluation-group calculations are not measured gradient differences. The local inspected TRL source is a later 1.17 development snapshot; the normalized values are an explicit formula reconstruction, not an authenticated historical trainer replay.', '',
        'The complete grading fixtures and per-completion scores are exported for re-scoring. Original cache/completion hashes are in config.json. No fresh sandbox or model was run.']
    (a.out/'REPORT.md').write_text('\n'.join(report)+'\n')
    (a.out/'rows.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    print(json.dumps({k:v for k,v in summary.items() if k!='per_policy'},indent=2))

if __name__=='__main__':main()
