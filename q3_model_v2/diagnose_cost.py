"""Local counterfactual action audit. Hidden truth is used by the harness only.

Saved holdout layouts from the previous round now serve as development data.
The decision function still receives PublicAPI and evidence, never source truth.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import numpy as np

import geometry as core
import forecast_solver as engine
from local_policy import choose_action, area_scenarios, short_candidates, _measure_score, best_optical_plan
from validate_model import (BoundedArena, PublicAPI, independent_cells, independent_time_audit,
    observation_audit, certificate_audit, trace_truth_audit)

ROOT=Path(__file__).resolve().parent
CASES=[('uniform','8304-uniform-extreme'),('sparse','8402-sparse10-smooth'),
       ('boundary','8501-boundary10-smooth'),('uniform','8305-uniform-extreme'),
       ('one-side','8603-one_side10-smooth'),('uniform','8311-uniform-hash')]


def clone(model,arena):
    result=copy.copy(model)
    result.__dict__=copy.deepcopy({k:v for k,v in model.__dict__.items() if k not in ('api','decision_hook','forced_action')})
    arena=copy.deepcopy(arena)
    result.api=PublicAPI(arena)
    result.decision_hook=None
    result.forced_action=None
    return result,arena


def options(model,c):
    item=model.beliefs[c]
    P,s,w=item['P'],np.asarray(model.position),item['witness']
    args=dict(current_channel=model.channel,state=item['state'],mode='hybrid',
              observed_positions=model.positions[c],candidate_mode='short')
    chosen=choose_action(P,s,w,c,**args)
    samples=area_scenarios(P)
    qs=short_candidates(P,s,w)
    if core.reception_certified(P,s,w):qs.insert(0,s.copy())
    qs=[q for q in qs if all(np.linalg.norm(q-old)>.1 for old in model.positions[c])]
    ranked=sorted([(_measure_score(P,s,q,samples)+int(c!=model.channel),i,q) for i,q in enumerate(qs)])
    selected=[dict(chosen, label='selected')]
    ids=[0, min(range(len(ranked)),key=lambda i:np.linalg.norm(ranked[i][2]-s))]
    ids.append(max(range(min(8,len(ranked))),key=lambda i:np.linalg.norm(ranked[i][2]-ranked[0][2])))
    for i in ids:
        score,_,q=ranked[i]
        if any(a['kind']=='measure' and np.linalg.norm(a['q']-q)<.1 for a in selected):continue
        selected.append(dict(kind='measure',q=q,state=None,certified=False,rationale='diagnostic_alternative',
                             estimated_s=score,label=f'measure_{i}'))
    plan=best_optical_plan(P,s) if core.mec(P)[1]<=180 else None
    if plan is not None and not any(a['kind']=='clear' for a in selected):
        path=plan['path']
        selected.append(dict(kind='clear',q=path[0],state=dict(remaining=path[1:].tolist(),
            cover_size=len(path),exhausted=len(path)==1),certified=len(path)==1,
            optical_path=path.tolist(),estimated_s=plan['score'],rationale='diagnostic_optical',label='optical'))
    return selected


def legacy_forecast_checks(model,c,q):
    P=model.beliefs[c]['P']
    counts=dict(future_measures=0,uncertified_reception=0,repeated_position=0,branches=0)
    for g in area_scenarios(P):
        if np.linalg.norm(g-q)<=5:continue
        bearing=math.degrees(math.atan2(*(g-q)[::-1]))
        for error in (-1.,0.,1.):
            counts['branches']+=1
            region=core.wedge(P,q,bearing+error)
            s=np.asarray(q).copy()
            seen=[*model.positions[c],s.copy()]
            for _ in range(3):
                center,radius=core.mec(region)
                if radius<=20-1e-6:break
                _,(i,j)=core.diameter(region)
                v=region[j]-region[i];v/=max(np.linalg.norm(v),1e-10)
                next_q=center+.3*radius*np.array([-v[1],v[0]])
                counts['future_measures']+=1
                counts['uncertified_reception']+=int(not core.reception_certified(region,next_q,s))
                counts['repeated_position']+=int(any(np.linalg.norm(next_q-old)<=.1 for old in seen))
                if np.linalg.norm(g-next_q)<=5:break
                angle=math.degrees(math.atan2(*(g-next_q)[::-1]))
                region=core.wedge(region,next_q,angle)
                seen.append(next_q.copy());s=next_q
    return counts


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',type=int,default=6)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    rows=[]
    centers=independent_cells()
    for population,name in CASES[:args.cases]:
        saved=json.loads((ROOT/'results'/f'joint-holdout-{population}'/'cases'/f'{name}-route_interleave.json').read_text())
        case=saved['result']
        arena=BoundedArena(saved['sources'],case['seed'],case['field'],max_actions=3000)
        snapshots={}
        def capture(model,c):
            item=model.beliefs[c]
            radius=core.mec(item['P'])[1]
            category='broad' if radius>180 else 'optical' if radius>20.01 else None
            if category is None or category in snapshots or item['state'] is not None:return
            model_copy,arena_copy=clone(model,arena)
            # The hook is after the pending action's step counter was incremented.
            model_copy.local_steps[c]-=1
            iteration=next(r['iteration'] for r in reversed(model.trace) if r.get('phase')=='plan')
            snapshots[category]=(model_copy,arena_copy,c,iteration)
        result=engine.solve_multi(PublicAPI(arena),decision_hook=capture)
        assert result['complete'] and arena.events==saved['events'],'adapter changed baseline actions'
        for category,(model,original,c,iteration) in snapshots.items():
            actions=options(model,c)
            evaluations=[]
            for action in actions:
                local,local_arena=clone(model,original)
                local.interleave=False
                local.forced_action=dict(channel=c,action=copy.deepcopy(action))
                engine.service(local,c,'hybrid',True,30,False,'short')
                assert c in local_arena._removed
                independent_time_audit(local_arena);trace_truth_audit(local_arena,local.trace)
                full,full_arena=clone(model,original)
                full.forced_action=dict(channel=c,action=copy.deepcopy(action))
                outcome=engine.resume_model(full,start_iteration=iteration)
                assert outcome['complete'] and full_arena.evaluation()['all_cleared']
                independent_time_audit(full_arena)
                observations=observation_audit(full_arena)
                certificate_audit(full_arena,outcome,observations,centers)
                trace_truth_audit(full_arena,full.trace)
                if action['label']=='selected':assert full_arena.events==arena.events,'resume changed baseline'
                evaluation=dict(label=action['label'],kind=action['kind'],position=np.asarray(action['q']).tolist(),
                    predicted_local_s=action.get('estimated_s'),actual_local_s=local_arena.time_s-original.time_s,
                    actual_remaining_s=full_arena.time_s-original.time_s,complete=True,
                    forecast_checks=legacy_forecast_checks(model,c,action['q']) if action['kind']=='measure' else None)
                evaluations.append(evaluation)
            first=evaluations[0]
            row=dict(case=name,category=category,channel=c,prefix_actions=len(original.events),
                radius_m=core.mec(model.beliefs[c]['P'])[1],alternatives=evaluations,
                selected_local_regret_s=first['actual_local_s']-min(e['actual_local_s'] for e in evaluations),
                selected_global_regret_s=first['actual_remaining_s']-min(e['actual_remaining_s'] for e in evaluations),
                public_snapshot=dict(position=np.asarray(model.position).tolist(),current_channel=model.channel,
                    polygon=model.beliefs[c]['P'].tolist(),witness=model.beliefs[c]['witness'].tolist(),
                    observed_positions=[p.tolist() for p in model.positions[c]]))
            rows.append(row)
            (args.out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
            print(json.dumps({k:row[k] for k in ('case','category','channel','selected_local_regret_s','selected_global_regret_s')}),flush=True)
    summary=dict(local_only=True,official_simulator_used=False,selection='diagnostic, intentionally chosen wins/losses; not an unbiased performance estimate',
        states=len(rows),branches=sum(len(r['alternatives']) for r in rows),
        mean_local_regret_s=float(np.mean([r['selected_local_regret_s'] for r in rows])),
        mean_global_regret_s=float(np.mean([r['selected_global_regret_s'] for r in rows])),
        states_with_better_local_action=sum(r['selected_local_regret_s']>1 for r in rows),
        states_with_better_global_action=sum(r['selected_global_regret_s']>1 for r in rows),
        baseline_replay_and_resume_exact=True,all_branch_audits_passed=True)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':main()
