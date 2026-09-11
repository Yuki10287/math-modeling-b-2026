"""Executable, bounded-error continuation costs for ranking local actions.

Only the cost evaluation changes. The outer belief, candidate locations,
area quadrature, risk weight, actual optical plans and global scheduler are
retained. Constant error fields are explicit design scenarios, not estimates
of the official simulator's error distribution. Future scenario trajectories
are not observations and never update the real evidence model.
"""
from __future__ import annotations
import math
import numpy as np
import geometry as core
from local_policy import (choose_action as original_action, area_scenarios,
    short_candidates, best_optical_plan, optical_cost)


def continuation(P, s, g, error, observed, *, variant, steps=3, audit=None):
    """Run a simple feasible tail policy; replanning can choose differently.

    legal: repair reception, repeated positions, range clipping and clear cost.
    bounded: additionally propagate fixed nonzero measurement error + rounding.
    policy: additionally allow a certified small-region optical termination.
    """
    P=np.asarray(P,float).copy()
    s=np.asarray(s,float).copy()
    seen=[np.asarray(q,float).copy() for q in observed]
    total=0.
    for _ in range(steps):
        c,r=core.mec(P)
        q=core.nearest_certified_clear(P,s) if r<=20-1e-6 else None
        if q is not None:
            if audit is not None:audit.append(dict(kind='clear',q=q.tolist(),certified=True))
            return total+np.linalg.norm(q-s)/5+5
        if variant=='policy' and r<=40:
            plan=best_optical_plan(P,s,max_parts=6)
            if plan is not None:
                costs,_=optical_cost(plan['path'],s,np.asarray([g]))
                if audit is not None:audit.append(dict(kind='optical',path=plan['path'].tolist(),polygon=P.tolist()))
                return total+float(costs[0])
        _,(i,j)=core.diameter(P)
        v=P[j]-P[i];v/=max(float(np.linalg.norm(v)),1e-10)
        q=c+.3*r*np.array([-v[1],v[0]])
        used=lambda p:any(np.linalg.norm(p-old)<=.1 for old in seen)
        if used(q) or not core.reception_certified(P,q,s):
            candidates=short_candidates(P,s,s)
            candidates=[p for p in candidates if not used(p)]
            if not candidates:
                # Rejected rollout only. The actual selector retains other
                # candidates/optical recovery; no infeasible move is executed.
                return math.inf
            q=min(candidates,key=lambda p:np.linalg.norm(p-q))
        assert core.reception_certified(P,q,s) and not used(q)
        if audit is not None:audit.append(dict(kind='measure',q=q.tolist(),polygon=P.tolist(),witness=s.tolist()))
        total+=np.linalg.norm(q-s)/5+5
        if np.linalg.norm(g-q)<=5:return total+5
        bearing=math.degrees(math.atan2(*(g-q)[::-1]))%360
        reading=bearing if variant=='legal' else round((bearing+error)%360,2)%360
        P=core.disk_clip(core.wedge(P,q,reading),q)
        if not len(P):return math.inf
        seen.append(q.copy());s=q
    # Finite-horizon terminal proxy retained explicitly. This term is an
    # estimate, not a proven cost or a completed optical action.
    c,r=core.mec(P)
    return total+np.linalg.norm(c-s)/5+5+2*r/5+10


def measure_score(P,s,q,samples,observed,*,variant,risk=.2):
    values=[]
    for g in samples:
        delta=g-q
        if np.linalg.norm(delta)<=5:
            values.append(5.)
            continue
        bearing=math.degrees(math.atan2(delta[1],delta[0]))%360
        per_error=[]
        for error in (-1.,0.,1.):
            reading=(bearing+error) if variant=='legal' else round((bearing+error)%360,2)%360
            region=core.disk_clip(core.wedge(P,q,reading),q)
            if not len(region):return math.inf
            per_error.append(continuation(region,q,g,error,[*observed,q],variant=variant))
        values.append(float(np.mean(per_error)))
    return np.linalg.norm(q-s)/5+5+(1-risk)*float(np.mean(values))+risk*float(np.max(values))


def choose_forecast_action(model,channel,*,variant):
    if variant not in ('legal','bounded','policy'):raise ValueError('unknown forecast variant')
    item=model.beliefs[channel]
    P,s,w=item['P'],np.asarray(model.position),item['witness']
    kwargs=dict(current_channel=model.channel,state=item['state'],mode='hybrid',
                observed_positions=model.positions[channel],candidate_mode='short')
    if item['state'] is not None or core.mec(P)[1]<=20-1e-6:
        return original_action(P,s,w,channel,**kwargs)
    observed=[w,*model.positions[channel]]
    used=lambda q:any(np.linalg.norm(q-old)<=.1 for old in observed)
    options=short_candidates(P,s,w)
    if not used(s) and core.reception_certified(P,s,w):options.insert(0,s.copy())
    options=[q for q in options if not used(q)]
    samples=area_scenarios(P)
    ranked=[(measure_score(P,s,q,samples,observed,variant=variant),i,q) for i,q in enumerate(options)]
    if not ranked:return original_action(P,s,w,channel,**kwargs)
    score,_,q=min(ranked,key=lambda row:(row[0],row[1]))
    score+=int(model.channel!=channel)
    plan=best_optical_plan(P,s) if core.mec(P)[1]<=180 else None
    if plan is not None and plan['score']<=score:
        path=plan['path']
        return dict(kind='clear',q=path[0],state=dict(remaining=path[1:].tolist(),
            cover_size=len(path),exhausted=len(path)==1),cover_size=len(path),certified=len(path)==1,
            rationale='certified_optical_partition_cheaper',optical_path=path.tolist(),
            estimated_s=plan['score'],worst_cover_s=plan['worst_s'],alternative_measure_s=score)
    if not math.isfinite(score):return original_action(P,s,w,channel,**kwargs)
    return dict(kind='measure',q=q,state=None,certified=False,
                rationale='cost_forecast_'+variant,estimated_s=score)
