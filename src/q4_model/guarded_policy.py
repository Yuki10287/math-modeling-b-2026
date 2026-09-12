"""Ambiguous source type: compare conditional costs before taking the worse type."""
import math
import numpy as np
from localization import optical_plan
from shared import core


def score_at(belief,start,q,channel,current_channel,robust=True,prepared=None):
    gs,scenarios=belief.scenarios() if prepared is None else prepared
    if not scenarios:return None
    weights={}
    if robust:
        for mode in ('omni','directional'):
            rows=[row for row in scenarios if (row[2] is None)==(mode=='omni')]
            if rows:weights[mode]=rows
    else:weights['mixture']=scenarios
    mass={}
    for mode,rows in weights.items():
        hit=np.zeros(len(gs));total=sum(row[4] for row in rows)
        for index,g,n,radius,weight in rows:
            if np.linalg.norm(q-g)<=radius and (n is None or (q-g)@n>=0):
                hit[index]+=weight/total
        mass[mode]=hit
    if max(float(x.sum()) for x in mass.values())<.05:return None
    negative_cost=optical_plan(belief.P,q,gs)['score']
    values=np.zeros(len(gs))
    active=np.flatnonzero(np.sum(list(mass.values()),axis=0))
    for k in active:
        g=gs[k]
        if np.linalg.norm(g-q)<=5:values[k]=5.;continue
        angle=math.degrees(math.atan2(g[1]-q[1],g[0]-q[0]))%360
        tails=[]
        for error in (-1.,0.,1.):
            bearing=round((angle+error)%360,2)%360
            P=core.disk_clip(core.wedge(belief.P,q,bearing),q)
            if not len(P):raise RuntimeError('empty predictive branch')
            tails.append(optical_plan(P,q)['score'])
        values[k]=np.mean(tails)
    costs={mode:float(hit@values+(1-hit.sum())*negative_cost) for mode,hit in mass.items()}
    immediate=float(np.linalg.norm(q-start))/5+5+int(channel!=current_channel)
    return dict(q=q,score=immediate+max(costs.values()),signal_mass=min(float(x.sum()) for x in mass.values()),
                scenarios=len(scenarios),type_costs=costs)


def choose_guarded(belief,start,channel,current_channel):
    prepared=belief.scenarios()
    if not prepared[1]:return None
    center,r=core.mec(belief.P);_,(i,j)=core.diameter(belief.P)
    u=belief.P[j]-belief.P[i]
    u=u/np.linalg.norm(u) if np.linalg.norm(u)>1e-9 else np.array([1.,0.])
    v=np.array([-u[1],u[0]])
    candidates=[center,(center+start)/2]
    candidates.extend(center+along*u+offset*v for along in (0.,-.35*r) for offset in (-100.,-40.,40.,100.))
    best=None
    for q in candidates:
        if any(np.linalg.norm(q-p)<=.1 for p in belief.measured):continue
        result=score_at(belief,start,q,channel,current_channel,prepared=prepared)
        if result is not None and (best is None or result['score']<best['score']):best=result
    return best

