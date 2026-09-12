"""Local-only Q4 prototype. Public feedback, direction-safe completion."""
import numpy as np
from directional_cover import DirectionalCover
from localization import Belief, optical_plan, choose_measure
from shared import core


def accepted(reply,key,allowed):
    if not isinstance(reply,dict) or reply.get('accepted',True) is not True or reply.get(key) not in allowed:
        raise ValueError('rejected or malformed feedback')
    if key=='measure_result' and reply[key]=='direction':
        angle=reply.get('svd_deg')
        if isinstance(angle,bool) or not isinstance(angle,(int,float)) or not np.isfinite(angle) or not 0<=angle<360:
            raise ValueError('invalid bearing')
    return reply[key]


def solve_multi(api,policy='active',trace=None,max_active_measures=6):
    if policy not in ('active','optical'):
        raise ValueError('unknown Q4 policy')
    trace=[] if trace is None else trace
    cover=DirectionalCover()
    beliefs={};cleared=set();discovered=set()

    def measure(q,c):
        reply=api.measure(q,c)
        kind=accepted(reply,'measure_result',('no_signal','near','direction'))
        if kind=='no_signal':
            cover.observe_negative(c,q)
            if c in beliefs:beliefs[c].update(q,reply)
        elif c not in beliefs:
            beliefs[c]=Belief(q,reply)
            beliefs[c].negatives=[p.copy() for p in cover.negative[c]]
            discovered.add(c)
        else:
            beliefs[c].update(q,reply)
        if c in beliefs:
            trace.append(dict(phase='belief',channel=c,polygon=beliefs[c].P.tolist(),
                              positives=[x.tolist() for x in beliefs[c].positives],
                              negatives=[x.tolist() for x in beliefs[c].negatives]))
        return kind

    def clear(q,c):
        result=accepted(api.clear(q,c),'clear_result',('success','no_target_in_range'))
        if result=='success':
            cleared.add(c);beliefs.pop(c,None)
            return True
        if c in beliefs:beliefs[c].failed_clears.append(np.asarray(q).copy())
        return False

    def service(c):
        belief=beliefs[c]
        for step in range(max_active_measures+1):
            if belief.near is not None:
                return clear(belief.near,c)
            safe=core.nearest_certified_clear(belief.P,np.asarray(api.position))
            if safe is not None:
                return clear(safe,c)
            plan=optical_plan(belief.P,np.asarray(api.position))
            action=choose_measure(belief,np.asarray(api.position),c,api.channel) if policy=='active' and step<max_active_measures else None
            if action is not None and action['score']<plan['score']:
                trace.append(dict(phase='decision',channel=c,action='measure',
                                  predicted_s=action['score'],optical_s=plan['score'],
                                  signal_mass=action['signal_mass'],scenarios=action['scenarios']))
                measure(action['q'],c)
                continue
            trace.append(dict(phase='optical_plan',channel=c,polygon=belief.P.tolist(),
                              **{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in plan.items()}))
            # Keep the complete cover on failure, independent of scenario weights.
            for q in plan['path']:
                if clear(q,c):return True
            return False
        return False

    # The origin is a useful free-to-reach initial stop, also a lattice vertex.
    q=np.asarray(api.position).copy()
    for iteration in range(len(cover.stations)+1):
        unknown=[c for c in range(1,21) if c not in discovered and not cover.complete(c)] if len(discovered)<16 else []
        for c in sorted(unknown,key=lambda c:(c!=api.channel,c)):
            if len(discovered)==16:break
            if not any(np.linalg.norm(q-p)<1e-7 for p in cover.negative[c]):
                measure(q,c)
        while beliefs:
            c=min(beliefs,key=lambda c:float(np.linalg.norm(core.mec(beliefs[c].P)[0]-api.position)))
            if not service(c):
                return dict(complete=False,reason='local_cover_or_feedback_conflict',cleared=sorted(cleared))
        absent=[c for c in range(1,21) if c not in discovered and cover.complete(c)]
        if len(cleared)==16 or len(cleared)+len(absent)==20:
            certificate=dict(basis='count_upper_bound' if len(cleared)==16 else 'directional_triangle_cover',
                cleared=sorted(cleared),absent=absent,
                channels={str(c):cover.certificate(c) for c in absent},
                spacing=cover.spacing,stations=cover.stations.tolist(),triangles=cover.indices.tolist())
            return dict(complete=True,cleared=sorted(cleared),certificate=certificate)
        unknown=[c for c in range(1,21) if c not in discovered and not cover.complete(c)]
        remaining=cover.needed_stations(unknown)
        if not remaining:break
        station=min(remaining,key=lambda k:(float(np.linalg.norm(cover.stations[k]-api.position)),k))
        q=cover.stations[station].copy()
    return dict(complete=False,reason='discovery_certificate_incomplete',cleared=sorted(cleared))
