"""Q4 research variants; the first solver and its evidence remain untouched."""
import numpy as np
from localization import Belief,optical_plan,choose_measure
from polar_cover import PolarCover
from route_planning import open_route
from guarded_policy import choose_guarded,score_at
from solver import accepted
from shared import core


VARIANTS=('polar','route','shared','guarded')


def solve_multi(api,variant='shared',trace=None,max_active_measures=6):
    if variant not in VARIANTS:raise ValueError('unknown variant')
    trace=[] if trace is None else trace
    cover=PolarCover();beliefs={};cleared=set();discovered=set();calls=0
    use_route=variant!='polar';sharing=variant in ('shared','guarded');robust=variant=='guarded'

    def unknown():
        return [c for c in range(1,21) if c not in discovered and not cover.complete(c)] if len(discovered)<16 else []

    def measure(q,c,reason):
        nonlocal calls
        before=np.asarray(api.position).copy();calls+=1
        reply=api.measure(q,c);kind=accepted(reply,'measure_result',('no_signal','near','direction'))
        trace.append(dict(phase='actual_action',action='measure',reason=reason,channel=c,event=calls,
                          position=np.asarray(q).tolist(),move_s=float(np.linalg.norm(q-before))/5,result=kind))
        if kind=='no_signal':
            cover.observe_negative(c,q)
            if c in beliefs:beliefs[c].update(q,reply)
        elif c not in beliefs:
            beliefs[c]=Belief(q,reply);beliefs[c].negatives=[p.copy() for p in cover.negative[c]]
            discovered.add(c)
        else:beliefs[c].update(q,reply)
        if c in beliefs:
            b=beliefs[c]
            trace.append(dict(phase='belief',channel=c,polygon=b.P.tolist(),
                positives=[x.tolist() for x in b.positives],negatives=[x.tolist() for x in b.negatives]))

    def clear(q,c,reason):
        nonlocal calls
        before=np.asarray(api.position).copy();calls+=1
        kind=accepted(api.clear(q,c),'clear_result',('success','no_target_in_range'))
        trace.append(dict(phase='actual_action',action='clear',reason=reason,channel=c,event=calls,
            position=np.asarray(q).tolist(),move_s=float(np.linalg.norm(q-before))/5,result=kind))
        if kind=='success':cleared.add(c);beliefs.pop(c,None);return True
        if c in beliefs:beliefs[c].failed_clears.append(np.asarray(q).copy())
        return False

    def service(c):
        belief=beliefs[c]
        for step in range(max_active_measures+1):
            if belief.near is not None:return clear(belief.near,c,'near_clear')
            safe=core.nearest_certified_clear(belief.P,np.asarray(api.position))
            if safe is not None:return clear(safe,c,'certified_clear')
            plan=optical_plan(belief.P,np.asarray(api.position))
            choose=choose_guarded if robust else choose_measure
            action=choose(belief,np.asarray(api.position),c,api.channel) if step<max_active_measures else None
            if action is not None and action['score']<plan['score']:
                trace.append(dict(phase='decision',channel=c,action='measure',predicted_s=action['score'],
                    optical_s=plan['score'],signal_mass=action['signal_mass'],scenarios=action['scenarios'],
                    type_costs=action.get('type_costs')))
                measure(action['q'],c,'source_measure');continue
            trace.append(dict(phase='optical_plan',channel=c,polygon=belief.P.tolist(),
                **{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in plan.items()}))
            for q in plan['path']:
                if clear(q,c,'optical_cover'):return True
            return False
        return False

    def share_at_stop(newly_found):
        q=np.asarray(api.position).copy();choices=[]
        for c,b in list(beliefs.items()):
            if np.max(np.linalg.norm(b.P-q,axis=1))<=20-1e-6:
                if not clear(q,c,'shared_clear'):return False
                continue
            if c in newly_found or min(np.linalg.norm(q-p) for p in b.measured)<80:continue
            # Only paid observations at an already reached stop; no extra travel.
            result=score_at(b,q,q,c,api.channel,robust=robust)
            if result is None:continue
            gain=optical_plan(b.P,q)['score']-result['score']
            if gain>0:choices.append((gain,c))
        for gain,c in sorted(choices,reverse=True)[:2]:
            # Re-evaluate after any intervening switch, since switching is charged.
            result=score_at(beliefs[c],q,q,c,api.channel,robust=robust)
            if result is not None and result['score']<optical_plan(beliefs[c].P,q)['score']:
                trace.append(dict(phase='shared_prediction',channel=c,gain_s=gain,
                                  signal_mass=result['signal_mass']))
                measure(q,c,'shared_measure')
        return True

    def scan(q):
        new=set()
        for c in sorted(unknown(),key=lambda c:(c!=api.channel,c)):
            if len(discovered)==16:break
            if any(np.linalg.norm(q-p)<1e-7 for p in cover.negative[c]):continue
            measure(q,c,'scan')
            if c in discovered:new.add(c)
        return share_at_stop(new) if sharing else True

    if not scan(np.asarray(api.position).copy()):return dict(complete=False,reason='shared_clear_conflict')
    for iteration in range(len(cover.stations)+17):
        absent=[c for c in range(1,21) if c not in discovered and cover.complete(c)]
        if len(cleared)==16 or len(cleared)+len(absent)==20:
            return dict(complete=True,cleared=sorted(cleared),certificate=dict(
                basis='count_upper_bound' if len(cleared)==16 else 'directional_triangle_cover',
                cleared=sorted(cleared),absent=absent,channels={str(c):cover.certificate(c) for c in absent},
                spacing=cover.spacing,stations=cover.stations.tolist(),triangles=cover.indices.tolist()))
        pending=cover.needed_stations(unknown());tasks=[];points=[]
        for k in pending:tasks.append(('scan',k));points.append(cover.stations[k])
        for c,b in beliefs.items():tasks.append(('source',c));points.append(core.mec(b.P)[0])
        if not tasks:break
        if use_route:
            order=open_route(np.array(points),api.position);task,index=tasks[order[0]]
        elif beliefs:
            task='source';index=min(beliefs,key=lambda c:float(np.linalg.norm(core.mec(beliefs[c].P)[0]-api.position)))
        else:
            task='scan';index=min(pending,key=lambda k:(float(np.linalg.norm(cover.stations[k]-api.position)),k))
        trace.append(dict(phase='task_choice',task=task,index=index,known=len(beliefs),unknown=len(unknown()),pending_stations=len(pending)))
        if task=='scan':
            if not scan(cover.stations[index].copy()):return dict(complete=False,reason='shared_clear_conflict')
        elif not service(index):return dict(complete=False,reason='local_cover_or_feedback_conflict',cleared=sorted(cleared))
    return dict(complete=False,reason='discovery_certificate_incomplete',cleared=sorted(cleared))
