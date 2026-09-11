"""Independent candidate. Frozen Q3 source and official entry remain unchanged."""
import math
import numpy as np
from bootstrap import core, baseline, FeedbackModel
from local_policy import choose_action
from recovery import _recover
from scan_planning import build_scan_tasks
from v1_solver import rolling_route
from probe_policy import choose_probe

VARIANTS = ('reference', 'probe', 'scan', 'both')


def solve_multi(api, *, variant='probe', trace=None, max_local_steps=30):
    if variant not in VARIANTS:
        raise ValueError('unknown exploration variant')
    trace = [] if trace is None else trace
    model = FeedbackModel(api, trace, range_cuts=True, search_fraction=.30)
    used_probes = {c:0 for c in range(1,21)}
    probe_outcomes = dict(direction=0, near=0, no_signal=0)

    def service(channel, hints):
        for step in range(max_local_steps):
            item = model.beliefs[channel]
            if item['conflict']:
                break
            P, s = item['P'], np.asarray(model.position).copy()
            trace.append(dict(phase='localize', channel=channel, step=step,
                position=s.tolist(), radius_m=core.mec(P)[1], polygon=P.tolist()))
            try:
                if item['near'] is not None:
                    action = dict(kind='clear', q=item['near'], state=None,
                                  rationale='near_observation', certified=True)
                else:
                    action = choose_action(P, s, item['witness'], channel,
                        current_channel=model.channel, state=item['state'], mode='hybrid',
                        observed_positions=model.positions[channel], candidate_mode='short')
                    if variant in ('probe','both') and used_probes[channel] < 2:
                        action = choose_probe(model, channel, action, hints)
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                item['conflict'] = str(exc)
                break
            item['state'] = action.get('state')
            trace.append(dict(phase='local_action', channel=channel, action=action['kind'],
                position=np.asarray(action['q']).tolist(), rationale=action['rationale'],
                certified=action.get('certified',False), optical_path=action.get('optical_path'),
                polygon=P.tolist(), estimated_s=action.get('estimated_s'),
                uncertain_reception=action.get('uncertain_reception',False),
                design_signal_mass=action.get('design_signal_mass')))
            if action['kind'] == 'clear':
                reply = model.clear(action['q'], channel)
                if reply['clear_result'] == 'success':
                    baseline.opportunistic(model, shared=True)
                    return True
                if action.get('certified'):
                    item['conflict'] = 'certified_clear_failed'
                    break
            else:
                is_probe = action.get('uncertain_reception',False)
                if is_probe:
                    used_probes[channel] += 1
                reply = model.measure(action['q'], channel)
                if is_probe:
                    probe_outcomes[reply['measure_result']] += 1
                    trace.append(dict(phase='probe_feedback', channel=channel,
                        result=reply['measure_result'], count=used_probes[channel]))
                if reply['measure_result']=='no_signal' and not is_probe:
                    item['conflict'] = 'certified_reception_failed'
                    break
                baseline.opportunistic(model, active=channel, shared=True)
        item = model.beliefs[channel]
        if item['first']['measure_result']=='near':
            return model.clear(item['anchor'],channel)['clear_result']=='success'
        return _recover(model,channel,item['anchor'],item['first']['svd_deg'],
                        trace,item['conflict'] or 'local_step_limit')

    status = 'iteration_limit'
    for iteration in range(100):
        if model.certificate()['valid']:
            status = 'complete'
            break
        unknown = model.unknown()
        sources = [dict(kind='source',key=c,position=core.mec(item['P'])[0].tolist())
                   for c,item in sorted(model.beliefs.items())]
        scans = baseline.fixed_scan_tasks(model,unknown)
        if unknown and iteration<60:
            scans = build_scan_tasks(model.coverage,unknown,model.position,sources)
        tasks = scans+sources
        if not tasks:
            status = 'incomplete_no_tasks'
            break
        route,distance = rolling_route(np.asarray(model.position),tasks)
        trace.append(dict(phase='plan',iteration=iteration,position=np.asarray(model.position).tolist(),
            route=route,estimated_remaining_distance_m=distance,unresolved_channels=unknown))
        task = route[0]
        if variant in ('scan','both'):
            from scan_preview import choose_scan_task
            alternative = choose_scan_task(model,tasks,route)
            if alternative is not None:
                task = alternative
                trace.append(dict(phase='scan_preview_choice',position=task['position'],key=task['key']))
        if task['kind']=='scan':
            q = np.array(task['position'])
            for c in sorted(unknown,key=lambda c:(c!=model.channel,c)):
                if len(model.discovered)==16:
                    break
                if model.coverage.channel_gain(q,c):
                    model.measure(q,c)
            baseline.opportunistic(model,shared=True)
        elif not service(task['key'],[np.asarray(t['position']) for t in route if t!=task]):
            status = 'recovery_exhausted'
            break
    certificate = model.certificate()
    return dict(status=status,complete=certificate['valid'],cleared=len(model.cleared),
        completion_certificate=certificate,fallback_count=sum(r.get('phase')=='fallback_start' for r in trace),
        experiment_variant=variant,probe_outcomes=probe_outcomes,probes_by_channel=used_probes,
        model_configuration=dict(base='frozen_time_lean',maximum_probes_per_channel=2,
                                 observation_bound_degrees=math.degrees(core.ANGLE_EPS)))
