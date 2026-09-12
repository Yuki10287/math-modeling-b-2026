"""Q3 model v2: persistent feedback, arbitrary coverage, joint local actions.

Every virtual movement and observation goes through api.measure/api.clear.
The statistical design scores never replace the continuous safety certificates.
"""
from __future__ import annotations
import math
import numpy as np
import geometry as core
from belief_model import FeedbackModel
from local_policy import choose_action, area_scenarios
from recovery import _recover
from v1_solver import solve_multi as solve_v1, rolling_route


def clear_at_stop(model, active, q):
    """Serve any other source already guaranteed within 20 m, without moving."""
    for c, item in list(model.beliefs.items()):
        if c != active and np.max(np.linalg.norm(item['P']-q, axis=1)) <= 20-1e-6:
            model.trace.append(dict(phase='shared_clear', channel=c, position=q.tolist(),
                                    polygon=item['P'].tolist(), certified=True))
            result = model.clear(q, c)
            if result['clear_result'] != 'success':
                item['conflict'] = 'shared_certified_clear_failed'


def opportunistic(model, *, active=None, shared=True):
    """Buy useful observations at an already reached point; movement cost is zero."""
    if len(model.cleared) == 16:
        return
    q = np.asarray(model.position).copy()
    unknown = model.unknown()
    eligible = []
    for c in unknown:
        gain = model.coverage.channel_gain(q, c)
        residual = model.coverage.uncovered_count(c)
        if gain and (gain == residual or gain >= max(300, model.search_fraction*residual)):
            eligible.append(c)
    for c in sorted(eligible, key=lambda c: (c != model.channel, c)):
        if len(model.discovered) == 16:
            break
        if not model.recorded(c, q):
            model.trace.append(dict(phase='opportunity_search', channel=c, position=q.tolist()))
            model.measure(q, c)
    clear_at_stop(model, active, q)
    if not shared:
        return
    candidates = []
    for c, item in list(model.beliefs.items()):
        P, witness = item['P'], item['witness']
        if c == active or item['state'] or model.recorded(c, q) or item['near'] is not None:
            continue
        if np.linalg.norm(q-witness) < 80 or not core.reception_certified(P, q, witness):
            continue
        radius = core.mec(P)[1]
        if radius <= 20:
            continue
        radii = []
        for g in area_scenarios(P, 4):
            if np.linalg.norm(g-q) <= 5:
                radii.append(5.)
            else:
                angle = math.degrees(math.atan2(*(g-q)[::-1]))
                radii.append(core.mec(core.wedge(P, q, angle))[1])
        # Information value is a proxy, evaluated without access to source truth.
        gain = radius-float(np.mean(radii))
        if gain > 30 and np.mean(radii) < .7*radius:
            candidates.append((gain, c))
    for gain, c in sorted(candidates, reverse=True)[:2]:
        model.trace.append(dict(phase='shared_bearing', channel=c, position=q.tolist(),
                               estimated_radius_reduction_m=gain))
        model.measure(q, c)
    clear_at_stop(model, active, q)


def service(model, channel, mode, shared, max_steps, opportunities, candidate_mode='standard'):
    for step in range(max_steps):
        item = model.beliefs[channel]
        if item['conflict']:
            break
        P, s = item['P'], np.asarray(model.position).copy()
        model.trace.append(dict(phase='localize', channel=channel, step=step,
            position=s.tolist(), radius_m=core.mec(P)[1], polygon=P.tolist()))
        try:
            if item['near'] is not None:
                action = dict(kind='clear', q=item['near'], state=None,
                              rationale='near_observation', certified=True)
            else:
                action = choose_action(P, s, item['witness'], channel,
                    current_channel=model.channel, state=item['state'], mode=mode,
                    observed_positions=model.positions[channel], candidate_mode=candidate_mode)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            item['conflict'] = str(exc)
            break
        item['state'] = action.get('state')
        model.trace.append(dict(phase='local_action', channel=channel,
            action=action['kind'], position=np.asarray(action['q']).tolist(),
            rationale=action['rationale'], certified=action.get('certified', False),
            optical_path=action.get('optical_path'), polygon=P.tolist(),
            estimated_s=action.get('estimated_s')))
        if action['kind'] == 'clear':
            reply = model.clear(action['q'], channel)
            if reply['clear_result'] == 'success':
                if opportunities:
                    opportunistic(model, shared=shared)
                return True
            if action.get('certified'):
                item['conflict'] = 'certified_clear_failed'
                break
        else:
            reply = model.measure(action['q'], channel)
            if reply['measure_result'] == 'no_signal':
                item['conflict'] = 'certified_reception_failed'
                break
            if opportunities:
                opportunistic(model, active=channel, shared=shared)
    item = model.beliefs[channel]
    if item['first']['measure_result'] == 'near':
        return model.clear(item['anchor'], channel)['clear_result'] == 'success'
    return _recover(model, channel, item['anchor'], item['first']['svd_deg'],
                    model.trace, item['conflict'] or 'local_step_limit')


def fixed_scan_tasks(model, unknown):
    """Only retain fixed stations if they still exclude some unresolved area."""
    return [dict(kind='scan', key=i, position=p.tolist())
            for i, p in enumerate(core.coverage_points()) if model.coverage.gain(p, unknown)]


def solve_multi(api, policy='time', schedule='lean', trace=None, max_local_steps=30):
    if schedule in ('baseline', 'v1'):
        return solve_v1(api, policy=policy, schedule='baseline' if schedule == 'baseline' else 'dynamic',
                        trace=trace, max_local_steps=max_local_steps)
    if schedule not in ('coverage', 'shared', 'optical', 'full', 'adaptive', 'short', 'adaptive_short', 'range', 'adaptive_range', 'lean'):
        raise ValueError('unknown schedule')
    if policy != 'time':
        raise ValueError('v2 uses the time model')
    if isinstance(max_local_steps, bool) or not isinstance(max_local_steps, int) or max_local_steps < 0:
        raise ValueError('max_local_steps must be nonnegative integer')
    trace = [] if trace is None else trace
    model = FeedbackModel(api, trace, range_cuts=schedule in ('range', 'adaptive_range', 'lean'),
                          search_fraction=.30 if schedule == 'lean' else .15)
    shared = schedule != 'coverage'
    mode = 'hybrid' if schedule in ('full', 'adaptive', 'short', 'adaptive_short', 'range', 'adaptive_range', 'lean') else 'optical' if schedule == 'optical' else 'baseline'
    candidate_mode = 'short' if schedule in ('short', 'adaptive_short', 'range', 'adaptive_range', 'lean') else 'standard'
    status = 'iteration_limit'
    for iteration in range(100):
        if model.certificate()['valid']:
            status = 'complete'
            break
        unknown = model.unknown()
        sources = [dict(kind='source', key=c, position=core.mec(item['P'])[0].tolist())
                   for c, item in sorted(model.beliefs.items())]
        scans = fixed_scan_tasks(model, unknown)
        if schedule in ('adaptive', 'adaptive_short', 'adaptive_range', 'lean') and unknown and iteration < 60:
            from scan_planning import build_scan_tasks
            scans = build_scan_tasks(model.coverage, unknown, model.position, sources)
        tasks = scans+sources
        if not tasks:
            status = 'incomplete_no_tasks'
            break
        route, distance = rolling_route(np.asarray(model.position), tasks)
        trace.append(dict(phase='plan', iteration=iteration, position=np.asarray(model.position).tolist(),
            route=route, estimated_remaining_distance_m=distance,
            unresolved_channels=unknown))
        task = route[0]
        if task['kind'] == 'scan':
            q = np.array(task['position'])
            for c in sorted(unknown, key=lambda c: (c != model.channel, c)):
                if len(model.discovered) == 16:
                    break
                if model.coverage.channel_gain(q, c):
                    model.measure(q, c)
            if shared:
                opportunistic(model, shared=True)
        elif not service(model, task['key'], mode, shared, max_local_steps, True, candidate_mode):
            status = 'recovery_exhausted'
            break
    certificate = model.certificate()
    return dict(status=status, complete=certificate['valid'], cleared=len(model.cleared),
        completion_certificate=certificate,
        fallback_count=sum(x.get('phase') == 'fallback_start' for x in trace),
        model_configuration=dict(schedule=schedule, local_mode=mode, candidate_mode=candidate_mode, shared_bearings=shared,
            fixed_radius_halfplanes=model.range_cuts,
            opportunistic_search_fraction=model.search_fraction,
            observation_bound_degrees=math.degrees(core.ANGLE_EPS), cell_size_m=20,
            design_prior='uniform area quadrature for ranking only', risk_weight=.2))
