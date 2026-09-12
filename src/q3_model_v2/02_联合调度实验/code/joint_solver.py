"""Experimental joint route solver. The frozen lean solver is unchanged.

Every virtual movement and observation goes through api.measure/api.clear.
The statistical design scores never replace the continuous safety certificates.
"""
from __future__ import annotations
import math
import numpy as np
import geometry as core
from belief_model import FeedbackModel
from local_policy import choose_action as base_action, area_scenarios, _measure_score
from joint_planning import ServiceContext, joint_route, shortest_visit, route_cost
from flexible_coverage import flexible_route
from scan_planning import build_scan_tasks
from recovery import _recover
from v1_solver import rolling_route
from solver import opportunistic, fixed_scan_tasks

DEFAULT_SCHEDULE = 'route_interleave'


def choose_action(model, channel, candidate_mode):
    item = model.beliefs[channel]
    P, s, witness = item['P'], np.asarray(model.position), item['witness']
    action = base_action(P, s, witness, channel, current_channel=model.channel,
        state=item['state'], mode='hybrid', observed_positions=model.positions[channel],
        candidate_mode=candidate_mode)
    if model.joint_mode not in ('joint', 'fused') or item['state'] is not None:
        return action
    context = ServiceContext(model, channel, flexible_route if model.joint_mode=='fused' else joint_route)
    if action['kind'] == 'clear' and action.get('certified'):
        after = context.route[0]['position'] if context.route else None
        q = shortest_visit(s, after, P, 20-1e-6, action['q'])
        action = dict(action, q=q, rationale='joint_certified_clear_entry_exit')
    baseline_cost = action.get('estimated_s', np.linalg.norm(action['q']-s)/5+5)
    baseline_saving, channels = context.value(action['q'])
    best = (baseline_cost-baseline_saving, dict(action, joint_scan_channels=channels,
            joint_coverage_saving_s=baseline_saving))
    if action['kind'] == 'clear' and action.get('certified'):
        return best[1]
    samples = area_scenarios(P)
    observed = model.positions[channel]
    for q in context.candidates(P, s, witness, action['q']):
        if any(np.linalg.norm(q-old) <= .1 for old in observed):
            continue
        saving, channels = context.value(q)
        if saving <= 0:
            continue
        cost = _measure_score(P, s, q, samples) + int(model.channel != channel)
        score = cost-saving
        if score < best[0]-1e-6:
            best = (score, dict(kind='measure', q=q, state=None, certified=False,
                rationale='joint_localization_residual_coverage', estimated_s=cost,
                joint_coverage_saving_s=saving, joint_scan_channels=channels))
    return best[1]


def execute_joint_scan(model, channels):
    q = np.asarray(model.position).copy()
    for c in sorted(channels, key=lambda c: (c != model.channel, c)):
        if c in model.unknown() and not model.recorded(c, q):
            model.trace.append(dict(phase='joint_search', channel=c, position=q.tolist()))
            model.measure(q, c)


def service(model, channel, mode, shared, max_steps, opportunities, candidate_mode='standard'):
    for step in range(max_steps):
        item = model.beliefs[channel]
        if model.interleave:
            step = model.local_steps.get(channel, 0)
            if step >= max_steps:
                item['conflict'] = 'local_step_limit'
                break
            model.local_steps[channel] = step+1
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
                action = choose_action(model, channel, candidate_mode)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            item['conflict'] = str(exc)
            break
        item['state'] = action.get('state')
        model.trace.append(dict(phase='local_action', channel=channel,
            action=action['kind'], position=np.asarray(action['q']).tolist(),
            rationale=action['rationale'], certified=action.get('certified', False),
            optical_path=action.get('optical_path'), polygon=P.tolist(),
            estimated_s=action.get('estimated_s'),
            joint_coverage_saving_s=action.get('joint_coverage_saving_s'),
            joint_scan_channels=action.get('joint_scan_channels')))
        if action['kind'] == 'clear':
            reply = model.clear(action['q'], channel)
            if reply['clear_result'] == 'success':
                execute_joint_scan(model, action.get('joint_scan_channels', []))
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
            execute_joint_scan(model, action.get('joint_scan_channels', []))
            if opportunities:
                opportunistic(model, active=channel, shared=shared)
        if model.interleave:
            return True  # An unfinished source remains a task with all evidence/state.
    item = model.beliefs[channel]
    if item['first']['measure_result'] == 'near':
        return model.clear(item['anchor'], channel)['clear_result'] == 'success'
    return _recover(model, channel, item['anchor'], item['first']['svd_deg'],
                    model.trace, item['conflict'] or 'local_step_limit')


def solve_multi(api, policy='time', schedule=DEFAULT_SCHEDULE, trace=None, max_local_steps=30):
    if schedule not in ('route', 'joint', 'flex', 'fused', 'interleave', 'route_interleave'):
        raise ValueError('unknown schedule')
    if policy != 'time':
        raise ValueError('v2 uses the time model')
    if isinstance(max_local_steps, bool) or not isinstance(max_local_steps, int) or max_local_steps < 0:
        raise ValueError('max_local_steps must be nonnegative integer')
    trace = [] if trace is None else trace
    model = FeedbackModel(api, trace, range_cuts=True, search_fraction=.30)
    model.joint_mode = schedule
    model.interleave = schedule in ('interleave', 'route_interleave')
    model.local_steps = {}
    shared, mode, candidate_mode = True, 'hybrid', 'short'
    status = 'iteration_limit'
    for iteration in range(600 if model.interleave else 100):
        if model.certificate()['valid']:
            status = 'complete'
            break
        unknown = model.unknown()
        sources = [dict(kind='source', key=c, position=core.mec(item['P'])[0].tolist())
                   for c, item in sorted(model.beliefs.items())]
        scans = fixed_scan_tasks(model, unknown)
        if iteration < 60:
            if schedule == 'interleave':
                scans = build_scan_tasks(model.coverage, unknown, model.position, sources) if unknown else []
                tasks, _ = rolling_route(np.asarray(model.position), scans+sources)
            else:
                route_builder = flexible_route if schedule in ('flex', 'fused') else joint_route
                tasks, _ = route_builder(model.coverage, unknown, model.position, sources)
        else:
            tasks = scans+sources
        if not tasks:
            status = 'incomplete_no_tasks'
            break
        route, distance = ((tasks, route_cost(model.position, tasks)) if iteration < 60
                           else rolling_route(np.asarray(model.position), tasks))
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
            interleaved_service=model.interleave,
            scan_optimizer='flexible' if schedule in ('flex','fused') else 'original' if schedule=='interleave' else 'disk_intersection',
            local_coverage_value=schedule in ('joint','fused'),
            opportunistic_search_fraction=model.search_fraction,
            observation_bound_degrees=math.degrees(core.ANGLE_EPS), cell_size_m=20,
            design_prior='uniform area quadrature for ranking only', risk_weight=.2))
