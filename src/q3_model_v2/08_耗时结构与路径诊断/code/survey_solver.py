"""Local-only survey-first structural ablation; frozen lean geometry is reused."""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
MAIN = STUDY.parent / '00_主方案_lean/code'
JOINT = STUDY.parent / '02_联合调度实验/code'
ROOT = STUDY.parents[2]
for directory in (MAIN, JOINT):
    if str(directory) not in sys.path:
        sys.path.append(str(directory))

import geometry as core
import solver as baseline
from belief_model import FeedbackModel
from scan_planning import build_scan_tasks
from v1_solver import rolling_route


def additional_scan_bearings(model):
    """One fresh certified bearing per eligible channel at the reached stop.

    All eligibility is derived from public beliefs. Measurements update the
    original FeedbackModel, and only actual replies become evidence.
    """
    q = np.asarray(model.position).copy()
    for channel in sorted(list(model.beliefs), key=lambda c: (c != model.channel, c)):
        item = model.beliefs.get(channel)
        if item is None or item['near'] is not None or item['conflict']:
            continue
        P = item['P']
        if (model.recorded(channel, q) or core.nearest_certified_clear(P, q) is not None
                or not core.reception_certified(P, q, item['witness'])):
            continue
        model.trace.append(dict(phase='survey_additional_bearing', channel=channel,
            position=q.tolist(), polygon=P.tolist(), witness=item['witness'].tolist(),
            certified_reception=True, one_measurement_per_channel_per_stop=True))
        reply = model.measure(q, channel)
        if reply['measure_result'] == 'no_signal':
            item['conflict'] = 'survey_certified_reception_failed'
            raise RuntimeError('Certified supplemental bearing unexpectedly returned no signal')
    # Success already guaranteed at this reached point incurs zero movement.
    baseline.clear_at_stop(model, None, q)


def solve_multi(api, *, variant='survey', trace=None, max_local_steps=30):
    if variant not in ('reference', 'survey', 'survey_all'):
        raise ValueError('Unknown survey experiment variant')
    if isinstance(max_local_steps, bool) or not isinstance(max_local_steps, int) or max_local_steps < 0:
        raise ValueError('max_local_steps must be a nonnegative integer')
    trace = [] if trace is None else trace
    model = FeedbackModel(api, trace, range_cuts=True, search_fraction=.30)
    status = 'iteration_limit'
    for iteration in range(100):
        if model.certificate()['valid']:
            status = 'complete'
            break
        unknown = model.unknown()
        sources = [dict(kind='source', key=c, position=core.mec(item['P'])[0].tolist())
                   for c, item in sorted(model.beliefs.items())]
        scans = baseline.fixed_scan_tasks(model, unknown)
        if unknown and iteration < 60:
            scans = build_scan_tasks(model.coverage, unknown, model.position, sources)
        tasks = scans + sources
        if not tasks:
            status = 'incomplete_no_tasks'
            break
        route, distance = rolling_route(np.asarray(model.position), tasks)
        trace.append(dict(phase='plan', iteration=iteration, position=np.asarray(model.position).tolist(),
                          route=route, estimated_remaining_distance_m=distance,
                          unresolved_channels=unknown))
        task = route[0]
        if variant != 'reference':
            if unknown:
                eligible = [t for t in route if t['kind'] == 'scan']
                if not eligible:
                    status = 'incomplete_no_covering_scan'
                    break
                task = eligible[0]
            trace.append(dict(phase='survey_selection', iteration=iteration,
                              unknown_channels=unknown, discovered=len(model.discovered),
                              chosen=task, survey_stage=bool(unknown)))
        if task['kind'] == 'scan':
            q = np.asarray(task['position'])
            for channel in sorted(unknown, key=lambda c: (c != model.channel, c)):
                if len(model.discovered) == 16:
                    break
                if model.coverage.channel_gain(q, channel):
                    model.measure(q, channel)
            baseline.opportunistic(model, shared=True)
            if variant == 'survey_all':
                additional_scan_bearings(model)
        elif not baseline.service(model, task['key'], 'hybrid', True, max_local_steps, True, 'short'):
            status = 'recovery_exhausted'
            break
    certificate = model.certificate()
    return dict(status=status, complete=certificate['valid'], cleared=len(model.cleared),
                completion_certificate=certificate,
                fallback_count=sum(x.get('phase') == 'fallback_start' for x in trace),
                model_configuration=dict(schedule='lean' if variant == 'reference' else variant,
                    local_mode='hybrid', candidate_mode='short', shared_bearings=True,
                    fixed_radius_halfplanes=True, opportunistic_search_fraction=.30,
                    observation_bound_degrees=np.degrees(core.ANGLE_EPS), cell_size_m=20,
                    design_prior='uniform area quadrature for ranking only', risk_weight=.2))
