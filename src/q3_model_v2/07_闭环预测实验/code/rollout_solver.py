"""Lean continuation using its actual localization, feedback and replanning.

The same continuation runs the real candidate and each isolated scenario.
Scenario selection is never recursive: rollouts always continue with lean.
"""
import numpy as np

from bootstrap import baseline, core, FeedbackModel
from scan_planning import build_scan_tasks
from v1_solver import rolling_route


def plan(model, iteration):
    unknown = model.unknown()
    sources = [dict(kind='source', key=c, position=core.mec(item['P'])[0].tolist())
               for c, item in sorted(model.beliefs.items())]
    scans = baseline.fixed_scan_tasks(model, unknown)
    if unknown and iteration < 60:
        scans = build_scan_tasks(model.coverage, unknown, model.position, sources)
    tasks = scans + sources
    if not tasks:
        return [], [], 0.
    route, distance = rolling_route(np.asarray(model.position), tasks)
    return tasks, route, distance


def execute(model, task, max_local_steps=30):
    if task['kind'] == 'scan':
        q = np.asarray(task['position'])
        for c in sorted(model.unknown(), key=lambda c: (c != model.channel, c)):
            if len(model.discovered) == 16:
                break
            if model.coverage.channel_gain(q, c):
                model.measure(q, c)
        baseline.opportunistic(model, shared=True)
        return True
    return baseline.service(model, task['key'], 'hybrid', True,
                            max_local_steps, True, 'short')


def continue_lean(model, *, start_iteration=0, first_task=None, selector=None,
                  checkpoint=None, max_local_steps=30):
    """Consume one global iteration per task, retaining the 60/100 boundaries."""
    if not isinstance(start_iteration, int) or not 0 <= start_iteration <= 100:
        raise ValueError('invalid continuation iteration')
    status = 'iteration_limit'
    for iteration in range(start_iteration, 100):
        if model.certificate()['valid']:
            status = 'complete'
            break
        tasks, route, distance = plan(model, iteration)
        if not tasks:
            status = 'incomplete_no_tasks'
            break
        model.trace.append(dict(phase='plan', iteration=iteration,
            position=np.asarray(model.position).tolist(), route=route,
            estimated_remaining_distance_m=distance, unresolved_channels=model.unknown()))
        if checkpoint is not None:
            checkpoint(model, iteration, tasks, route)
        if first_task is not None and iteration == start_iteration:
            if not any(task['kind'] == first_task['kind'] and task['key'] == first_task['key']
                       and np.allclose(task['position'], first_task['position'], rtol=0, atol=1e-7)
                       for task in tasks):
                raise ValueError('forced first task must belong to the current plan')
            task = first_task
        else:
            task = route[0]
            if selector is not None:
                selected = selector(model, iteration, tasks, route)
                if selected is not None:
                    task = selected
        if not execute(model, task, max_local_steps):
            status = 'recovery_exhausted'
            break
    certificate = model.certificate()
    return dict(status=status, complete=certificate['valid'], cleared=len(model.cleared),
        completion_certificate=certificate,
        fallback_count=sum(r.get('phase') == 'fallback_start' for r in model.trace),
        model_configuration=dict(base='frozen_time_lean', continuation='actual_lean',
                                 start_iteration=start_iteration, max_iterations=100))


def solve_multi(api, *, variant='closed', trace=None, selector=None, checkpoint=None):
    if variant not in ('reference', 'closed', 'legacy'):
        raise ValueError('unknown closed-loop experiment variant')
    trace = [] if trace is None else trace
    model = FeedbackModel(api, trace, range_cuts=True, search_fraction=.30)
    if variant == 'closed' and selector is None:
        from closed_preview import ClosedLoopSelector
        selector = ClosedLoopSelector()
    if variant == 'legacy':
        from scan_preview import choose_scan_task
        selector = lambda model, iteration, tasks, route: choose_scan_task(model, tasks, route)
    result = continue_lean(model, selector=selector, checkpoint=checkpoint)
    result['experiment_variant'] = variant
    return result
