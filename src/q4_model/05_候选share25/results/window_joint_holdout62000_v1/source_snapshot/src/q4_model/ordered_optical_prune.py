"""Original 25-station sharing solver with proof-certified ordered optical pruning.

Static equivalent of make_solver() from the frozen experiment:
05_候选share25/code/ordered_optical_prune_experiment.py
SHA256: 1717df71ff1d3307375427d5c17608390b96e94752e4cd11168229d3a6fe7e97
Only impossible optical points are skipped; the original complete cover and
all deletion proofs remain in trace. Local sensing, global routing and stopping
are inherited unchanged. Default variant is explicitly share25 for this entry.
"""
from fractions import Fraction as F

import numpy as np

from localization import Belief, optical_plan, choose_measure
from polar_cover import PolarCover
from route_planning import open_route
from guarded_policy import score_at
from solver import accepted
from shared import core
from negative_region import contract_position


VARIANTS = ('resume25', 'share25')


def demand(condition, message):
    if not condition:
        raise ValueError(message)


def rational_point(p):
    return tuple(F(float(x)) for x in p)


def cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def exact_hull(points):
    points = sorted(set(map(rational_point, points)))
    demand(bool(points), 'empty polygon')
    if len(points) <= 1:
        return points
    halves = []
    for sequence in (points, points[::-1]):
        stack = []
        for p in sequence:
            while len(stack) >= 2 and cross(stack[-2], stack[-1], p) <= 0:
                stack.pop()
            stack.append(p)
        halves.append(stack)
    return halves[0][:-1]+halves[1][:-1]


def distance_squared(q, hull):
    """Exact distance to the convex hull of the ACTUAL binary-float vertices."""
    q = rational_point(q)
    if len(hull) >= 3 and all(cross(a, b, q) >= 0 for a, b in zip(hull, hull[1:]+hull[:1])):
        return F(0)
    values = []
    for a, b in zip(hull, hull[1:]+hull[:1]):
        v = (b[0]-a[0], b[1]-a[1])
        length2 = v[0]*v[0]+v[1]*v[1]
        t = ((q[0]-a[0])*v[0]+(q[1]-a[1])*v[1])/length2 if length2 else F(0)
        t = min(F(1), max(F(0), t))
        values.append(sum((q[i]-a[i]-t*v[i])**2 for i in (0, 1)))
    return min(values)


def prune_path(P, positives, negatives, path):
    """Selector receives geometry and prior measurements, never outcomes/truth."""
    original = np.asarray(P, float)
    contracted, info = contract_position(original, positives, negatives)
    region = np.asarray(contracted, float) if info['applied'] else original
    h0, h1 = exact_hull(original), exact_hull(region)
    d0 = [distance_squared(q, h0) for q in path]
    d1 = [distance_squared(q, h1) for q in path]
    # The output float hull may slightly expand original P. Keep both sound
    # source sets by intersecting their logical constraints: either disjoint
    # disk proof is sufficient, without constructing another float polygon.
    removed0 = [i for i, d in enumerate(d0) if d > 400]
    removed1 = [i for i, d in enumerate(d1) if d > 400]
    removed = sorted(set(removed0) | set(removed1))
    kept = [i for i in range(len(path)) if i not in set(removed)]
    demand(bool(kept), 'all optical points pruned; fail closed')
    return dict(original_polygon=original.tolist(), polygon=region.tolist(),
                positives=[np.asarray(p).tolist() for p in positives],
                negatives=[np.asarray(p).tolist() for p in negatives],
                original_path=np.asarray(path).tolist(), original_removed_indices=removed0,
                contracted_removed_indices=removed1, removed_indices=removed,
                kept_indices=kept, original_distances_squared=[str(d) for d in d0],
                contracted_distances_squared=[str(d) for d in d1], info=info)


_prune_path = prune_path


def solve_multi(api, variant='share25', trace=None, max_active_measures=6):
    if variant not in VARIANTS:
        raise ValueError('unknown task-sharing variant')
    trace = [] if trace is None else trace
    cover = PolarCover()
    beliefs, local_steps = {}, {}
    cleared, discovered = set(), set()
    calls = 0

    def unknown():
        if len(discovered) >= 16:
            return []
        return [c for c in range(1, 21) if c not in discovered and not cover.complete(c)]

    def measure(q, c, reason):
        nonlocal calls
        before = np.asarray(api.position).copy()
        calls += 1
        reply = api.measure(q, c)
        kind = accepted(reply, 'measure_result', ('no_signal', 'near', 'direction'))
        trace.append(dict(phase='actual_action', action='measure', reason=reason, channel=c,
                          event=calls, position=np.asarray(q).tolist(),
                          move_s=float(np.linalg.norm(q-before))/5, result=kind))
        if kind == 'no_signal':
            cover.observe_negative(c, q)
            if c in beliefs:
                beliefs[c].update(q, reply)
        elif c not in beliefs:
            beliefs[c] = Belief(q, reply)
            beliefs[c].negatives = [p.copy() for p in cover.negative[c]]
            local_steps[c] = 0
            discovered.add(c)
        else:
            beliefs[c].update(q, reply)
        if c in beliefs:
            b = beliefs[c]
            trace.append(dict(phase='belief', channel=c, polygon=b.P.tolist(),
                              positives=[x.tolist() for x in b.positives],
                              negatives=[x.tolist() for x in b.negatives]))

    def clear(q, c, reason):
        nonlocal calls
        before = np.asarray(api.position).copy()
        calls += 1
        kind = accepted(api.clear(q, c), 'clear_result', ('success', 'no_target_in_range'))
        trace.append(dict(phase='actual_action', action='clear', reason=reason, channel=c,
                          event=calls, position=np.asarray(q).tolist(),
                          move_s=float(np.linalg.norm(q-before))/5, result=kind))
        if kind == 'success':
            cleared.add(c)
            beliefs.pop(c, None)
            return True
        if c in beliefs:
            beliefs[c].failed_clears.append(np.asarray(q).copy())
        return False

    def service(c):
        belief = beliefs[c]
        if belief.near is not None:
            return 'cleared' if clear(belief.near, c, 'near_clear') else 'failed'
        safe = core.nearest_certified_clear(belief.P, np.asarray(api.position))
        if safe is not None:
            return 'cleared' if clear(safe, c, 'certified_clear') else 'failed'
        plan = optical_plan(belief.P, np.asarray(api.position))
        action = (choose_measure(belief, np.asarray(api.position), c, api.channel)
                  if local_steps[c] < max_active_measures else None)
        if action is not None and action['score'] < plan['score']:
            trace.append(dict(phase='decision', channel=c, action='measure',
                              predicted_s=action['score'], optical_s=plan['score'],
                              signal_mass=action['signal_mass'], scenarios=action['scenarios'],
                              type_costs=action.get('type_costs')))
            measure(action['q'], c, 'source_measure')
            local_steps[c] += 1
            # The global route is reconsidered after exactly one active view.
            return 'measured'
        pruning = _prune_path(belief.P, belief.positives, belief.negatives, plan['path'])
        trace.append(dict(phase='ordered_optical_prune', channel=c, event=calls, **pruning))
        execution_path = [plan['path'][i] for i in pruning['kept_indices']]
        trace.append(dict(phase='optical_plan', channel=c, polygon=belief.P.tolist(),
                          **{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                             for k, v in plan.items()}))
        # Failed optical attempts do not stop or truncate the continuous cover.
        for q in execution_path:
            if clear(q, c, 'optical_cover'):
                return 'cleared'
        return 'failed'

    def share_at_stop(newly_found):
        q = np.asarray(api.position).copy()
        choices = []
        for c, b in list(beliefs.items()):
            if np.max(np.linalg.norm(b.P-q, axis=1)) <= 20-1e-6:
                if not clear(q, c, 'shared_clear'):
                    return False
                continue
            if c in newly_found or min(np.linalg.norm(q-p) for p in b.measured) < 80:
                continue
            value = score_at(b, q, q, c, api.channel, robust=False)
            if value is None:
                continue
            gain = optical_plan(b.P, q)['score']-value['score']
            if gain > 0:
                choices.append((gain, c))
        for gain, c in sorted(choices, reverse=True)[:2]:
            # Switching may have changed after the preceding shared action.
            value = score_at(beliefs[c], q, q, c, api.channel, robust=False)
            if value is not None and value['score'] < optical_plan(beliefs[c].P, q)['score']:
                trace.append(dict(phase='shared_prediction', channel=c, gain_s=gain,
                                  signal_mass=value['signal_mass']))
                measure(q, c, 'shared_measure')
        return True

    def scan(q):
        newly_found = set()
        for c in sorted(unknown(), key=lambda c: (c != api.channel, c)):
            if len(discovered) == 16:
                break
            if any(np.linalg.norm(q-p) < 1e-7 for p in cover.negative[c]):
                continue
            measure(q, c, 'scan')
            if c in discovered:
                newly_found.add(c)
        return share_at_stop(newly_found)

    if not scan(np.asarray(api.position).copy()):
        return dict(complete=False, reason='shared_clear_conflict')
    # Fixed stations are visited at most once for each still-unknown channel.
    # Every source service either consumes one of six lifetime active views or
    # clears it with a complete fallback. Sharing adds no outer iterations.
    for _ in range(len(cover.stations)+16*(max_active_measures+1)+2):
        absent = [c for c in range(1, 21) if c not in discovered and cover.complete(c)]
        if len(cleared) == 16 or len(cleared)+len(absent) == 20:
            return dict(complete=True, cleared=sorted(cleared), certificate=dict(
                basis='count_upper_bound' if len(cleared) == 16 else 'directional_triangle_cover',
                cleared=sorted(cleared), absent=absent,
                channels={str(c): cover.certificate(c) for c in absent},
                spacing=cover.spacing, stations=cover.stations.tolist(),
                triangles=cover.indices.tolist()))
        pending = cover.needed_stations(unknown())
        tasks, points = [], []
        for k in pending:
            tasks.append(('scan', k))
            points.append(cover.stations[k])
        for c, b in beliefs.items():
            tasks.append(('source', c))
            points.append(core.mec(b.P)[0])
        if not tasks:
            break
        order = open_route(np.asarray(points), api.position)
        task, index = tasks[order[0]]
        trace.append(dict(phase='task_choice', task=task, index=index, known=len(beliefs),
                          unknown=len(unknown()), pending_stations=len(pending)))
        if task == 'scan':
            if not scan(cover.stations[index].copy()):
                return dict(complete=False, reason='shared_clear_conflict')
        else:
            status = service(index)
            if status == 'failed':
                return dict(complete=False, reason='local_cover_or_feedback_conflict',
                            cleared=sorted(cleared))
            if variant == 'share25' and not share_at_stop(set()):
                return dict(complete=False, reason='service_shared_clear_conflict',
                            cleared=sorted(cleared))
    return dict(complete=False, reason='discovery_certificate_incomplete', cleared=sorted(cleared))
