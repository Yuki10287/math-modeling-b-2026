"""Independent complete ordered-pruning proof audit for accepted action ledgers.

Derived from frozen optical auditors v1 and v2, without importing the planner
or any experiment module. Every proof check uses explicit exceptions, including
under python -O. Input events contain action, channel, position and the relevant
measure_result or clear_result; timing and hidden source positions are unused.
"""

from fractions import Fraction as F
import math
from negative_region_proof import verify_certificate

def exact(p):
    return tuple((F(float(x)) for x in p))

def minus(a, b):
    return (a[0] - b[0], a[1] - b[1])

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]

def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]

def hull(points):
    points = sorted(set(map(exact, points)))
    if not points:
        raise ValueError('Proof obligation failed: points')
    if len(points) < 3:
        return points
    first, current, result = (points[0], points[0], [])
    while True:
        result.append(current)
        candidate = next((p for p in points if p != current))
        for p in points:
            a, b = (minus(candidate, current), minus(p, current))
            sign = cross(a, b)
            if sign < 0 or (sign == 0 and dot(b, b) > dot(a, a)):
                candidate = p
        current = candidate
        if current == first:
            return result
        if not len(result) <= len(points):
            raise ValueError('Proof obligation failed: len(result) <= len(points)')

def independent_distance2(query, polygon):
    q, vertices = (exact(query), hull(polygon))
    if len(vertices) == 1:
        v = minus(q, vertices[0])
        return dot(v, v)
    edges = list(zip(vertices, vertices[1:] + vertices[:1]))
    if len(vertices) > 2 and all((cross(minus(b, a), minus(q, a)) >= 0 for a, b in edges)):
        return F(0)
    bounds = []
    for a, b in edges:
        v, w = (minus(b, a), minus(q, a))
        length2, projection = (dot(v, v), dot(v, w))
        if projection <= 0:
            bounds.append(dot(w, w))
        elif projection >= length2:
            end = minus(q, b)
            bounds.append(dot(end, end))
        else:
            bounds.append(cross(v, w) ** 2 / length2)
    return min(bounds)

def verify_old_rectangle(plan):
    P, path = ([exact(p) for p in plan['polygon']], [exact(p) for p in plan['path']])
    origin = exact(plan['origin'])
    R = [[F(float(v)) for v in row] for row in plan['rotation']]
    determinant = R[0][0] * R[1][1] - R[0][1] * R[1][0]
    if not determinant != 0:
        raise ValueError('singular old rectangle transform')

    def local(p):
        x, y = minus(p, origin)
        return ((R[1][1] * x - R[0][1] * y) / determinant, (-R[1][0] * x + R[0][0] * y) / determinant)

    def world(p):
        return tuple((origin[k] + R[k][0] * p[0] + R[k][1] * p[1] for k in range(2)))
    lo, hi = (exact(plan['lower']), exact(plan['upper']))
    nx, ny = plan['cells']
    if not (type(nx) is int and type(ny) is int and (nx > 0) and (ny > 0)):
        raise ValueError('Proof obligation failed: type(nx) is int and type(ny) is int and (nx > 0) and (ny > 0)')
    if not len(path) == nx * ny:
        raise ValueError('Proof obligation failed: len(path) == nx * ny')
    if not all((lo[k] <= hi[k] for k in range(2))):
        raise ValueError('Proof obligation failed: all((lo[k] <= hi[k] for k in range(2)))')
    local_P = list(map(local, P))
    lower = [min(lo[k], min((p[k] for p in local_P))) for k in range(2)]
    upper = [max(hi[k], max((p[k] for p in local_P))) for k in range(2)]
    ownership = {}
    for q in path:
        p = local(q)
        cell = tuple((0 if hi[k] == lo[k] else int(round((p[k] - lo[k]) * count / (hi[k] - lo[k]) - F(1, 2))) for k, count in enumerate((nx, ny))))
        if not (0 <= cell[0] < nx and 0 <= cell[1] < ny):
            raise ValueError('Proof obligation failed: 0 <= cell[0] < nx and 0 <= cell[1] < ny')
        if not cell not in ownership:
            raise ValueError('multiple points assigned same old rectangle cell')
        ownership[cell] = q
    largest = F(0)
    for x in range(nx):
        for y in range(ny):
            bounds = []
            for k, (index, count) in enumerate(((x, nx), (y, ny))):
                low = lower[k] if index == 0 else lo[k] + (hi[k] - lo[k]) * index / count
                high = upper[k] if index == count - 1 else lo[k] + (hi[k] - lo[k]) * (index + 1) / count
                bounds.append((low, high))
            q = ownership[x, y]
            for a in bounds[0]:
                for b in bounds[1]:
                    delta = minus(world((a, b)), q)
                    distance = dot(delta, delta)
                    if not distance <= 400:
                        raise ValueError('old continuous optical cover exceeds radius 20')
                    largest = max(largest, distance)
    return dict(passed=True, cells=nx * ny, max_corner_distance_squared=float(largest), method='exact inverse transform; outermost bounds include every original vertex; all cell corners checked')

def verify_pruning(proof, plan, belief, observed_positive, observed_negative):
    if not proof['original_polygon'] == belief['polygon'] == plan['polygon']:
        raise ValueError("Proof obligation failed: proof['original_polygon'] == belief['polygon'] == plan['polygon']")
    if not (proof['positives'] == belief['positives'] and proof['negatives'] == belief['negatives']):
        raise ValueError("Proof obligation failed: proof['positives'] == belief['positives'] and proof['negatives'] == belief['negatives']")
    if not all((tuple(p) in observed_positive for p in proof['positives'])):
        raise ValueError('unobserved positive premise')
    if not all((tuple(p) in observed_negative for p in proof['negatives'])):
        raise ValueError('unobserved negative premise')
    if not proof['original_path'] == plan['path']:
        raise ValueError('original path modified')
    if not type(proof['info']['applied']) is bool:
        raise ValueError("Proof obligation failed: type(proof['info']['applied']) is bool")
    if proof['info']['applied']:
        verify_certificate(proof['info']['certificate'], proof['original_polygon'], proof['positives'], proof['negatives'], proof['polygon'], require_reduction=True)
    elif not proof['polygon'] == proof['original_polygon']:
        raise ValueError('unproved contraction')
    old_cover = verify_old_rectangle(plan)
    d0 = [independent_distance2(q, proof['original_polygon']) for q in plan['path']]
    d1 = [independent_distance2(q, proof['polygon']) for q in plan['path']]
    if not [F(s) for s in proof['original_distances_squared']] == d0:
        raise ValueError('forged original distance')
    if not [F(s) for s in proof['contracted_distances_squared']] == d1:
        raise ValueError('forged contracted distance')
    r0 = [i for i, d in enumerate(d0) if d > 400]
    r1 = [i for i, d in enumerate(d1) if d > 400]
    removed = sorted(set(r0) | set(r1))
    kept = [i for i in range(len(d0)) if i not in removed]
    if not all((type(i) is int for key in ('original_removed_indices', 'contracted_removed_indices', 'removed_indices', 'kept_indices') for i in proof[key])):
        raise ValueError("Proof obligation failed: all((type(i) is int for key in ('original_removed_indices', 'contracted_removed_indices', 'removed_indices', 'kept_indices') for i in proof[key]))")
    if not (proof['original_removed_indices'] == r0 and proof['contracted_removed_indices'] == r1):
        raise ValueError("Proof obligation failed: proof['original_removed_indices'] == r0 and proof['contracted_removed_indices'] == r1")
    if not proof['removed_indices'] == removed:
        raise ValueError('not exactly justified deletion union')
    if not (proof['kept_indices'] == kept and kept):
        raise ValueError('retained points reordered, moved or empty')
    return dict(passed=True, original_cover=old_cover, removed=len(removed), kept=len(kept), from_original=len(r0), additional_from_contraction=len(set(r1) - set(r0)))

def _verify_geometry(case, replay=None):
    trace, events = (case['trace'], case['events'])
    beliefs, positives, negatives = ({}, {}, {})
    latest, checks, removed_events = (0, [], set())
    replay_by_event = {p['event']: p for p in replay['proofs']} if replay else {}
    if not (not replay or len(replay_by_event) == len(replay['proofs'])):
        raise ValueError('duplicate replay plan event')
    used_replay = set()
    for index, row in enumerate(trace):
        if row['phase'] == 'actual_action':
            latest += 1
            if not row['event'] == latest:
                raise ValueError("Proof obligation failed: row['event'] == latest")
            actual = events[latest - 1]
            if not (row['action'] == actual['action'] and row['channel'] == actual['channel'] and (row['position'] == actual['position'])):
                raise ValueError("Proof obligation failed: row['action'] == actual['action'] and row['channel'] == actual['channel'] and (row['position'] == actual['position'])")
            if not row['result'] == actual['measure_result' if actual['action'] == 'measure' else 'clear_result']:
                raise ValueError("Proof obligation failed: row['result'] == actual['measure_result' if actual['action'] == 'measure' else 'clear_result']")
            if actual['action'] == 'measure':
                ledger = negatives if actual['measure_result'] == 'no_signal' else positives
                ledger.setdefault(actual['channel'], set()).add(tuple(actual['position']))
        if row['phase'] == 'belief':
            beliefs[row['channel']] = row
        if replay and row['phase'] == 'optical_plan':
            if not latest in replay_by_event:
                raise ValueError('original fallback absent from replay proof')
            proof, plan = (replay_by_event[latest], row)
            used_replay.add(latest)
            actual_start = index + 1
        elif not replay and row['phase'] == 'ordered_optical_prune':
            proof, plan = (row, trace[index + 1])
            if not plan['phase'] == 'optical_plan':
                raise ValueError("Proof obligation failed: plan['phase'] == 'optical_plan'")
            actual_start = index + 2
        else:
            continue
        channel = proof['channel']
        if not (proof['event'] == latest and plan['channel'] == channel):
            raise ValueError("Proof obligation failed: proof['event'] == latest and plan['channel'] == channel")
        audit = verify_pruning(proof, plan, beliefs[channel], positives.get(channel, set()), negatives.get(channel, set()))
        expected_ids = list(range(len(plan['path']))) if replay else proof['kept_indices']
        success = False
        count = 0
        for action in trace[actual_start:]:
            if not (action['phase'] == 'actual_action' and action['action'] == 'clear'):
                raise ValueError("Proof obligation failed: action['phase'] == 'actual_action' and action['action'] == 'clear'")
            if not (action['channel'] == channel and action['reason'] == 'optical_cover'):
                raise ValueError("Proof obligation failed: action['channel'] == channel and action['reason'] == 'optical_cover'")
            old_index = expected_ids[count]
            if not action['position'] == plan['path'][old_index]:
                raise ValueError("Proof obligation failed: action['position'] == plan['path'][old_index]")
            event = events[action['event'] - 1]
            if not (event['position'] == action['position'] and event['channel'] == channel and (event['action'] == 'clear')):
                raise ValueError("Proof obligation failed: event['position'] == action['position'] and event['channel'] == channel and (event['action'] == 'clear')")
            if replay and old_index in proof['removed_indices']:
                if not event['clear_result'] == 'no_target_in_range':
                    raise ValueError('proved deletion would remove success')
                removed_events.add(action['event'])
            count += 1
            if event['clear_result'] == 'success':
                success = True
                if not old_index in proof['kept_indices']:
                    raise ValueError('first success is not retained')
                break
        if not success:
            raise ValueError('Proof obligation failed: success')
        audit.update(channel=channel, executed_original_points=count)
        checks.append(audit)
    if not latest == len(events):
        raise ValueError('Proof obligation failed: latest == len(events)')
    if replay:
        if not used_replay == set(replay_by_event):
            raise ValueError('Proof obligation failed: used_replay == set(replay_by_event)')
        if not sorted(removed_events) == replay['removed_event_numbers']:
            raise ValueError('replay removed unjustified events')
        expected = [{k: v for k, v in e.items() if k != 'time_s'} for i, e in enumerate(events, 1) if i not in removed_events]
        actual = [{k: v for k, v in e.items() if k != 'time_s'} for e in replay['events']]
        if not expected == actual:
            raise ValueError('replay changed a retained action or feedback')
    return dict(passed=True, plans=len(checks), rows=checks, removed_events=len(removed_events))

def _verify_complete(case, replay=None):
    trace = case['trace']
    plans = [i for i, row in enumerate(trace) if row['phase'] == 'optical_plan']
    declared = [i for i, row in enumerate(trace) if row['phase'] == 'ordered_optical_prune']
    if replay is None:
        if not [i + 1 for i in declared] == plans:
            raise ValueError('each original optical plan requires its adjacent pruning proof')
    else:
        if not not declared:
            raise ValueError('restricted replay must use the original baseline trace')
        if not len(replay['proofs']) == len(plans):
            raise ValueError('each original plan requires one replay proof')
    assigned = []
    for index in plans:
        plan = trace[index]
        succeeded = False
        for action in trace[index + 1:]:
            if not action['phase'] == 'actual_action':
                raise ValueError('optical block abandoned without a successful clear')
            if not (action['reason'] == 'optical_cover' and action['action'] == 'clear'):
                raise ValueError("Proof obligation failed: action['reason'] == 'optical_cover' and action['action'] == 'clear'")
            if not action['channel'] == plan['channel']:
                raise ValueError("Proof obligation failed: action['channel'] == plan['channel']")
            assigned.append(action['event'])
            if action['result'] == 'success':
                succeeded = True
                break
            if not action['result'] == 'no_target_in_range':
                raise ValueError("Proof obligation failed: action['result'] == 'no_target_in_range'")
        if not succeeded:
            raise ValueError('Proof obligation failed: succeeded')
    expected = [row['event'] for row in trace if row['phase'] == 'actual_action' and row['reason'] == 'optical_cover']
    if not assigned == expected:
        raise ValueError('an actual optical action has no proof, or belongs to duplicate proofs')
    if not len(set(assigned)) == len(assigned):
        raise ValueError('Proof obligation failed: len(set(assigned)) == len(assigned)')
    result = _verify_geometry(case, replay)
    if not result['plans'] == len(plans):
        raise ValueError("Proof obligation failed: result['plans'] == len(plans)")
    result.update(all_optical_plans_audited=True, all_actual_optical_actions_accounted=True, actual_optical_actions=len(assigned), auditor_version=2)
    return result

def verify_trace(trace, events):
    """Validate every optical proof and real action; raise ValueError on failure.

    events must be the independent accepted-action ledger from the HTTP client,
    not a ledger reconstructed from the decision trace. The checker needs no
    hidden source truth and does not contact the server.
    """
    try:
        if not isinstance(trace, list) or not isinstance(events, list):
            raise ValueError('Trace and accepted action ledger must be lists')
        for event in events:
            if not isinstance(event, dict) or event.get('accepted', True) is not True:
                raise ValueError('Only accepted real actions may enter the ledger')
            if event.get('action') not in ('measure', 'clear'):
                raise ValueError('Unknown accepted action')
            channel = event.get('channel')
            if type(channel) is not int or not 1 <= channel <= 20:
                raise ValueError('Invalid accepted action channel')
            position = event.get('position')
            if not isinstance(position, (list, tuple)) or len(position) != 2:
                raise ValueError('Invalid accepted action position')
            if not all(math.isfinite(float(v)) for v in position):
                raise ValueError('Nonfinite accepted action position')
            kind = event.get('measure_result' if event['action'] == 'measure' else 'clear_result')
            allowed = ('no_signal', 'direction', 'near') if event['action'] == 'measure' else ('success', 'no_target_in_range')
            if kind not in allowed:
                raise ValueError('Invalid accepted action feedback')
        for row in trace:
            if not isinstance(row, dict):
                raise ValueError('Invalid trace record')
            if row.get('phase') == 'actual_action' and type(row.get('event')) is not int:
                raise ValueError('Invalid real-action trace index')
        verified = _verify_complete(dict(trace=trace, events=events))
        proofs = [row for row in trace if row['phase'] == 'ordered_optical_prune']
        removed_planned = skipped_before_success = 0
        for proof, audit in zip(proofs, verified['rows']):
            removed_planned += len(proof['removed_indices'])
            actual_count = audit['executed_original_points']
            first_success_index = proof['kept_indices'][actual_count-1]
            skipped_before_success += sum(i < first_success_index for i in proof['removed_indices'])
        return dict(passed=True, plans=verified['plans'],
            removed_planned=removed_planned, skipped_before_success=skipped_before_success,
            accepted_actions=len(events), actual_optical_actions=verified['actual_optical_actions'],
            contractions_applied=sum(row['info']['applied'] for row in proofs),
            all_optical_plans_audited=True, all_optical_actions_accounted=True,
            feedback_bound_to_independent_accepted_ledger=True)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError('Malformed or inconsistent ordered-pruning proof: '+type(exc).__name__) from exc
