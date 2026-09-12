"""Independent exact audit of preserved-order doomed-clearance deletion.

No import from ordered_optical_prune_experiment. Segment distances use signed
projection classification and cross-product squares, rather than reconstructing
the closest point. Original rectangle coverage is certified in the exact inverse
of its actual binary-float transform, without assuming float orthogonality.
"""
import argparse
import copy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

from audit_negative_region_boxes_v1 import verify_certificate


def exact(p):
    return tuple(F(float(x)) for x in p)


def minus(a, b):
    return a[0]-b[0], a[1]-b[1]


def dot(a, b):
    return a[0]*b[0]+a[1]*b[1]


def cross(a, b):
    return a[0]*b[1]-a[1]*b[0]


def hull(points):
    points = sorted(set(map(exact, points)))
    assert points
    if len(points) < 3:
        return points
    first, current, result = points[0], points[0], []
    while True:
        result.append(current)
        candidate = next(p for p in points if p != current)
        for p in points:
            a, b = minus(candidate, current), minus(p, current)
            sign = cross(a, b)
            if sign < 0 or (sign == 0 and dot(b, b) > dot(a, a)):
                candidate = p
        current = candidate
        if current == first:
            return result
        assert len(result) <= len(points)


def independent_distance2(query, polygon):
    q, vertices = exact(query), hull(polygon)
    if len(vertices) == 1:
        v = minus(q, vertices[0])
        return dot(v, v)
    edges = list(zip(vertices, vertices[1:]+vertices[:1]))
    if len(vertices) > 2 and all(cross(minus(b, a), minus(q, a)) >= 0 for a, b in edges):
        return F(0)
    bounds = []
    for a, b in edges:
        v, w = minus(b, a), minus(q, a)
        length2, projection = dot(v, v), dot(v, w)
        if projection <= 0:
            bounds.append(dot(w, w))
        elif projection >= length2:
            end = minus(q, b)
            bounds.append(dot(end, end))
        else:
            bounds.append(cross(v, w)**2/length2)
    return min(bounds)


def verify_old_rectangle(plan):
    P, path = [exact(p) for p in plan['polygon']], [exact(p) for p in plan['path']]
    origin = exact(plan['origin'])
    R = [[F(float(v)) for v in row] for row in plan['rotation']]
    determinant = R[0][0]*R[1][1]-R[0][1]*R[1][0]
    assert determinant != 0, 'singular old rectangle transform'
    def local(p):
        x, y = minus(p, origin)
        return (R[1][1]*x-R[0][1]*y)/determinant, (-R[1][0]*x+R[0][0]*y)/determinant
    def world(p):
        return tuple(origin[k]+R[k][0]*p[0]+R[k][1]*p[1] for k in range(2))
    lo, hi = exact(plan['lower']), exact(plan['upper'])
    nx, ny = plan['cells']
    assert type(nx) is int and type(ny) is int and nx > 0 and ny > 0
    assert len(path) == nx*ny
    assert all(lo[k] <= hi[k] for k in range(2))
    local_P = list(map(local, P))
    lower = [min(lo[k], min(p[k] for p in local_P)) for k in range(2)]
    upper = [max(hi[k], max(p[k] for p in local_P)) for k in range(2)]
    ownership = {}
    for q in path:
        p = local(q)
        cell = tuple(0 if hi[k] == lo[k] else int(round((p[k]-lo[k])*count/(hi[k]-lo[k])-F(1, 2)))
                     for k, count in enumerate((nx, ny)))
        assert 0 <= cell[0] < nx and 0 <= cell[1] < ny
        assert cell not in ownership, 'multiple points assigned same old rectangle cell'
        ownership[cell] = q
    largest = F(0)
    for x in range(nx):
        for y in range(ny):
            bounds = []
            for k, (index, count) in enumerate(((x, nx), (y, ny))):
                low = lower[k] if index == 0 else lo[k]+(hi[k]-lo[k])*index/count
                high = upper[k] if index == count-1 else lo[k]+(hi[k]-lo[k])*(index+1)/count
                bounds.append((low, high))
            q = ownership[(x, y)]
            for a in bounds[0]:
                for b in bounds[1]:
                    delta = minus(world((a, b)), q)
                    distance = dot(delta, delta)
                    assert distance <= 400, 'old continuous optical cover exceeds radius 20'
                    largest = max(largest, distance)
    return dict(passed=True, cells=nx*ny, max_corner_distance_squared=float(largest),
        method='exact inverse transform; outermost bounds include every original vertex; all cell corners checked')


def verify_pruning(proof, plan, belief, observed_positive, observed_negative):
    assert proof['original_polygon'] == belief['polygon'] == plan['polygon']
    assert proof['positives'] == belief['positives'] and proof['negatives'] == belief['negatives']
    assert all(tuple(p) in observed_positive for p in proof['positives']), 'unobserved positive premise'
    assert all(tuple(p) in observed_negative for p in proof['negatives']), 'unobserved negative premise'
    assert proof['original_path'] == plan['path'], 'original path modified'
    assert type(proof['info']['applied']) is bool
    if proof['info']['applied']:
        verify_certificate(proof['info']['certificate'], proof['original_polygon'], proof['positives'],
                           proof['negatives'], proof['polygon'], require_reduction=True)
    else:
        assert proof['polygon'] == proof['original_polygon'], 'unproved contraction'
    old_cover = verify_old_rectangle(plan)
    d0 = [independent_distance2(q, proof['original_polygon']) for q in plan['path']]
    d1 = [independent_distance2(q, proof['polygon']) for q in plan['path']]
    assert [F(s) for s in proof['original_distances_squared']] == d0, 'forged original distance'
    assert [F(s) for s in proof['contracted_distances_squared']] == d1, 'forged contracted distance'
    r0 = [i for i, d in enumerate(d0) if d > 400]
    r1 = [i for i, d in enumerate(d1) if d > 400]
    removed = sorted(set(r0) | set(r1))
    kept = [i for i in range(len(d0)) if i not in removed]
    assert all(type(i) is int for key in ('original_removed_indices', 'contracted_removed_indices', 'removed_indices', 'kept_indices') for i in proof[key])
    assert proof['original_removed_indices'] == r0 and proof['contracted_removed_indices'] == r1
    assert proof['removed_indices'] == removed, 'not exactly justified deletion union'
    assert proof['kept_indices'] == kept and kept, 'retained points reordered, moved or empty'
    return dict(passed=True, original_cover=old_cover, removed=len(removed), kept=len(kept),
        from_original=len(r0), additional_from_contraction=len(set(r1)-set(r0)))


def verify_case(case, replay=None):
    trace, events = case['trace'], case['events']
    beliefs, positives, negatives = {}, {}, {}
    latest, checks, removed_events = 0, [], set()
    replay_by_event = {p['event']: p for p in replay['proofs']} if replay else {}
    assert not replay or len(replay_by_event) == len(replay['proofs']), 'duplicate replay plan event'
    used_replay = set()
    for index, row in enumerate(trace):
        if row['phase'] == 'actual_action':
            latest += 1
            assert row['event'] == latest
            actual = events[latest-1]
            assert row['action'] == actual['action'] and row['channel'] == actual['channel'] and row['position'] == actual['position']
            assert row['result'] == actual['measure_result' if actual['action'] == 'measure' else 'clear_result']
            if actual['action'] == 'measure':
                ledger = negatives if actual['measure_result'] == 'no_signal' else positives
                ledger.setdefault(actual['channel'], set()).add(tuple(actual['position']))
        if row['phase'] == 'belief':
            beliefs[row['channel']] = row
        if replay and row['phase'] == 'optical_plan':
            assert latest in replay_by_event, 'original fallback absent from replay proof'
            proof, plan = replay_by_event[latest], row
            used_replay.add(latest)
            actual_start = index+1
        elif not replay and row['phase'] == 'ordered_optical_prune':
            proof, plan = row, trace[index+1]
            assert plan['phase'] == 'optical_plan'
            actual_start = index+2
        else:
            continue
        channel = proof['channel']
        assert proof['event'] == latest and plan['channel'] == channel
        audit = verify_pruning(proof, plan, beliefs[channel], positives.get(channel, set()), negatives.get(channel, set()))
        expected_ids = list(range(len(plan['path']))) if replay else proof['kept_indices']
        success = False
        count = 0
        for action in trace[actual_start:]:
            assert action['phase'] == 'actual_action' and action['action'] == 'clear'
            assert action['channel'] == channel and action['reason'] == 'optical_cover'
            old_index = expected_ids[count]
            assert action['position'] == plan['path'][old_index]
            event = events[action['event']-1]
            assert event['position'] == action['position'] and event['channel'] == channel and event['action'] == 'clear'
            if replay and old_index in proof['removed_indices']:
                assert event['clear_result'] == 'no_target_in_range', 'proved deletion would remove success'
                removed_events.add(action['event'])
            count += 1
            if event['clear_result'] == 'success':
                success = True
                assert old_index in proof['kept_indices'], 'first success is not retained'
                break
        assert success
        audit.update(channel=channel, executed_original_points=count)
        checks.append(audit)
    assert latest == len(events)
    if replay:
        assert used_replay == set(replay_by_event)
        assert sorted(removed_events) == replay['removed_event_numbers'], 'replay removed unjustified events'
        expected = [{k: v for k, v in e.items() if k != 'time_s'} for i, e in enumerate(events, 1) if i not in removed_events]
        actual = [{k: v for k, v in e.items() if k != 'time_s'} for e in replay['events']]
        assert expected == actual, 'replay changed a retained action or feedback'
    return dict(passed=True, plans=len(checks), rows=checks, removed_events=len(removed_events))


def self_tests():
    cases = [([20, 0], [[0, 0]], F(400)),
             ([20+2**-35, 0], [[0, 0]], F(20+2**-35)**2),
             ([0, 20], [[-10, 0], [10, 0]], F(400)),
             ([13, 4], [[-10, 0], [10, 0]], F(25)),
             ([0, 0], [[-1, -1], [1, -1], [1, 1], [-1, 1]], F(0)),
             ([21, 0], [[-1, -1], [1, -1], [1, 1], [-1, 1]], F(400)),
             ([4, 5], [[0, 0], [3, 4]], F(2))]
    for q, P, expected in cases:
        assert independent_distance2(q, P) == expected
    return dict(passed=True, exact_distance_boundary_cases=len(cases), equality_20_not_deletable=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--replay-batch', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    snapshot = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cases = []
    if args.replay_batch:
        for path in sorted(args.batch.glob('*-share25.json')):
            replay_path = args.replay_batch/(path.stem+'-replay.json')
            case = json.loads(path.read_text(encoding='utf-8'))
            replay = json.loads(replay_path.read_text(encoding='utf-8'))
            assert replay['input_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
            audit = verify_case(case, replay)
            cases.append(dict(file=path.name, audit=audit))
    else:
        for path in sorted(args.batch.glob('*-ordered_prune.json')):
            case = json.loads(path.read_text(encoding='utf-8'))
            cases.append(dict(file=path.name, audit=verify_case(case)))
    assert cases
    result = dict(passed=True, self_tests=self_tests(), cases=cases,
        total_plans=sum(c['audit']['plans'] for c in cases),
        auditor_sha256=snapshot, local_only=True, official_simulator_contacted=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    assert snapshot == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps(dict(passed=True, cases=len(cases), total_plans=result['total_plans'],
                         self_tests=result['self_tests']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
