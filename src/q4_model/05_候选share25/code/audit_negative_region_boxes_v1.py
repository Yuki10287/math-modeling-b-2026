"""Independent exact verifier of negative-region-fraction-v1 certificates.

The verifier never calls the producer's hull, split, exclusion or interval
functions. Certificate premises must be supplied by the caller from the real
trace. Feedback provenance is checked against earlier same-channel events.
All arithmetic used to accept an exclusion is rational, including the final
containment check against the actual binary floating-point output polygon.
"""
import argparse
import copy
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
from pathlib import Path
import time


def demand(condition, message):
    if not condition:
        raise ValueError(message)


def rational_points(rows):
    result = []
    for row in rows:
        demand(len(row) == 2, 'point dimension')
        point = []
        for value in row:
            demand(math.isfinite(float(value)), 'nonfinite coordinate')
            point.append(F(float(value)))
        result.append(tuple(point))
    return tuple(result)


def turn(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def squared(a, b):
    return (a[0]-b[0])**2+(a[1]-b[1])**2


def convex_hull(points):
    """Exact gift wrapping; independent of the producer's monotone-chain hull."""
    points = sorted(set(points))
    if len(points) < 3:
        return tuple(points)
    start, output = points[0], []
    current = start
    while True:
        output.append(current)
        candidate = next(p for p in points if p != current)
        for point in points:
            orientation = turn(current, candidate, point)
            if orientation < 0 or orientation == 0 and squared(current, point) > squared(current, candidate):
                candidate = point
        current = candidate
        if current == start:
            break
        demand(len(output) <= len(points), 'invalid hull cycle')
    return tuple(output)


def polygon_area(poly):
    return abs(sum(a[0]*b[1]-a[1]*b[0] for a, b in zip(poly, poly[1:]+poly[:1])))/2


def encloses(poly, points):
    return len(poly) >= 3 and all(turn(a, b, p) >= 0
        for a, b in zip(poly, poly[1:]+poly[:1]) for p in points)


def descendants(root, branch):
    triangle = root
    for bit in branch:
        lengths = [squared(triangle[k], triangle[(k+1) % 3]) for k in range(3)]
        i = lengths.index(max(lengths))
        a, b, c = triangle[i], triangle[(i+1) % 3], triangle[(i+2) % 3]
        midpoint = ((a[0]+b[0])/2, (a[1]+b[1])/2)
        triangle = (a, midpoint, c) if bit == '0' else (midpoint, b, c)
    return triangle


def linear_interval_feasible(constraints):
    """Exact sign partition, not floating heading samples or producer bounds.

    A constraint consists of its values at w=0 and w=1, and strictness. All
    roots and interval midpoints exhaust possible sign patterns on [0,1].
    """
    critical = {F(0), F(1)}
    for left, right, strict in constraints:
        if left == right:
            if left < 0 or strict and left == 0:
                return False
        else:
            root = left/(left-right)
            if 0 <= root <= 1:
                critical.add(root)
    critical = sorted(critical)
    candidates = critical+[(a+b)/2 for a, b in zip(critical, critical[1:])]
    for weight in candidates:
        values = [(left*(1-weight)+right*weight, strict) for left, right, strict in constraints]
        if all(value > 0 if strict else value >= 0 for value, strict in values):
            return True
    return False


def local_transform(original):
    pairs = list(itertools.combinations(range(len(original)), 2))
    i, j = max(pairs, key=lambda pair: squared(original[pair[0]], original[pair[1]]))
    anchor = original[i]
    u = (original[j][0]-anchor[0], original[j][1]-anchor[1])
    scale = u[0]**2+u[1]**2
    def transform(p):
        x, y = p[0]-anchor[0], p[1]-anchor[1]
        return (x*u[0]+y*u[1])/scale, (-x*u[1]+y*u[0])/scale
    return transform


def all_quadrants_impossible(triangle, positives, selected, transform):
    corners = tuple(map(transform, triangle))
    bounds = [(min(p[k] for p in corners), max(p[k] for p in corners)) for k in (0, 1)]
    pos, neg = tuple(map(transform, positives)), tuple(map(transform, selected))
    for sx, sy in itertools.product((1, -1), repeat=2):
        # Each constraint gets the most favorable possible position in the
        # enclosing box. This only RELAXES joint position-heading feasibility.
        low_x, high_x = bounds[0] if sx > 0 else bounds[0][::-1]
        low_y, high_y = bounds[1] if sy > 0 else bounds[1][::-1]
        constraints = [(sy*(p[1]-low_y), sx*(p[0]-low_x), False) for p in pos]
        constraints += [(sy*(high_y-q[1]), sx*(high_x-q[0]), True) for q in neg]
        if linear_interval_feasible(constraints):
            return False
    return True


def verify_certificate(certificate, original_polygon, positives, negatives, output_polygon,
                       require_reduction=False):
    """Verify from externally supplied premises; raise on any failed obligation."""
    demand(isinstance(certificate, dict), 'missing certificate')
    demand(certificate.get('version') == 'negative-region-fraction-v1', 'certificate version')
    demand(certificate.get('split_rule') == 'exact_longest_edge_midpoint_first_max_index', 'split rule')
    original = convex_hull(rational_points(original_polygon))
    positive, negative = rational_points(positives), rational_points(negatives)
    output = rational_points(output_polygon)
    demand(len(original) >= 3 and positive and negative, 'invalid contraction premises')
    demand(rational_points(certificate['original_polygon']) == original, 'original region not bound to trace')
    demand(rational_points(certificate['positive_points']) == positive, 'positive evidence not bound to trace')
    demand(rational_points(certificate['negative_points']) == negative, 'negative evidence not bound to trace')
    demand(rational_points(certificate['output_polygon']) == output, 'output not bound to applied polygon')
    demand(output == convex_hull(output) and len(output) >= 3, 'output not a canonical convex polygon')
    depth = certificate.get('max_depth')
    demand(type(depth) is int and 0 <= depth <= 16, 'subdivision depth')
    roots = [(original[0], original[k], original[k+1]) for k in range(1, len(original)-1)]
    demand(certificate['roots'] == len(roots), 'wrong original fan root count')
    trees, leaves = [{} for _ in roots], []
    for excluded, rows in ((False, certificate['retained_leaves']), (True, certificate['excluded_leaves'])):
        demand(isinstance(rows, list), 'leaf list')
        for leaf in rows:
            root, branch = leaf.get('root'), leaf.get('branch')
            demand(type(root) is int and 0 <= root < len(roots), 'root index')
            demand(isinstance(branch, str) and len(branch) <= depth and set(branch) <= {'0', '1'}, 'branch')
            node = trees[root]
            for bit in branch:
                demand('leaf' not in node, 'overlapping ancestor leaf')
                node = node.setdefault(bit, {})
            demand(not node, 'duplicate leaf or overlapping descendant')
            node['leaf'] = True
            leaves.append((excluded, leaf, descendants(roots[root], branch)))
    def full(node):
        if 'leaf' in node:
            return set(node) == {'leaf'}
        return set(node) == {'0', '1'} and full(node['0']) and full(node['1'])
    demand(all(full(tree) for tree in trees), 'leaf partition has missing regions')
    demand(sum(polygon_area(t) for _, _, t in leaves) == polygon_area(original), 'partition area mismatch')
    transform, retained = local_transform(original), []
    reasons = {}
    for excluded, leaf, triangle in leaves:
        if not excluded:
            retained.extend(triangle)
            continue
        forced = leaf.get('forced')
        demand(isinstance(forced, list) and forced, 'missing forced-negative premises')
        selected, seen = [], set()
        for proof in forced:
            index = proof.get('negative')
            demand(type(index) is int and 0 <= index < len(negative) and index not in seen, 'forced negative index')
            seen.add(index)
            q = negative[index]
            if proof.get('kind') == 'minimum_radius':
                demand(all(squared(g, q) <= 1000**2 for g in triangle), 'false minimum-radius exclusion premise')
            elif proof.get('kind') == 'positive_radius_dominates':
                k = proof.get('positive')
                demand(type(k) is int and 0 <= k < len(positive), 'positive radius witness index')
                # The difference of squared distances is affine in g. Checking
                # its three vertex values proves it on the full closed cell.
                demand(all(squared(g, q) <= squared(g, positive[k]) for g in triangle),
                       'false positive-radius dominance premise')
            else:
                raise ValueError('unknown forced-negative proof kind')
            selected.append(q)
        reason = leaf.get('reason')
        if reason == 'cell_inside_forced_negative_hull':
            demand(encloses(convex_hull(selected), triangle), 'cell outside forced-negative convex hull')
        elif reason == 'all_heading_quadrants_infeasible':
            demand(all_quadrants_impossible(triangle, positive, selected, transform),
                   'a relaxed heading quadrant is feasible')
        else:
            raise ValueError('unknown cell exclusion reason')
        reasons[reason] = reasons.get(reason, 0)+1
    demand(retained, 'no retained region')
    demand(encloses(output, retained), 'actual float output cuts retained rational vertices')
    if require_reduction:
        demand(polygon_area(output) < polygon_area(original), 'applied result does not strictly reduce area')
    return dict(passed=True, roots=len(roots), leaves=len(leaves),
                excluded_leaves=len(certificate['excluded_leaves']),
                retained_leaves=len(certificate['retained_leaves']), exclusion_reasons=reasons,
                premises_bound_to_external_trace=True, original_partition_complete=True,
                actual_float_contains_all_retained_vertices=True)


def verify_case_contractions(case):
    events, trace = case['events'], case['trace']
    reports, latest_event = [], None
    for i, row in enumerate(trace):
        if row['phase'] == 'actual_action':
            latest_event = row.get('event')
        if row['phase'] != 'negative_contraction':
            continue
        channel, event = row['channel'], row['event']
        demand(type(event) is int and 1 <= event <= len(events) and event == latest_event,
               'contraction not attached to latest real action')
        latest = events[event-1]
        demand(latest['action'] == 'measure' and latest['channel'] == channel, 'wrong contraction event/channel')
        positive, negative = set(), set()
        for entry in events[:event]:
            if entry['action'] == 'measure' and entry['channel'] == channel:
                point = rational_points([entry['position']])[0]
                if entry['measure_result'] == 'no_signal':
                    negative.add(point)
                elif entry['measure_result'] in ('direction', 'near'):
                    positive.add(point)
        demand(set(rational_points(row['positives'])) <= positive, 'unobserved positive premise')
        demand(set(rational_points(row['negatives'])) <= negative, 'unobserved negative premise')
        demand(row['info']['applied'] is True, 'nonapplied contraction trace')
        demand(i+1 < len(trace) and trace[i+1]['phase'] == 'belief' and
               trace[i+1]['channel'] == channel and trace[i+1]['polygon'] == row['polygon'],
               'contraction output not present in following belief trace')
        report = verify_certificate(row['info']['certificate'], row['original_polygon'],
            row['positives'], row['negatives'], row['polygon'], require_reduction=True)
        report.update(channel=channel, event=event, feedback_provenance_verified=True)
        reports.append(report)
    return dict(passed=True, contractions=len(reports), rows=reports)


def self_tests():
    # Imported only to produce test certificates, never as verification logic.
    import negative_region_boxes_v1 as producer
    results = []
    for name, constraints, expected in (
        ('closed_axis_zero', [(F(0), F(1), False), (F(0), F(-1), False)], True),
        ('strict_axis_zero_rejected', [(F(0), F(-1), True)], False),
        ('closed_axis_one', [(F(-1), F(0), False)], True),
        ('strict_axis_one_rejected', [(F(-1), F(0), True)], False),
        ('closed_singleton_half', [(F(-1), F(1), False), (F(1), F(-1), False)], True),
        ('strict_singleton_rejected', [(F(-1), F(1), True), (F(1), F(-1), False)], False),
        ('constant_strict_zero_rejected', [(F(0), F(0), True)], False)):
        demand(linear_interval_feasible(constraints) == expected, name)
        results.append(dict(name=name, passed=True))
    P = [[-50., -50.], [50., -50.], [50., 50.], [-50., 50.]]
    positives, negatives = [[100., 0.], [-100., 0.]], [[0., -100.], [50., -100.]]
    result = producer.contract_region(P, positives, negatives, max_depth=3, max_nodes=255)
    certificate = result['certificate']
    verify_certificate(certificate, P, positives, negatives, result['polygon'])
    demand(encloses(rational_points(result['polygon']), [(F(0), F(0))]), 'closed heading-axis truth was excluded')
    results.append(dict(name='closed_positive_axis_true_source_retained', passed=True))
    bounded = producer.contract_region(P, positives, negatives, max_depth=6, max_nodes=1)
    verify_certificate(bounded['certificate'], P, positives, negatives, bounded['polygon'])
    results.append(dict(name='budget_retained_shallow_leaves_cover_original', passed=True))

    def reject(name, changed, pos=positives, neg=negatives, polygon=None):
        try:
            verify_certificate(changed, P, pos, neg,
                               changed['output_polygon'] if polygon is None else polygon)
        except (ValueError, KeyError, TypeError):
            results.append(dict(name=name, passed=True, forgery_rejected=True))
        else:
            raise AssertionError(f'Accepted forged certificate: {name}')

    bad = copy.deepcopy(certificate); bad['retained_leaves'].pop()
    reject('missing_retained_leaf', bad)
    bad = copy.deepcopy(certificate); bad['retained_leaves'].append(copy.deepcopy(bad['retained_leaves'][0]))
    reject('duplicate_retained_leaf', bad)
    bad = copy.deepcopy(certificate)
    selected = next(r for r in bad['retained_leaves'] if r['branch'])
    selected['branch'] = selected['branch'][:-1]
    reject('overlapping_parent_leaf', bad)
    bad = copy.deepcopy(certificate); bad['original_polygon'][0][0] -= 1.
    reject('forged_original_polygon', bad)
    bad = copy.deepcopy(certificate); bad['negative_points'][0][0] += 1.
    reject('forged_negative_premise', bad)
    bad = copy.deepcopy(certificate)
    center = [sum(p[k] for p in bad['output_polygon'])/len(bad['output_polygon']) for k in (0, 1)]
    bad['output_polygon'] = [[center[k]+.999*(p[k]-center[k]) for k in (0, 1)] for p in bad['output_polygon']]
    reject('float_output_shrunk_through_rational_boundary', bad)

    original = convex_hull(rational_points(P))
    roots = [(original[0], original[k], original[k+1]) for k in range(1, len(original)-1)]
    leaf = next(r for r in certificate['retained_leaves']
                if encloses(descendants(roots[r['root']], r['branch']), [(F(0), F(0))]))
    for reason, name in (('all_heading_quadrants_infeasible', 'fake_axis_heading_exclusion'),
                         ('cell_inside_forced_negative_hull', 'fake_negative_hull_exclusion')):
        bad = copy.deepcopy(certificate)
        bad['retained_leaves'].remove(leaf)
        bad['excluded_leaves'].append(dict(leaf, reason=reason,
            forced=[dict(negative=k, kind='minimum_radius') for k in range(2)]))
        reject(name, bad)
    for kind in ('minimum_radius', 'positive_radius_dominates'):
        bad = copy.deepcopy(certificate)
        neg = negatives+[[10000., 10000.]]
        bad['negative_points'] = copy.deepcopy(neg)
        proof = dict(negative=2, kind=kind)
        if kind == 'positive_radius_dominates':
            proof['positive'] = 0
        bad['excluded_leaves'][0]['forced'] = [proof]
        reject(f'false_{kind}_premise_with_valid_index', bad, neg=neg)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    tests = self_tests()
    rows = []
    if args.batch:
        for path in sorted(args.batch.glob('*.json')):
            case = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(case, dict) or 'events' not in case or 'trace' not in case:
                continue
            report = verify_case_contractions(case)
            report.update(case_file=path.name, case_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            rows.append(report)
            print(f'{path.name}: {report["contractions"]} exact certificates verified', flush=True)
    report = dict(passed=True, self_tests=len(tests), tests=tests, cases=len(rows),
        contractions=sum(r['contractions'] for r in rows), rows=rows,
        auditor_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        producer_source_sha256=hashlib.sha256(Path(__file__).with_name('negative_region_boxes_v1.py').read_bytes()).hexdigest(),
        official_simulator_contacted=False, actual_binary_float_output_checked_exactly=True,
        runtime_s=time.perf_counter()-started)
    with (args.out/'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('tests','rows')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
