"""Independent exact negative-feedback proof checker, extracted from frozen audit_negative_region_boxes_v1.py. No planning or research-module imports."""

from fractions import Fraction as F
import itertools
import math

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
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

def squared(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def convex_hull(points):
    """Exact gift wrapping; independent of the producer's monotone-chain hull."""
    points = sorted(set(points))
    if len(points) < 3:
        return tuple(points)
    start, output = (points[0], [])
    current = start
    while True:
        output.append(current)
        candidate = next((p for p in points if p != current))
        for point in points:
            orientation = turn(current, candidate, point)
            if orientation < 0 or (orientation == 0 and squared(current, point) > squared(current, candidate)):
                candidate = point
        current = candidate
        if current == start:
            break
        demand(len(output) <= len(points), 'invalid hull cycle')
    return tuple(output)

def polygon_area(poly):
    return abs(sum((a[0] * b[1] - a[1] * b[0] for a, b in zip(poly, poly[1:] + poly[:1])))) / 2

def encloses(poly, points):
    return len(poly) >= 3 and all((turn(a, b, p) >= 0 for a, b in zip(poly, poly[1:] + poly[:1]) for p in points))

def descendants(root, branch):
    triangle = root
    for bit in branch:
        lengths = [squared(triangle[k], triangle[(k + 1) % 3]) for k in range(3)]
        i = lengths.index(max(lengths))
        a, b, c = (triangle[i], triangle[(i + 1) % 3], triangle[(i + 2) % 3])
        midpoint = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
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
            if left < 0 or (strict and left == 0):
                return False
        else:
            root = left / (left - right)
            if 0 <= root <= 1:
                critical.add(root)
    critical = sorted(critical)
    candidates = critical + [(a + b) / 2 for a, b in zip(critical, critical[1:])]
    for weight in candidates:
        values = [(left * (1 - weight) + right * weight, strict) for left, right, strict in constraints]
        if all((value > 0 if strict else value >= 0 for value, strict in values)):
            return True
    return False

def local_transform(original):
    pairs = list(itertools.combinations(range(len(original)), 2))
    i, j = max(pairs, key=lambda pair: squared(original[pair[0]], original[pair[1]]))
    anchor = original[i]
    u = (original[j][0] - anchor[0], original[j][1] - anchor[1])
    scale = u[0] ** 2 + u[1] ** 2

    def transform(p):
        x, y = (p[0] - anchor[0], p[1] - anchor[1])
        return ((x * u[0] + y * u[1]) / scale, (-x * u[1] + y * u[0]) / scale)
    return transform

def all_quadrants_impossible(triangle, positives, selected, transform):
    corners = tuple(map(transform, triangle))
    bounds = [(min((p[k] for p in corners)), max((p[k] for p in corners))) for k in (0, 1)]
    pos, neg = (tuple(map(transform, positives)), tuple(map(transform, selected)))
    for sx, sy in itertools.product((1, -1), repeat=2):
        low_x, high_x = bounds[0] if sx > 0 else bounds[0][::-1]
        low_y, high_y = bounds[1] if sy > 0 else bounds[1][::-1]
        constraints = [(sy * (p[1] - low_y), sx * (p[0] - low_x), False) for p in pos]
        constraints += [(sy * (high_y - q[1]), sx * (high_x - q[0]), True) for q in neg]
        if linear_interval_feasible(constraints):
            return False
    return True

def verify_certificate(certificate, original_polygon, positives, negatives, output_polygon, require_reduction=False):
    """Verify from externally supplied premises; raise on any failed obligation."""
    demand(isinstance(certificate, dict), 'missing certificate')
    demand(certificate.get('version') == 'negative-region-fraction-v1', 'certificate version')
    demand(certificate.get('split_rule') == 'exact_longest_edge_midpoint_first_max_index', 'split rule')
    original = convex_hull(rational_points(original_polygon))
    positive, negative = (rational_points(positives), rational_points(negatives))
    output = rational_points(output_polygon)
    demand(len(original) >= 3 and positive and negative, 'invalid contraction premises')
    demand(rational_points(certificate['original_polygon']) == original, 'original region not bound to trace')
    demand(rational_points(certificate['positive_points']) == positive, 'positive evidence not bound to trace')
    demand(rational_points(certificate['negative_points']) == negative, 'negative evidence not bound to trace')
    demand(rational_points(certificate['output_polygon']) == output, 'output not bound to applied polygon')
    demand(output == convex_hull(output) and len(output) >= 3, 'output not a canonical convex polygon')
    depth = certificate.get('max_depth')
    demand(type(depth) is int and 0 <= depth <= 16, 'subdivision depth')
    roots = [(original[0], original[k], original[k + 1]) for k in range(1, len(original) - 1)]
    demand(certificate['roots'] == len(roots), 'wrong original fan root count')
    trees, leaves = ([{} for _ in roots], [])
    for excluded, rows in ((False, certificate['retained_leaves']), (True, certificate['excluded_leaves'])):
        demand(isinstance(rows, list), 'leaf list')
        for leaf in rows:
            root, branch = (leaf.get('root'), leaf.get('branch'))
            demand(type(root) is int and 0 <= root < len(roots), 'root index')
            demand(isinstance(branch, str) and len(branch) <= depth and (set(branch) <= {'0', '1'}), 'branch')
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
    demand(all((full(tree) for tree in trees)), 'leaf partition has missing regions')
    demand(sum((polygon_area(t) for _, _, t in leaves)) == polygon_area(original), 'partition area mismatch')
    transform, retained = (local_transform(original), [])
    reasons = {}
    for excluded, leaf, triangle in leaves:
        if not excluded:
            retained.extend(triangle)
            continue
        forced = leaf.get('forced')
        demand(isinstance(forced, list) and forced, 'missing forced-negative premises')
        selected, seen = ([], set())
        for proof in forced:
            index = proof.get('negative')
            demand(type(index) is int and 0 <= index < len(negative) and (index not in seen), 'forced negative index')
            seen.add(index)
            q = negative[index]
            if proof.get('kind') == 'minimum_radius':
                demand(all((squared(g, q) <= 1000 ** 2 for g in triangle)), 'false minimum-radius exclusion premise')
            elif proof.get('kind') == 'positive_radius_dominates':
                k = proof.get('positive')
                demand(type(k) is int and 0 <= k < len(positive), 'positive radius witness index')
                demand(all((squared(g, q) <= squared(g, positive[k]) for g in triangle)), 'false positive-radius dominance premise')
            else:
                raise ValueError('unknown forced-negative proof kind')
            selected.append(q)
        reason = leaf.get('reason')
        if reason == 'cell_inside_forced_negative_hull':
            demand(encloses(convex_hull(selected), triangle), 'cell outside forced-negative convex hull')
        elif reason == 'all_heading_quadrants_infeasible':
            demand(all_quadrants_impossible(triangle, positive, selected, transform), 'a relaxed heading quadrant is feasible')
        else:
            raise ValueError('unknown cell exclusion reason')
        reasons[reason] = reasons.get(reason, 0) + 1
    demand(retained, 'no retained region')
    demand(encloses(output, retained), 'actual float output cuts retained rational vertices')
    if require_reduction:
        demand(polygon_area(output) < polygon_area(original), 'applied result does not strictly reduce area')
    return dict(passed=True, roots=len(roots), leaves=len(leaves), excluded_leaves=len(certificate['excluded_leaves']), retained_leaves=len(certificate['retained_leaves']), exclusion_reasons=reasons, premises_bound_to_external_trace=True, original_partition_complete=True, actual_float_contains_all_retained_vertices=True)
