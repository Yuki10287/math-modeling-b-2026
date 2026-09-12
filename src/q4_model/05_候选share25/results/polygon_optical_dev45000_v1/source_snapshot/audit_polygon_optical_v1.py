"""Independent exact optical audit using vertex/edge intersection enumeration.

Does not import the planning module or its clipping/hull predicates. All grid
cells are reconstructed, including closed boundary-only intersections. A
nonempty convex intersection has a vertex from either operand or an edge
intersection, so this enumerates its exact extreme-point candidates.
"""
from fractions import Fraction as F
import copy


def point(p):
    return tuple(F(float(v)) for v in p)


def determinant(a, b):
    return a[0]*b[1]-a[1]*b[0]


def subtract(a, b):
    return a[0]-b[0], a[1]-b[1]


def orientation(a, b, c):
    return determinant(subtract(b, a), subtract(c, a))


def convex_hull(points):
    values = sorted(set(points))
    if len(values) < 3:
        return values
    first = values[0]
    result, p = [], first
    while True:
        result.append(p)
        q = next(v for v in values if v != p)
        for r in values:
            cross = orientation(p, q, r)
            if cross < 0 or (cross == 0 and sum(x*x for x in subtract(r, p)) > sum(x*x for x in subtract(q, p))):
                q = r
        p = q
        if p == first:
            return result
        assert len(result) <= len(values)


def inside(poly, p):
    if len(poly) == 1:
        return p == poly[0]
    if len(poly) == 2:
        a, b = poly
        return orientation(a, b, p) == 0 and all(min(a[k], b[k]) <= p[k] <= max(a[k], b[k]) for k in range(2))
    signs = [orientation(a, b, p) for a, b in zip(poly, poly[1:]+poly[:1])]
    return all(v >= 0 for v in signs) or all(v <= 0 for v in signs)


def intersection_vertices(poly, bounds):
    (a, b), (c, d) = bounds
    corners = list(dict.fromkeys(((a, c), (b, c), (b, d), (a, d))))
    rectangle = convex_hull(corners)
    candidates = {p for p in poly if inside(rectangle, p)}
    candidates.update(p for p in rectangle if inside(poly, p))
    for p, q in zip(poly, poly[1:]+poly[:1]):
        edge = subtract(q, p)
        for r, s in zip(rectangle, rectangle[1:]+rectangle[:1]):
            other = subtract(s, r)
            denominator = determinant(edge, other)
            if denominator == 0:
                # Endpoints of collinear overlap were included above.
                continue
            t = determinant(subtract(r, p), other)/denominator
            u = determinant(subtract(r, p), edge)/denominator
            if 0 <= t <= 1 and 0 <= u <= 1:
                candidates.add(tuple(p[k]+t*edge[k] for k in range(2)))
    return candidates


def audit_certificate(certificate, polygon, path):
    assert certificate['version'] == 'convex-grid-optical-v1'
    assert certificate['partition'] == 'closed_uniform_grid_in_exact_orthogonal_basis'
    assert certificate['radius_m'] == 20
    assert certificate['original_polygon'] == polygon, 'original region not externally bound'
    assert certificate['actual_points'] == path, 'actual execution path does not match proof'
    vertices = [point(p) for p in polygon]
    assert vertices
    i, j = certificate['basis_indices']
    assert type(i) is int and type(j) is int and 0 <= i < len(vertices) and 0 <= j < len(vertices)
    base = vertices[i]
    horizontal = subtract(vertices[j], base)
    if horizontal == (0, 0):
        horizontal = (F(1), F(0))
    vertical = (-horizontal[1], horizontal[0])
    denominator = sum(v*v for v in horizontal)
    def local(p):
        delta = subtract(p, base)
        return tuple(sum(delta[k]*axis[k] for k in range(2))/denominator for axis in (horizontal, vertical))
    def world(p):
        return tuple(base[k]+p[0]*horizontal[k]+p[1]*vertical[k] for k in range(2))
    poly = convex_hull([local(p) for p in vertices])
    lower = tuple(min(p[k] for p in poly) for k in range(2))
    upper = tuple(max(p[k] for p in poly) for k in range(2))
    nx, ny = certificate['grid_counts']
    assert type(nx) is int and type(ny) is int and 1 <= nx <= 10000 and 1 <= ny <= 10000
    ids = certificate['occupied_cells']
    assert len(ids) == len(path)
    assert all(len(c) == 2 and all(type(k) is int for k in c) for c in ids)
    assert len({tuple(c) for c in ids}) == len(ids), 'repeated grid cell'
    claimed = {tuple(cell): point(q) for cell, q in zip(ids, path)}
    expected = set()
    checked_vertices = 0
    largest_radius2 = F(0)
    for x in range(nx):
        for y in range(ny):
            bounds = [(lower[k]+(upper[k]-lower[k])*index/count,
                       lower[k]+(upper[k]-lower[k])*(index+1)/count)
                      for k, (index, count) in enumerate(((x, nx), (y, ny)))]
            vertices_here = intersection_vertices(poly, bounds)
            if not vertices_here:
                continue
            expected.add((x, y))
            assert (x, y) in claimed, 'nonempty grid intersection omitted'
            q = claimed[(x, y)]
            for p in vertices_here:
                distance2 = sum((a-b)**2 for a, b in zip(world(p), q))
                assert distance2 <= 400, 'actual float clearing point exceeds radius 20'
                largest_radius2 = max(largest_radius2, distance2)
                checked_vertices += 1
    assert set(claimed) == expected, 'invalid or genuinely empty cell claimed'
    return dict(passed=True, occupied_cells=len(expected), rectangle_cells=nx*ny,
                omitted_cells=nx*ny-len(expected), exact_vertices_checked=checked_vertices,
                max_squared_radius_m2=float(largest_radius2))


def audit_run(events, trace):
    latest = {}
    checks = []
    for row in trace:
        if row['phase'] == 'belief':
            latest[row['channel']] = row['polygon']
        if row['phase'] != 'polygon_optical_plan':
            continue
        c, index = row['channel'], row['event']
        assert row['polygon'] == latest[c], 'planner changed region before clearance'
        assert row['start'] == (events[index-1]['position'] if index else [0., 0.])
        audit = audit_certificate(row['optical_certificate'], row['polygon'], row['path'])
        succeeded = False
        for k, q in enumerate(row['path']):
            assert index+k < len(events), 'optical execution abandoned before clearing'
            event = events[index+k]
            assert event['action'] == 'clear' and event['channel'] == c
            assert event['position'] == q, 'actual clearance path differs from certified prefix'
            if event['clear_result'] == 'success':
                succeeded = True
                break
            assert event['clear_result'] == 'no_target_in_range'
        assert succeeded, 'complete optical cover did not clear source'
        audit.update(channel=c, executed_points=k+1)
        checks.append(audit)
    return dict(passed=True, optical_plans=len(checks), plans=checks,
                independent_geometry='exact vertex/edge intersections; no planner predicates imported')
