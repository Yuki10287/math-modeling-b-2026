"""Bounded exact negative-feedback region used by the independent prune client.

Core functions copied unchanged from the frozen local research implementation:
05_候选share25/code/negative_region_boxes_v1.py
SHA256: 020a8f9088c9710412753d3d027d5d44d3a996f1b6bb7cb8e67b023c1a3e2da5
Only imports and the offline diagnostic entry point are removed. The original
research file remains unchanged; no true source data enters this module.
"""
from collections import Counter
from fractions import Fraction as F
import time

import numpy as np


def cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def distance2(a, b):
    return sum((x-y)**2 for x, y in zip(a, b))


def exact_points(points):
    a = np.asarray(points, float)
    if a.size == 0:
        return ()
    if a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all():
        raise ValueError('Expected finite 2D coordinates')
    return tuple(tuple(F(float(x)) for x in row) for row in a)


def hull(points):
    points = sorted(set(points))
    if len(points) <= 1:
        return tuple(points)
    chains = []
    for ordered in (points, points[::-1]):
        chain = []
        for p in ordered:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], p) <= 0:
                chain.pop()
            chain.append(p)
        chains.append(chain)
    return tuple(chains[0][:-1]+chains[1][:-1])


def contains(poly, points):
    return len(poly) >= 3 and all(cross(a, b, p) >= 0 for a, b in zip(poly, poly[1:]+poly[:1]) for p in points)


def area(poly):
    return abs(sum(a[0]*b[1]-a[1]*b[0] for a, b in zip(poly, poly[1:]+poly[:1])))/2


def split(triangle):
    i = max(range(3), key=lambda k: distance2(triangle[k], triangle[(k+1)%3]))
    j, k = (i+1)%3, (i+2)%3
    mid = tuple((a+b)/2 for a, b in zip(triangle[i], triangle[j]))
    return (triangle[i], mid, triangle[k]), (mid, triangle[j], triangle[k])


def interval_nonempty(constraints):
    """Whether w in [0,1] satisfies A*w+B >=0 (or >0 if strict)."""
    lo, hi, lo_open, hi_open = F(0), F(1), False, False
    for A, B, strict in constraints:
        if A == 0:
            if B < 0 or B == 0 and strict:
                return False
            continue
        boundary = -B/A
        if A > 0:
            if boundary > lo:
                lo, lo_open = boundary, strict
            elif boundary == lo:
                lo_open |= strict
        else:
            if boundary < hi:
                hi, hi_open = boundary, strict
            elif boundary == hi:
                hi_open |= strict
        if lo > hi or lo == hi and (lo_open or hi_open):
            return False
    return True


def _outer_float_polygon(exact_hull, original):
    # Conversion to nearest binary float alone could shrink a boundary. Expand
    # then verify EVERY retained rational vertex against the actual float hull.
    arr = np.array(exact_hull, float)
    center = np.array([float(sum(p[k] for p in exact_hull)/len(exact_hull)) for k in range(2)])
    for epsilon in (1e-10, 1e-9, 1e-8, 1e-7):
        candidate = center+(arr-center)*(1+epsilon)
        exact_candidate = hull(exact_points(candidate))
        if contains(exact_candidate, exact_hull):
            return [[float(x) for x in p] for p in exact_candidate], epsilon, False
    assert contains(original, exact_hull)
    return [[float(x) for x in p] for p in original], 0., True


def contract_region(P, positives, negatives, max_depth=6, max_nodes=255):
    original, positive, negative = exact_points(P), exact_points(positives), exact_points(negatives)
    if not positive:
        raise ValueError('A known-source contraction requires a positive observation')
    if not isinstance(max_depth, int) or not 0 <= max_depth <= 16:
        raise ValueError('Invalid subdivision depth')
    if not isinstance(max_nodes, int) or max_nodes < 1:
        raise ValueError('Invalid proof-node budget')
    original = hull(original)
    if len(original) < 3 or area(original) < F(1, 10**12) or not negative:
        return dict(polygon=np.asarray(P, float).tolist(), changed=False, skipped=True,
            reason='degenerate_region_or_no_negative_feedback', certificate=None)
    i, j = max(((i,j) for i in range(len(original)) for j in range(i+1, len(original))),
               key=lambda pair: distance2(original[pair[0]], original[pair[1]]))
    origin = original[i]
    u = tuple(b-a for a, b in zip(original[i], original[j]))
    v, scale = (-u[1], u[0]), distance2(original[i], original[j])
    def local(p):
        d = tuple(x-y for x, y in zip(p, origin))
        return sum(x*y for x,y in zip(d,u))/scale, sum(x*y for x,y in zip(d,v))/scale
    local_positive, local_negative = tuple(map(local, positive)), tuple(map(local, negative))
    affine = [[(2*(p[0]-q[0]), 2*(p[1]-q[1]),
                q[0]**2+q[1]**2-p[0]**2-p[1]**2) for p in positive] for q in negative]
    hull_cache = {}
    def force(triangle):
        selected, proofs = [], []
        for index, q in enumerate(negative):
            if all(distance2(g, q) <= 1000**2 for g in triangle):
                selected.append(index)
                proofs.append(dict(negative=index, kind='minimum_radius'))
                continue
            for k, (A, B, C) in enumerate(affine[index]):
                if all(A*g[0]+B*g[1]+C <= 0 for g in triangle):
                    selected.append(index)
                    proofs.append(dict(negative=index, kind='positive_radius_dominates', positive=k))
                    break
        return tuple(selected), proofs
    def impossible(triangle):
        selected, proofs = force(triangle)
        if not selected:
            return None
        if selected not in hull_cache:
            hull_cache[selected] = hull(tuple(negative[k] for k in selected))
        negative_hull = hull_cache[selected]
        if contains(negative_hull, triangle):
            return dict(reason='cell_inside_forced_negative_hull', forced=proofs)
        coordinates = tuple(map(local, triangle))
        lo = tuple(min(p[k] for p in coordinates) for k in range(2))
        hi = tuple(max(p[k] for p in coordinates) for k in range(2))
        for sx, sy in ((1,1), (1,-1), (-1,1), (-1,-1)):
            lower = (lo[0] if sx > 0 else hi[0], lo[1] if sy > 0 else hi[1])
            upper = (hi[0] if sx > 0 else lo[0], hi[1] if sy > 0 else lo[1])
            constraints = []
            for p in local_positive:
                dx, dy = p[0]-lower[0], p[1]-lower[1]
                constraints.append((sx*dx-sy*dy, sy*dy, False))
            for index in selected:
                q = local_negative[index]
                dx, dy = upper[0]-q[0], upper[1]-q[1]
                constraints.append((sx*dx-sy*dy, sy*dy, True))
            if interval_nonempty(constraints):
                return None
        return dict(reason='all_heading_quadrants_infeasible', forced=proofs)
    roots = [(original[0], original[k], original[k+1]) for k in range(1, len(original)-1)]
    excluded, retained = [], []
    retained_vertices = []
    excluded_area = F(0)
    tested_nodes = 0
    budget_retained = 0
    stack = [(root, '', triangle) for root, triangle in enumerate(roots)]
    while stack:
        root, branch, triangle = stack.pop()
        if tested_nodes >= max_nodes:
            retained.append(dict(root=root, branch=branch))
            retained_vertices.extend(triangle)
            budget_retained += 1
            continue
        tested_nodes += 1
        proof = impossible(triangle)
        if proof is not None:
            excluded.append(dict(root=root, branch=branch, **proof))
            excluded_area += area(triangle)
        elif len(branch) < max_depth:
            first, second = split(triangle)
            stack.extend(((root, branch+'0', first), (root, branch+'1', second)))
        else:
            retained.append(dict(root=root, branch=branch))
            retained_vertices.extend(triangle)
    if not retained:
        # Do not make an empty region available to an online solver. A valid
        # known source must survive; such an event requests a data/model audit.
        raise ValueError('All known-source positions excluded; feedback/model conflict')
    kept_hull = hull(retained_vertices)
    if excluded:
        polygon, expansion, fallback = _outer_float_polygon(kept_hull, original)
    else:
        polygon, expansion, fallback = [[float(x) for x in p] for p in original], 0., False
    assert contains(hull(exact_points(polygon)), kept_hull)
    before_area, convex_area, output_area = area(original), area(kept_hull), area(exact_points(polygon))
    initial_span = max(local(p)[0] for p in original)-min(local(p)[0] for p in original)
    kept_span = max(local(p)[0] for p in kept_hull)-min(local(p)[0] for p in kept_hull)
    certificate = dict(version='negative-region-fraction-v1',
        original_polygon=[[float(x) for x in p] for p in original],
        positive_points=[[float(x) for x in p] for p in positive],
        negative_points=[[float(x) for x in p] for p in negative],
        split_rule='exact_longest_edge_midpoint_first_max_index', max_depth=max_depth,
        max_nodes=max_nodes, tested_nodes=tested_nodes,
        roots=len(roots), excluded_leaves=excluded, retained_leaves=retained,
        output_polygon=polygon, float_outer_expansion=expansion,
        original_polygon_fallback=fallback, heading_boundary_strictness_preserved=True)
    return dict(polygon=polygon, changed=bool(excluded) and output_area < before_area,
        skipped=False, certificate=certificate, statistics=dict(
            original_area_m2=float(before_area), exactly_excluded_area_m2=float(excluded_area),
            exactly_excluded_area_fraction=float(excluded_area/before_area),
            retained_union_area_m2=float(before_area-excluded_area),
            retained_convex_hull_area_m2=float(convex_area), output_area_m2=float(output_area),
            output_area_reduction_fraction=float(1-output_area/before_area),
            original_long_axis_span_m=float(initial_span)*float(scale)**.5,
            retained_long_axis_span_m=float(kept_span)*float(scale)**.5,
            excluded_leaves=len(excluded), retained_leaves=len(retained),
            tested_nodes=tested_nodes, budget_retained_leaves=budget_retained,
            exclusion_reasons=dict(Counter(p['reason'] for p in excluded))))


def contract_position(P, positives, negatives):
    """Bounded online adapter. Failure or no strict benefit preserves old P.

    Returns (new_P, info), where info always has applied/certificate/statistics.
    Only genuine no-signal positions belong in negatives; failed clears are
    deliberately outside this isolated experiment's model.
    """
    original = np.asarray(P, float).copy()
    started = time.perf_counter()
    info = dict(applied=False, certificate=None, statistics={}, max_depth=6, max_nodes=255)
    if len(negatives) < 2:
        info.update(reason='fewer_than_two_negative_measurements', runtime_s=time.perf_counter()-started)
        return original, info
    if len(original) and np.max(np.linalg.norm(original-original.mean(axis=0), axis=1)) <= 20-1e-6:
        info.update(reason='already_small_enough_for_single_clear', runtime_s=time.perf_counter()-started)
        return original, info
    try:
        result = contract_region(original, positives, negatives, max_depth=6, max_nodes=255)
        info['statistics'] = result.get('statistics', {})
        if result['changed']:
            info.update(applied=True, certificate=result['certificate'], reason='strict_verified_area_reduction',
                        runtime_s=time.perf_counter()-started)
            return np.asarray(result['polygon']), info
        info.update(reason=result.get('reason', 'no_strict_area_reduction'), runtime_s=time.perf_counter()-started)
    except Exception as exc:
        info.update(reason='proof_failed_original_region_preserved', error_type=type(exc).__name__,
                    error=str(exc), runtime_s=time.perf_counter()-started)
    return original, info
