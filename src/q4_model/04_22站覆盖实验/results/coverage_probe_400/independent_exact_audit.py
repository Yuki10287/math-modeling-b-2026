"""Independent exact geometry audit of a saved finite station certificate.

No simulator, solver, original certificate generator, or network is imported.
Station coordinates are interpreted as the exact rational values of their
parsed IEEE-754 doubles. Exact midpoint descendants replace rounded JSON leaf
vertices. NumPy is used only to reproduce the proposed subdivision lineage;
all containment, area, overlap, and range predicates use Fraction arithmetic.
"""
import argparse
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def area2(poly):
    return sum((a[0]*b[1]-a[1]*b[0]
                for a, b in zip(poly, poly[1:]+poly[:1])), F(0))


def hull(points):
    points = sorted(set(points))
    chains = []
    for order in (points, points[::-1]):
        chain = []
        for p in order:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], p) <= 0:
                chain.pop()
            chain.append(p)
        chains.append(chain)
    return chains[0][:-1]+chains[1][:-1]


def clip(poly, a, b):
    out = []
    for p, r in zip(poly, poly[1:]+poly[:1]):
        cp, cr = cross(a, b, p), cross(a, b, r)
        if cp >= 0:
            out.append(p)
        if (cp < 0 < cr) or (cr < 0 < cp):
            t = cp/(cp-cr)
            out.append(tuple(x+t*(y-x) for x, y in zip(p, r)))
    return out


def key(triangle):
    # Used to associate JSON labels only, never as a geometry predicate.
    return tuple(sorted(tuple(round(float(x), 8) for x in v) for v in triangle))


def audit(path):
    d = json.loads(path.read_text(encoding='utf-8'))
    q = [tuple(F(float(x)) for x in v) for v in d['stations']]
    outer = hull(q)
    assert len(q) == 22 and len(outer) == 12
    assert set(outer) == {q[i] for i in range(12)}
    original = [[q[i] for i in indices] for indices in d['initial_triangles']]
    assert len(original) == 30
    triangles = [t if area2(t) > 0 else t[::-1] for t in original]
    assert all(area2(t) > 0 for t in triangles)
    assert all(cross(a, b, v) >= 0 for t in triangles for v in t
               for a, b in zip(outer, outer[1:]+outer[:1]))
    edges = Counter(tuple(sorted((a, b))) for t in d['initial_triangles']
                    for a, b in zip(t, t[1:]+t[:1]))
    assert all(n in (1, 2) for n in edges.values())
    assert {e for e, n in edges.items() if n == 1} == {
        tuple(sorted((i, (i+1) % 12))) for i in range(12)}
    pair_count = 0
    for i, first in enumerate(triangles):
        for second in triangles[i+1:]:
            intersection = first
            for a, b in zip(second, second[1:]+second[:1]):
                intersection = clip(intersection, a, b)
            assert area2(intersection) == 0
            pair_count += 1
    assert sum(area2(t) for t in triangles) == area2(outer)
    # Closed triangles inside the polygon, disjoint in area, with equal total
    # area cover the polygon. The explicit boundary incidence is also checked.
    apothems = []
    for a, b in zip(outer, outer[1:]+outer[:1]):
        signed_numerator = a[0]*b[1]-a[1]*b[0]
        edge_squared = sum((x-y)**2 for x, y in zip(a, b))
        assert signed_numerator > 0
        assert signed_numerator**2 >= 1800**2*edge_squared
        apothems.append(float(signed_numerator)/math.sqrt(float(edge_squared)))

    lookup = {key(c['vertices']): i for i, c in enumerate(d['cells'])}
    assert len(lookup) == len(d['cells']) == 108
    seen, leaves = set(), {}
    maximum_saved_difference = 0.
    stack = [(t, np.array([[float(x) for x in v] for v in t]), 0)
             for t in original]
    while stack:
        exact, approximate, depth = stack.pop()
        label = lookup.get(key(approximate))
        if label is not None:
            assert label not in seen and depth == d['cells'][label]['depth']
            seen.add(label)
            leaves[label] = exact
            for v in exact:
                saved = min(d['cells'][label]['vertices'],
                            key=lambda z: math.dist([float(x) for x in v], z))
                difference = max(abs(float(F(float(x))-y)) for x, y in zip(saved, v))
                assert difference <= 1e-10
                maximum_saved_difference = max(maximum_saved_difference, difference)
            continue
        assert depth < 12, 'Bounded subdivision audit could not match a leaf'
        i = int(np.argmax(np.linalg.norm(np.roll(approximate, -1, axis=0)-approximate, axis=1)))
        j, k = (i+1) % 3, (i+2) % 3
        midpoint = tuple((x+y)/2 for x, y in zip(exact[i], exact[j]))
        approximate_midpoint = (approximate[i]+approximate[j])/2
        stack.extend([
            ([exact[i], midpoint, exact[k]],
             np.array([approximate[i], approximate_midpoint, approximate[k]]), depth+1),
            ([midpoint, exact[j], exact[k]],
             np.array([approximate_midpoint, approximate[j], approximate[k]]), depth+1)])
    assert seen == set(range(len(d['cells'])))
    assert sum(abs(area2(t)) for t in leaves.values()) == area2(outer)
    # Each exact midpoint split partitions its parent, so the leaf partition
    # inherits the initial triangles' full coverage and non-overlap.
    maximum_distance_squared = F(0)
    zero_hull_predicates = 0
    hull_predicates = 0
    distance_predicates = 0
    for label, triangle in leaves.items():
        witnesses = [q[i] for i in d['cells'][label]['witnesses']]
        witness_hull = hull(witnesses)
        assert len(witness_hull) >= 3
        for vertex in triangle:
            for station in witnesses:
                squared = sum((x-y)**2 for x, y in zip(vertex, station))
                assert squared <= 1000**2
                maximum_distance_squared = max(maximum_distance_squared, squared)
                distance_predicates += 1
            for a, b in zip(witness_hull, witness_hull[1:]+witness_hull[:1]):
                signed = cross(a, b, vertex)
                assert signed >= 0
                zero_hull_predicates += int(signed == 0)
                hull_predicates += 1
    maximum_distance = math.sqrt(float(maximum_distance_squared))
    return dict(
        passed=True, input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        arithmetic='Exact Fraction predicates on parsed binary-double station coordinates and exact midpoint descendants; displayed decimal margins are approximations.',
        station_count=len(q), initial_triangles=len(triangles),
        initial_unique_edges=len(edges), boundary_edges=sum(n == 1 for n in edges.values()),
        exact_pairwise_intersections_checked=pair_count, positive_area_overlaps=0,
        initial_area_equals_outer_polygon_exactly=True,
        matched_exact_leaf_cells=len(leaves), exact_bisection_partition_verified=True,
        leaf_area_equals_outer_polygon_exactly=True,
        outer_polygon_area_m2=float(area2(outer)/2),
        target_disk_radius_1800_contained_exactly=True,
        minimum_outer_apothem_m=min(apothems),
        outer_disk_margin_m=min(apothems)-1800,
        max_saved_leaf_coordinate_vs_exact_midpoint_difference_m=maximum_saved_difference,
        exact_distance_squared_checks=distance_predicates,
        maximum_witness_distance_m=maximum_distance,
        minimum_range_margin_m=1000-maximum_distance,
        exact_hull_halfplane_checks=hull_predicates,
        negative_hull_halfplane_checks=0, zero_hull_halfplane_checks=zero_hull_predicates,
        minimum_hull_slack_m=0.,
        conclusion='A continuous 22-station sufficient cover exists for radius >=1000 and closed 180-degree directional visibility over the radius-1800 disk. This refutes the necessity of 25 stations, but establishes neither minimum station count nor complete-task speed.',
        boundary='The saved JSON leaf coordinates are rounded approximations. The exact proof uses reconstructed rational midpoint leaves for the same station coordinates and witness lists. Zero hull slack means arbitrary coordinate perturbation is not certified. No solver, official simulator, or performance run was executed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--certificate', type=Path, default=Path(__file__).with_name('certificate-22-990.json'))
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('independent-exact-audit-22-990.json'))
    args = parser.parse_args()
    result = audit(args.certificate)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
