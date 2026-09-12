"""Independent exact audit of stored station-layout certificates.

No geometry generator, solver, NumPy or SciPy imports. Inputs are binary-double
station coordinates plus combinatorial parent triangles and midpoint paths.
"""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path


def cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def norm2(a, b):
    return sum((x-y)**2 for x, y in zip(a, b))


def hull(points):
    ordered = sorted(set(points))
    chains = []
    for group in (ordered, ordered[::-1]):
        chain = []
        for p in group:
            while len(chain) > 1 and cross(chain[-2], chain[-1], p) <= 0:
                chain.pop()
            chain.append(p)
        chains.append(chain)
    return tuple(chains[0][:-1]+chains[1][:-1])


def inside(point, polygon):
    return all(cross(a, b, point) >= 0 for a, b in zip(polygon, polygon[1:]+polygon[:1]))


def on(a, b, p):
    return cross(a, b, p) == 0 and all(min(x, y) <= z <= max(x, y) for x, y, z in zip(a, b, p))


def audit(path):
    data = json.loads(path.read_text())
    assert data['passed'] and not data['failed']
    points = tuple(tuple(F(x) for x in p) for p in data['stations'])
    polygon = hull(points)
    for a, b in zip(polygon, polygon[1:]+polygon[:1]):
        support = a[0]*b[1]-a[1]*b[0]
        assert support > 0 and support**2 > 1800**2*norm2(a, b)
    directed = Counter()
    parents = [tuple(points[i] for i in ids) for ids in data['initial_triangles']]
    for triangle in parents:
        assert cross(*triangle) > 0 and all(inside(p, polygon) for p in triangle)
        directed.update(zip(triangle, triangle[1:]+triangle[:1]))
    assert all(v == 1 for v in directed.values())
    assert {e for e in directed if e[::-1] not in directed} == set(zip(polygon, polygon[1:]+polygon[:1]))
    edges = list({tuple(sorted(e)) for e in directed})
    for index, (a, b) in enumerate(edges):
        for c, d in edges[index+1:]:
            if len({a, b, c, d}) == 4:
                assert not (cross(a, b, c)*cross(a, b, d) < 0 and cross(c, d, a)*cross(c, d, b) < 0)
                assert not any((on(a, b, c), on(a, b, d), on(c, d, a), on(c, d, b)))
            else:
                common = ({a, b} & {c, d}).pop()
                p = ({a, b}-{common}).pop()
                q = ({c, d}-{common}).pop()
                assert not on(common, p, q) and not on(common, q, p)
    area = sum(cross(*p) for p in parents)
    assert area == sum(a[0]*b[1]-a[1]*b[0] for a, b in zip(polygon, polygon[1:]+polygon[:1]))
    children, leaves, area_by_parent = defaultdict(set), set(), defaultdict(F)
    distance_checks, hull_checks = 0, 0
    for item in data['cells']:
        parent, branch = item['parent'], item['branch']
        assert (parent, branch) not in leaves
        leaves.add((parent, branch))
        triangle = parents[parent]
        for step, digit in enumerate(branch):
            code = int(digit)
            assert code in range(6)
            children[(parent, branch[:step])].add(code)
            i, j, k = code % 3, (code+1) % 3, (code+2) % 3
            middle = tuple((x+y)/2 for x, y in zip(triangle[i], triangle[j]))
            triangle = ((triangle[i], middle, triangle[k]) if code < 3
                        else (middle, triangle[j], triangle[k]))
        assert cross(*triangle) > 0
        area_by_parent[parent] += cross(*triangle)
        witnesses = tuple(points[i] for i in item['witnesses'])
        poly = hull(witnesses)
        assert len(poly) >= 3
        for p in triangle:
            assert inside(p, poly)
            hull_checks += len(poly)
            for q in witnesses:
                assert norm2(p, q) <= 1000**2
                distance_checks += 1
    for key, codes in children.items():
        assert key not in leaves and len(codes) == 2
        first = min(codes)
        assert first in range(3) and codes == {first, first+3}
    for parent, triangle in enumerate(parents):
        assert (parent, '') in leaves or (parent, '') in children
        assert area_by_parent[parent] == cross(*triangle)
    return dict(passed=True, file=str(path), input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        station_count=len(points), parent_count=len(parents), cell_count=len(leaves),
        exact_disk_enclosure=True, exact_planar_partition=True, exact_complete_midpoint_tree=True,
        exact_radius_squared_checks=distance_checks, exact_convex_hull_cross_checks=hull_checks,
        no_float_tolerance=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--certificates', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = [audit(path) for path in args.certificates]
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(dict(passed=all(x['passed'] for x in rows), certificates=rows,
            auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), stream, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
