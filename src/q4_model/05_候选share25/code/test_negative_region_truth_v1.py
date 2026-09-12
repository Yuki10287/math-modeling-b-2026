"""Independent finite true-source retention challenges for negative contraction.

This is a local correctness test, not a simulator performance experiment.
Inputs include legal behind-source negatives, exact heading boundaries, larger
minimum compatible radius, all four heading quadrants, and degenerate regions.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import random
import time

from negative_region_boxes_v1 import contract_region, interval_nonempty


def orientation(a, b, p):
    return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])


def exact(points):
    return [tuple(F(float(x)) for x in p) for p in points]


def monotone_hull(points):
    ordered = sorted(set(points))
    low, high = [], []
    for arr, seq in ((low, ordered), (high, ordered[::-1])):
        for p in seq:
            while len(arr) >= 2 and orientation(arr[-2], arr[-1], p) <= 0:
                arr.pop()
            arr.append(p)
    return tuple(low[:-1]+high[:-1])


def independent_contains(poly, point):
    if len(poly) == 1:
        return poly[0] == point
    if len(poly) == 2:
        a, b = poly
        return orientation(a, b, point) == 0 and all(min(x, y) <= z <= max(x, y) for x, y, z in zip(a, b, point))
    poly = monotone_hull(poly)
    return all(orientation(a, b, point) >= 0 for a, b in zip(poly, poly[1:]+poly[:1]))


def scenarios():
    rows = []
    normals = ((1, 0), (0, 1), (-1, 0), (0, -1), (3, 4), (-3, 4), (3, -4), (-3, -4))
    for index, n in enumerate(normals):
        scale = 1 if index < 4 else 5
        for radius in (1000, 1500):
            g = (float(index*125-500), float(index*75-300))
            def point(a, b):
                return [g[0]+(a*n[0]-b*n[1])/scale, g[1]+(a*n[1]+b*n[0])/scale]
            rows.append(dict(name=f'quadrant_{index}_radius_{radius}', source=g, radius=radius,
                normal=n, polygon=[point(-700, -100), point(700, -100), point(700, 100), point(-700, 100)],
                positives=[point(radius-10, 0), point(0, 200)],
                negatives=[point(-100, 0), point(-200, 100), point(radius+50, 0), point(-50, -300)]))
    # Directional near-boundary feedback uses binary powers, preserving strictness.
    for sign in (-1, 1):
        rows.append(dict(name=f'heading_boundary_{sign}', source=[0., 0.], radius=1000,
            normal=[sign, 0], polygon=[[-700., -5.], [700., -5.], [700., 5.], [-700., 5.]],
            positives=[[0., 200.], [sign*999., 0.]],
            negatives=[[-sign*2.**-30, 400.], [-sign*100., 0.]]))
    for radius in (1000, 1500):
        rows.append(dict(name=f'omni_radius_{radius}', source=[100., -100.], radius=radius,
            normal=None, polygon=[[-600., -300.], [800., -300.], [800., 100.], [-600., 100.]],
            positives=[[100.+radius-5., -100.], [100., -90.]],
            negatives=[[100.+radius+5., -100.], [100., radius-95.]]))
    randomizer = random.Random(88001)
    for index in range(12):
        g = [float(randomizer.randrange(-1000, 1001)), float(randomizer.randrange(-1000, 1001))]
        n = normals[index % len(normals)]
        radius = randomizer.choice((1000, 1300, 1500))
        positives, negatives = [], []
        while len(positives) < 3 or len(negatives) < 5:
            d = [randomizer.randrange(-2200, 2201), randomizer.randrange(-2200, 2201)]
            visible = sum(a*b for a, b in zip(d, n)) >= 0 and sum(x*x for x in d) <= radius**2
            target = positives if visible else negatives
            limit = 3 if visible else 5
            if len(target) < limit:
                target.append([g[0]+d[0], g[1]+d[1]])
        rows.append(dict(name=f'legal_random_{index}', source=g, radius=radius, normal=n,
            polygon=[[g[0]-700, g[1]-40], [g[0]+700, g[1]-40], [g[0]+700, g[1]+40], [g[0]-700, g[1]+40]],
            positives=positives, negatives=negatives))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    module = Path(__file__).with_name('negative_region_boxes_v1.py')
    before = hashlib.sha256(module.read_bytes()).hexdigest()
    interval_cases = [([], True), ([(F(0), F(0), True)], False),
        ([(F(1), F(0), False), (F(-1), F(0), False)], True),
        ([(F(1), F(0), True), (F(-1), F(0), False)], False),
        ([(F(1), F(-1), False)], True), ([(F(1), F(-1), True)], False),
        ([(F(2), F(-1), True), (F(-2), F(1), False)], False)]
    for constraints, expected in interval_cases:
        assert interval_nonempty(constraints) == expected
    rows = []
    for scenario in scenarios():
        g = exact([scenario['source']])[0]
        n, radius = scenario['normal'], scenario['radius']
        for kind in ('positives', 'negatives'):
            for p in exact(scenario[kind]):
                d = tuple(a-b for a, b in zip(p, g))
                visible = sum(x*x for x in d) <= radius**2 and (n is None or sum(x*y for x, y in zip(d, n)) >= 0)
                assert visible == (kind == 'positives'), (scenario['name'], kind)
        assert independent_contains(exact(scenario['polygon']), g)
        start = time.perf_counter()
        result = contract_region(scenario['polygon'], scenario['positives'], scenario['negatives'], max_depth=5)
        kept = independent_contains(exact(result['polygon']), g)
        assert kept, scenario['name']
        rows.append(dict(**scenario, passed=True, true_source_preserved_exactly=kept,
            result=result, runtime_s=time.perf_counter()-start))
    for P in ([[0., 0.]], [[-100., 0.], [100., 0.]]):
        result = contract_region(P, [[100., 0.]], [[-50., 0.]], max_depth=5)
        assert result['skipped'] and independent_contains(exact(result['polygon']), (F(0), F(0)))
    stable = hashlib.sha256(module.read_bytes()).hexdigest() == before
    assert stable
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(dict(passed=True, cases=rows, interval_boundary_checks=len(interval_cases),
            degenerate_region_checks=2, seed=88001, dependency_stable=stable,
            contraction_sha256=before, test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            official_contacted=False, finite_correctness_test_not_performance=True), stream, indent=2)
    print('PASSED', len(rows), 'true-source cases, 7 interval cases, 2 degenerate cases')


if __name__ == '__main__':
    main()
