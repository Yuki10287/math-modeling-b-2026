"""Local-only station geometry screening; no simulator/client imports.

Dense samples reject candidates only. Acceptance requires exact rational
triangulation, disk enclosure and per-cell convex-hull/radius witnesses.
Historical source files and results are read, never modified.
"""
import argparse
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import Delaunay

Q4 = next(p for p in Path(__file__).resolve().parents if (p/'polar_cover.py').is_file())
sys.path.insert(0, str(Q4))
from route_planning import open_route, route_length
from polar_cover import PolarCover


def cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def norm2(a, b):
    return sum((x-y)**2 for x, y in zip(a, b))


def hull(points):
    points = sorted(set(points))
    chains = []
    for order in (points, points[::-1]):
        chain = []
        for p in order:
            while len(chain) > 1 and cross(chain[-2], chain[-1], p) <= 0:
                chain.pop()
            chain.append(p)
        chains.append(chain)
    return tuple(chains[0][:-1]+chains[1][:-1])


def stations(m, n, radius, phase=0.):
    a = np.arange(m)*2*np.pi/m
    b = np.arange(n)*2*np.pi/n+phase
    outer = (1800.0001/np.cos(np.pi/m))*np.c_[np.cos(a), np.sin(a)]
    inner = radius*np.c_[np.cos(b), np.sin(b)]
    return np.vstack((np.zeros((1, 2)), inner, outer))


def grid(nang=120, nrad=31):
    a = np.arange(nang)*2*np.pi/nang+.000137
    r = np.linspace(0, 1800, nrad)
    return (r[:, None, None]*np.c_[np.cos(a), np.sin(a)][None, :, :]).reshape(-1, 2)


def assess(q, g):
    delta = q[None, :, :]-g[:, None, :]
    d2 = (delta*delta).sum(axis=2)
    eligible = d2 <= 1000**2
    angle = np.where(eligible, np.arctan2(delta[:, :, 1], delta[:, :, 0]), 10.)
    angle.sort(axis=1)
    count = eligible.sum(axis=1)
    gaps = np.diff(angle, axis=1)
    gaps = np.where(np.arange(len(q)-1)[None, :] < count[:, None]-1, gaps, 0.)
    last = angle[np.arange(len(g)), np.maximum(0, count-1)]
    gap = np.maximum(gaps.max(axis=1), angle[:, 0]+2*np.pi-last)
    gap = np.where(count == 0, 2*np.pi, gap)
    gap = np.where(d2.min(axis=1) == 0, 0., gap)
    k = int(np.argmax(gap))
    return dict(bad=int((gap > np.pi+1e-11).sum()), points=len(g),
                max_gap_deg=float(np.degrees(gap[k])), worst_point=g[k].tolist())


def route(q):
    order = open_route(q, np.zeros(2))
    return dict(route_m=route_length(q, np.zeros(2), order), route=order)


def segment_hit(a, b, c, d):
    def on(a, b, p):
        return cross(a, b, p) == 0 and all(min(x, y) <= z <= max(x, y) for x, y, z in zip(a, b, p))
    p, q, r, s = cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b)
    return p*q < 0 and r*s < 0 or on(a, b, c) or on(a, b, d) or on(c, d, a) or on(c, d, b)


def exact_certificate(q, max_depth=16):
    """Prove stored binary-double station geometry, not a sampled domain."""
    exact = tuple(tuple(F(float(x)) for x in point) for point in q)
    polygon = hull(exact)
    enclosure = []
    for a, b in zip(polygon, polygon[1:]+polygon[:1]):
        support = a[0]*b[1]-a[1]*b[0]
        enclosure.append(support > 0 and support**2 > 1800**2*norm2(a, b))
    if not all(enclosure):
        return dict(passed=False, reason='disk enclosure')
    initial = []
    for ids in Delaunay(q).simplices:
        ids = tuple(map(int, ids))
        if cross(*(exact[i] for i in ids)) < 0:
            ids = ids[::-1]
        initial.append(ids)
    directed = Counter(edge for ids in initial for edge in zip(ids, ids[1:]+ids[:1]))
    assert all(count == 1 for count in directed.values())
    edges = sorted({tuple(sorted(edge)) for edge in directed})
    for ei, (i, j) in enumerate(edges):
        for k, l in edges[ei+1:]:
            if len({i, j, k, l}) == 4:
                assert not segment_hit(exact[i], exact[j], exact[k], exact[l])
    assert sum(cross(*(exact[i] for i in ids)) for ids in initial) == sum(
        a[0]*b[1]-a[1]*b[0] for a, b in zip(polygon, polygon[1:]+polygon[:1]))
    boundary = {(exact[i], exact[j]) for i, j in directed if (j, i) not in directed}
    assert boundary == set(zip(polygon, polygon[1:]+polygon[:1]))
    for ids in initial:
        assert cross(*(exact[i] for i in ids)) > 0
        assert all(cross(a, b, exact[i]) >= 0 for i in ids for a, b in zip(polygon, polygon[1:]+polygon[:1]))
    leaves, failed, stack = [], [], [(tuple(exact[i] for i in ids), parent, '') for parent, ids in enumerate(initial)]
    witness_cache = {}
    max_distance2 = F(0)
    while stack:
        triangle, parent, branch = stack.pop()
        approx = np.array(triangle, float)
        eligible = tuple(map(int, np.flatnonzero(np.max(np.linalg.norm(approx[:, None, :]-q, axis=2), axis=0) <= 1000.-1e-6)))
        if eligible not in witness_cache:
            witness_cache[eligible] = hull(tuple(exact[k] for k in eligible))
        poly = witness_cache[eligible]
        if len(poly) >= 3 and all(cross(a, b, v) >= 0 for v in triangle for a, b in zip(poly, poly[1:]+poly[:1])):
            distances = [norm2(v, exact[k]) for v in triangle for k in eligible]
            if max(distances) <= 1000**2:
                max_distance2 = max(max_distance2, max(distances))
                leaves.append(dict(parent=parent, branch=branch, witnesses=eligible))
                continue
        if len(branch) >= max_depth:
            failed.append(dict(parent=parent, branch=branch, vertices=approx.tolist()))
            continue
        i = int(np.argmax(np.linalg.norm(np.roll(approx, -1, axis=0)-approx, axis=1)))
        j, k = (i+1) % 3, (i+2) % 3
        mid = tuple((a+b)/2 for a, b in zip(triangle[i], triangle[j]))
        # Store split edge in path, sufficient for independent exact reconstruction.
        stack.append(((triangle[i], mid, triangle[k]), parent, branch+str(i)))
        stack.append(((mid, triangle[j], triangle[k]), parent, branch+str(i+3)))
        if len(leaves)+len(failed)+len(stack) > 150000:
            return dict(passed=False, reason='cell budget', cells=len(leaves))
    return dict(passed=not failed, stations=q.tolist(), initial_triangles=initial,
        cells=leaves, failed=failed, cell_count=len(leaves), failed_count=len(failed),
        max_depth=max(map(lambda c:len(c['branch']), leaves+failed)),
        max_witness_distance_m=math.sqrt(float(max_distance2)),
        exact_binary_station_coordinates=True, exact_disk_enclosure=True,
        exact_parent_planarity_area_and_boundary=True,
        exact_midpoint_cell_partition=True, exact_convex_hull_and_radius=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    def save(name, value):
        with (args.out/name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
    coarse = grid()
    rows = []
    for m in range(12, 17):
        for n in range(6, min(12, 24-m)+1):
            for radius in range(900, 1001, 10):
                for fraction in (0., .25, .5):
                    phase = fraction*2*np.pi/math.lcm(m, n)
                    q = stations(m, n, radius, phase)
                    check = assess(q, coarse)
                    if check['bad'] == 0:
                        rows.append(dict(m=m, n=n, inner_radius=radius, phase=phase,
                            count=len(q), screening=check, **route(q)))
        print('SCREEN', m, 'passes', len(rows), 'wall_s', round(time.perf_counter()-started, 2), flush=True)
    rows.sort(key=lambda x:(x['route_m'], x['count']))
    save('screening.json', dict(coarse_points=len(coarse), passing_rows=rows,
        role='necessary pointwise screening only; not proof', elapsed_s=time.perf_counter()-started))
    baseline = PolarCover().stations
    selected = [dict(name='baseline25', points=baseline),
                dict(name='inward25_r925', points=stations(12, 12, 925., np.pi/12))]
    for row in rows:
        if len(selected) >= 5:
            break
        if all(len(s['points']) != row['count'] for s in selected):
            selected.append(dict(name='screen_best_'+str(row['count']),
                points=stations(row['m'], row['n'], row['inner_radius'], row['phase']), specification=row))
    summary = []
    for item in selected:
        q = item.pop('points')
        cert = exact_certificate(q)
        save(item['name']+'_certificate.json', cert)
        result = dict(**item, station_count=len(q), **route(q),
            certificate_passed=cert['passed'], cells=cert.get('cell_count'), failures=cert.get('failed_count'))
        summary.append(result)
        print('CERTIFICATE', result, flush=True)
    save('summary.json', dict(local_geometry_only=True, official_contacted=False,
        seed_not_applicable=True, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dependencies={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Q4/'route_planning.py', Q4/'polar_cover.py', Q4/'directional_cover.py')},
        candidates=summary, elapsed_s=time.perf_counter()-started,
        note='All routes are feasible heuristic open paths, not optimality claims or full-simulation results.'))


if __name__ == '__main__':
    main()
