"""Independent exact audit for the experimental same-domain refined cover.

The proof cells are rational quarter-barycentric descendants of the stored
binary-double PolarCover stations. No experimental geometry or policy module
is imported. Floating point selects candidate witnesses only; acceptance uses
Fraction predicates without hull tolerances. Predicted/future points are valid
only in exact_plan_valid; exact_certificate_audit checks a real event ledger.
"""
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

Q4 = next(p for p in Path(__file__).resolve().parents if (p/'polar_cover.py').is_file())
sys.path.insert(0, str(Q4))
from polar_cover import PolarCover


def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _norm2(a, b):
    return sum((x-y)**2 for x, y in zip(a, b))


def _on_segment(a, b, p):
    return (_cross(a, b, p) == 0 and
            all(min(x, y) <= z <= max(x, y) for x, y, z in zip(a, b, p)))


def _segments_intersect(a, b, c, d):
    p, q, r, s = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
    return (p*q < 0 and r*s < 0 or _on_segment(a, b, c) or
            _on_segment(a, b, d) or _on_segment(c, d, a) or _on_segment(c, d, b))


@lru_cache(maxsize=1)
def _canonical():
    original = PolarCover()
    point_keys = tuple(tuple(float(x) for x in p) for p in original.stations)
    points = tuple(tuple(Fraction(x) for x in p) for p in point_keys)
    parents = tuple(tuple(int(i) for i in row) for row in original.indices)
    outer_ids = tuple(range(13, 25))
    outer = tuple(points[i] for i in outer_ids)
    # Exact half-plane enclosure of the 1800 m disk by the stored polygon.
    margins = []
    for a, b in zip(outer, outer[1:]+outer[:1]):
        support = a[0]*b[1]-a[1]*b[0]
        edge2 = _norm2(a, b)
        assert support > 0 and edge2 > 0
        assert support**2 > 1800**2*edge2, 'stored outer edge fails exact disk enclosure'
        assert all(_cross(a, b, p) >= 0 for p in outer), 'outer polygon not convex'
        margins.append(float(support**2/edge2-1800**2))
    # Independently check the original triangulation's planar topology and area.
    directed = Counter()
    parent_area2 = Fraction(0)
    static_parent_keys = []
    for ids in parents:
        triangle = tuple(points[i] for i in ids)
        area2 = _cross(*triangle)
        assert area2 > 0
        parent_area2 += area2
        for a, b in zip(triangle, triangle[1:]+triangle[:1]):
            assert all(_cross(x, y, a) >= 0 for x, y in zip(outer, outer[1:]+outer[:1]))
            assert _norm2(a, b) < 1000**2, 'parent reception witness exceeds radius'
        for i, j in zip(ids, ids[1:]+ids[:1]):
            directed[(i, j)] += 1
        static_parent_keys.append(tuple(point_keys[i] for i in ids))
    polygon_area2 = sum(a[0]*b[1]-a[1]*b[0] for a, b in zip(outer, outer[1:]+outer[:1]))
    assert parent_area2 == polygon_area2
    expected_boundary = set(zip(outer_ids, outer_ids[1:]+outer_ids[:1]))
    boundary = set()
    for edge, count in directed.items():
        assert count == 1
        if edge[::-1] not in directed:
            boundary.add(edge)
    assert boundary == expected_boundary
    edges = sorted({tuple(sorted(edge)) for edge in directed})
    for index, (i, j) in enumerate(edges):
        for k, l in edges[index+1:]:
            if len({i, j, k, l}) == 4:
                assert not _segments_intersect(points[i], points[j], points[k], points[l]), 'mesh edges cross'
            elif {i, j} != {k, l}:
                common = ({i, j} & {k, l}).pop()
                first, second = ({i, j}-{common}).pop(), ({k, l}-{common}).pop()
                assert not _on_segment(points[common], points[first], points[second])
                assert not _on_segment(points[common], points[second], points[first])
    cells = []
    for ids in parents:
        a, b, c = (points[i] for i in ids)
        def bary(i, j):
            return tuple(x+(y-x)*Fraction(i, 4)+(z-x)*Fraction(j, 4) for x, y, z in zip(a, b, c))
        begin = len(cells)
        for i in range(4):
            for j in range(4-i):
                cells.append((bary(i, j), bary(i+1, j), bary(i, j+1)))
                if i+j < 3:
                    cells.append((bary(i+1, j), bary(i+1, j+1), bary(i, j+1)))
        group = cells[begin:]
        assert len(group) == 16 and all(_cross(*cell) > 0 for cell in group)
        assert sum(_cross(*cell) for cell in group) == _cross(a, b, c)
    assert len(cells) == 576
    return dict(cells=tuple(cells), approx=np.array(cells, dtype=float),
        parent_keys=tuple(static_parent_keys),
        report=dict(passed=True, cells=576, original_stations=25,
            exact_binary_station_coordinates=True, exact_quarter_barycentric_cells=True,
            exact_outer_disk_enclosure=True, exact_parent_planarity_and_area=True,
            exact_parent_radius_bound=True, float_tolerance_used_for_acceptance=False,
            minimum_edge_squared_distance_margin_m2=min(margins)))


def domain_audit():
    """Exact disk enclosure, parent mesh topology, and quarter-grid partition."""
    return dict(_canonical()['report'])


def _point_keys(points):
    a = np.asarray(points, float)
    if a.size == 0:
        return ()
    if a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all():
        raise ValueError('Expected finite 2D witness points')
    return tuple(sorted(set(tuple(float(x) for x in p) for p in a)))


@lru_cache(maxsize=8192)
def _exact_hull(keys):
    points = tuple(tuple(Fraction(x) for x in p) for p in keys)
    chains = []
    for ordered in (points, points[::-1]):
        chain = []
        for p in ordered:
            while len(chain) >= 2 and _cross(chain[-2], chain[-1], p) <= 0:
                chain.pop()
            chain.append(p)
        chains.append(chain)
    return tuple(chains[0][:-1]+chains[1][:-1])


@lru_cache(maxsize=65536)
def _exact_cell_covered(cell_id, keys):
    polygon = _exact_hull(keys)
    if len(polygon) < 3:
        return False
    points = tuple(tuple(Fraction(x) for x in p) for p in keys)
    for v in _canonical()['cells'][cell_id]:
        if any(_cross(a, b, v) < 0 for a, b in zip(polygon, polygon[1:]+polygon[:1])):
            return False
        if any(_norm2(v, p) > 1000**2 for p in points):
            return False
    return True


@lru_cache(maxsize=4096)
def _plan_valid(keys):
    if len(keys) < 3:
        return False
    geometry = _canonical()
    available = set(keys)
    parent_present = [all(p in available for p in group) for group in geometry['parent_keys']]
    # A complete parent triple was independently certified above. Its convex
    # descendants are also covered, by convexity of distance and the hull.
    unresolved = [i for i in range(576) if not parent_present[i//16]]
    if not unresolved:
        return True
    a = np.asarray(keys)
    cells = geometry['approx'][unresolved]
    eligible = np.linalg.norm(cells[:, :, None, :]-a, axis=3).max(axis=1) <= 1000-1e-6
    for i, selection in zip(unresolved, eligible):
        witnesses = tuple(keys[j] for j in np.flatnonzero(selection))
        if len(witnesses) < 3 or not _exact_cell_covered(i, witnesses):
            return False
    return True


def exact_plan_valid(points):
    """Sufficient full-domain cover proof for a hypothetical or real point set."""
    return _plan_valid(_point_keys(points))


def exact_certificate_audit(result, events):
    """Reprove completion using only exact-matching real no-signal feedback."""
    def channels(values):
        assert isinstance(values, (list, tuple))
        assert all(isinstance(c, (int, np.integer)) and not isinstance(c, (bool, np.bool_)) and 1 <= c <= 20 for c in values)
        assert len(values) == len(set(values)), 'duplicate certificate channel'
        return set(values)
    assert result['complete']
    for event in events:
        channels([event['channel']])
        assert event['action'] in ('measure', 'clear')
        if event['action'] == 'measure':
            assert event['measure_result'] in ('no_signal', 'near', 'direction')
        else:
            assert event['clear_result'] in ('success', 'no_target_in_range')
    removed = {e['channel'] for e in events if e['action'] == 'clear' and e['clear_result'] == 'success'}
    assert channels(result['cleared']) == removed
    cert = result['certificate']
    assert cert['basis'] in ('directional_triangle_cover', 'count_upper_bound')
    assert channels(cert['cleared']) == removed
    absent = channels(cert['absent'])
    detected = {e['channel'] for e in events if e['action'] == 'measure' and e['measure_result'] != 'no_signal'}
    assert not absent & detected, 'absent channel has actual positive feedback'
    assert not absent & removed
    assert set(cert['channels']) == {str(c) for c in absent}
    if cert['basis'] == 'count_upper_bound':
        assert len(removed) == 16
        return dict(passed=True, basis='count_upper_bound', exact_domain=domain_audit(), channels=0)
    assert not absent & removed and absent | removed == set(range(1, 21))
    negative = {c:set() for c in absent}
    for event in events:
        if event['action'] == 'measure' and event['channel'] in absent and event['measure_result'] == 'no_signal':
            negative[event['channel']].add(tuple(float(x) for x in event['position']))
    for c in absent:
        points = _point_keys(cert['channels'][str(c)]['negative_points'])
        assert set(points) <= negative[c], 'certificate contains an unobserved point'
        assert exact_plan_valid(points), 'real negative points do not exactly prove full-domain cover'
    return dict(passed=True, basis='exact_quarter_cell_negative_hulls',
        exact_domain=domain_audit(), channels=len(absent), cells_per_channel=576,
        verified_cells=576*len(absent), real_negative_points_only=True,
        independent_of_claimed_mesh_and_witness_indices=True)
