"""Experimental continuous certificates on a midpoint refinement of PolarCover.

Planning witnesses may include future stations. Actual channel certificates
are calculated separately, exclusively from observe_negative's real ledger.
"""
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import ConvexHull, QhullError

Q4 = next(p for p in Path(__file__).resolve().parents if (p/'polar_cover.py').is_file())
sys.path.insert(0, str(Q4))
from directional_cover import hull, inside
from polar_cover import PolarCover
from refined_cover_audit import exact_plan_valid


def refine(cells):
    a, b, c = cells[:, 0], cells[:, 1], cells[:, 2]
    ab, bc, ca = (a+b)/2, (b+c)/2, (c+a)/2
    return np.stack((np.stack((a, ab, ca), 1), np.stack((ab, b, bc), 1),
                     np.stack((ca, bc, c), 1), np.stack((ab, bc, ca), 1)), 1).reshape(-1, 3, 2)


class RefinedCover:
    def __init__(self):
        original = PolarCover()
        self.fixed_stations = original.stations.copy()
        self.parent_indices = original.indices.copy()
        cells = refine(refine(original.triangles))
        vertices = [tuple(p) for p in original.stations]
        registry = {p:i for i,p in enumerate(vertices)}
        indices = []
        for triangle in cells:
            row = []
            for point in map(tuple, triangle):
                if point not in registry:
                    registry[point] = len(vertices)
                    vertices.append(point)
                row.append(registry[point])
            indices.append(row)
        self.stations = np.asarray(vertices)
        self.indices = np.asarray(indices)
        self.triangles = self.stations[self.indices]
        self.parents = np.repeat(np.arange(36), 16)
        self.spacing = float(np.max(np.linalg.norm(np.roll(cells, -1, axis=1)-cells, axis=2)))
        self.negative = {c:[] for c in range(1, 21)}
        self.covered = {c:np.zeros(len(cells), bool) for c in range(1, 21)}
        self.witnesses = {c:{} for c in range(1, 21)}
        self._cache = {}

    def _calculate(self, points, stop_early=False, independent=False):
        points = np.asarray(points, float).reshape(-1, 2)
        covered = np.zeros(len(self.triangles), bool)
        witnesses = {}
        if len(points) < 3:
            return covered, witnesses
        eligible = np.linalg.norm(self.triangles[:, :, None, :]-points, axis=3).max(axis=1) <= 1000-1e-6
        lookup = {tuple(p):i for i,p in enumerate(points)}
        parent_witnesses = [[lookup.get(tuple(self.fixed_stations[k])) for k in row]
                            for row in self.parent_indices]
        for t, cell in enumerate(self.triangles):
            original = parent_witnesses[self.parents[t]]
            if not independent and all(i is not None for i in original) and np.all(eligible[t, original]):
                ok, selected = True, original
            else:
                selected = np.flatnonzero(eligible[t]).tolist()
                if len(selected) < 3:
                    ok = False
                elif independent:
                    try:
                        polygon = ConvexHull(points[selected])
                        ok = np.max(cell @ polygon.equations[:, :2].T+polygon.equations[:, 2]) <= 1e-8
                    except QhullError:
                        ok = False
                else:
                    ok = bool(np.all(inside(hull(points[selected]), cell)))
            if ok:
                covered[t] = True
                witnesses[t] = selected
            elif stop_early:
                return covered, witnesses
        return covered, witnesses

    def observe_negative(self, channel, point):
        q = np.asarray(point, float)
        if not any(np.linalg.norm(q-p) <= 1e-8 for p in self.negative[channel]):
            self.negative[channel].append(q.copy())

    def _update(self, channel):
        key = tuple(map(tuple, self.negative[channel]))
        if key not in self._cache:
            self._cache[key] = self._calculate(self.negative[channel])
        self.covered[channel], self.witnesses[channel] = self._cache[key]

    def complete(self, channel):
        self._update(channel)
        return bool(np.all(self.covered[channel])) and exact_plan_valid(self.negative[channel])

    def certificate(self, channel):
        self._update(channel)
        return dict(complete=self.complete(channel), triangles=len(self.triangles),
            covered=int(self.covered[channel].sum()),
            negative_points=[p.tolist() for p in self.negative[channel]],
            triangle_witnesses=self.witnesses[channel])

    def joint_plan_valid(self, actual, remaining, extra=None, independent=False):
        points = [np.asarray(p) for p in actual]
        points.extend(self.fixed_stations[k] for k in sorted(remaining))
        if extra is not None:
            points.append(np.asarray(extra))
        return bool(np.all(self._calculate(points, stop_early=True, independent=independent)[0]))

    def independent_domain_audit(self):
        """Separate barycentric construction, not another call to refine()."""
        original = PolarCover()
        expected = []
        for a,b,c in original.triangles:
            def point(i,j):
                return a+(b-a)*(i/4)+(c-a)*(j/4)
            for i in range(4):
                for j in range(4-i):
                    expected.append([point(i,j), point(i+1,j), point(i,j+1)])
                    if i+j < 3:
                        expected.append([point(i+1,j), point(i+1,j+1), point(i,j+1)])
        def canonical(triangles):
            return sorted(tuple(sorted(tuple(np.round(p, 8)) for p in t)) for t in triangles)
        assert canonical(expected) == canonical(self.triangles)
        assert len(self.indices) == 576
        assert original.outer_radius*np.cos(np.pi/12) > 1800
        return dict(passed=True, cells=576, vertices=len(self.stations),
            independent_barycentric_partition=True, original_disk_enclosure=True,
            coordinate_comparison_rounding_decimals=8,
            rounding_used_only_for_domain_identity_not_observation_certificate=True)
