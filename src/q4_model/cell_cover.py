"""A 22-station directional cover with 108 proof cells.

Proof-cell vertices are geometric subdivisions, not receiver destinations.
Only ``observe_negative`` updates evidence; previewing an observation never does.
The exact cells are midpoint descendants of the stored binary-double stations.
"""
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from directional_cover import DirectionalCover, hull, inside


CERTIFICATE_PATH = Path(__file__).parent / 'results/coverage_probe_400/certificate-22-990.json'
CERTIFICATE_SHA256 = '29e52deef3efda2b39b2673677c766897f3537a81da2031ee395cec01163090a'


def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _label(triangle):
    # Rounded keys associate subdivision nodes with labels, not predicates.
    return tuple(sorted(tuple(round(float(x), 8) for x in v) for v in triangle))


@lru_cache(maxsize=1)
def _canonical():
    raw = CERTIFICATE_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CERTIFICATE_SHA256:
        raise ValueError('22-station geometry certificate changed; independent audit required')
    data = json.loads(raw)
    stations = tuple(tuple(float(x) for x in q) for q in data['stations'])
    exact_stations = [tuple(Fraction(x) for x in q) for q in stations]
    lookup = {_label(c['vertices']): i for i, c in enumerate(data['cells'])}
    leaves = {}
    stack = []
    for ids in data['initial_triangles']:
        exact = tuple(exact_stations[i] for i in ids)
        stack.append((exact, np.array([stations[i] for i in ids]), 0))
    while stack:
        exact, approx, depth = stack.pop()
        label = lookup.get(_label(approx))
        if label is not None:
            if label in leaves or depth != data['cells'][label]['depth']:
                raise ValueError('Ambiguous proof-cell subdivision')
            leaves[label] = exact if _cross(*exact) > 0 else exact[::-1]
            continue
        if depth >= 12:
            raise ValueError('Unmatched proof-cell subdivision')
        i = int(np.argmax(np.linalg.norm(np.roll(approx, -1, axis=0)-approx, axis=1)))
        j, k = (i+1) % 3, (i+2) % 3
        mid = tuple((a+b)/2 for a, b in zip(exact[i], exact[j]))
        approx_mid = (approx[i]+approx[j])/2
        stack.extend([
            ((exact[i], mid, exact[k]), np.array([approx[i], approx_mid, approx[k]]), depth+1),
            ((mid, exact[j], exact[k]), np.array([approx_mid, approx[j], approx[k]]), depth+1)])
    if set(leaves) != set(range(108)) or len(stations) != 22:
        raise ValueError('Incomplete canonical cell cover')
    triangles = tuple(leaves[i] for i in range(108))
    pre_witnesses = tuple(tuple(c['witnesses']) for c in data['cells'])
    return stations, triangles, pre_witnesses


@lru_cache(maxsize=8192)
def _exact_hull(points):
    pts = sorted(tuple(Fraction(x) for x in p) for p in points)
    chains = []
    for order in (pts, pts[::-1]):
        chain = []
        for p in order:
            while len(chain) >= 2 and _cross(chain[-2], chain[-1], p) <= 0:
                chain.pop()
            chain.append(p)
        chains.append(chain)
    return tuple(chains[0][:-1]+chains[1][:-1])


@lru_cache(maxsize=32768)
def _exact_witness_cover(cell_id, points):
    """Certify actual double-valued negative points against exact proof cells."""
    polygon = _exact_hull(points)
    if len(polygon) < 3:
        return False
    vertices = _canonical()[1][cell_id]
    for v in vertices:
        for a, b in zip(polygon, polygon[1:]+polygon[:1]):
            if _cross(a, b, v) < 0:
                return False
        for point in points:
            if sum((a-Fraction(b))**2 for a, b in zip(v, point)) > 1000**2:
                return False
    return True


class CellCover(DirectionalCover):
    def __init__(self):
        stations, exact_cells, pre_witnesses = _canonical()
        self.stations = np.array(stations, float)
        self.exact_cells = exact_cells
        # Stable exact-coordinate deduplication; adjacent cells share vertices.
        vertex_ids = {}
        indices = []
        for triangle in exact_cells:
            row = []
            for p in triangle:
                if p not in vertex_ids:
                    vertex_ids[p] = len(vertex_ids)
                row.append(vertex_ids[p])
            indices.append(row)
        self.vertices = np.array([[float(x) for x in p] for p in vertex_ids])
        self.indices = np.array(indices, int)
        self.triangles = self.vertices[self.indices]
        self.pre_witnesses = pre_witnesses
        self.static_witnesses = pre_witnesses
        self.negative = {c: [] for c in range(1, 21)}
        self.covered = {c: np.zeros(len(exact_cells), bool) for c in range(1, 21)}
        self.witnesses = {c: {} for c in range(1, 21)}

    @staticmethod
    def _point(q):
        q = np.asarray(q, float)
        if q.shape != (2,) or not np.isfinite(q).all():
            raise ValueError('Negative observation must be a finite 2D point')
        return q

    def _new_witnesses(self, channel, points):
        if len(points) < 3:
            return {}
        points = np.asarray(points, float)
        distance = np.linalg.norm(self.triangles[:, :, None, :]-points, axis=3)
        eligible = np.max(distance, axis=1) <= 1000-1e-6
        found = {}
        actual_ids = {tuple(point): i for i, point in enumerate(points)}
        for t in np.flatnonzero(~self.covered[channel] & (eligible.sum(axis=1) >= 3)):
            static_ids = [actual_ids.get(tuple(self.stations[k])) for k in self.pre_witnesses[t]]
            if all(i is not None for i in static_ids):
                # These exact receiver coordinates and exact midpoint cells
                # have an independent static Fraction certificate. Reusing it
                # requires every listed point in this channel's real ledger.
                found[int(t)] = static_ids
                continue
            ids = np.flatnonzero(eligible[t])
            w = points[ids]
            if not np.all(inside(hull(w), self.triangles[t])):
                continue
            key = tuple(sorted(map(tuple, w)))
            if _exact_witness_cover(int(t), key):
                found[int(t)] = ids.tolist()
        return found

    def observe_negative(self, channel, q):
        q = self._point(q)
        if any(np.array_equal(q, p) for p in self.negative[channel]):
            return
        self.negative[channel].append(q.copy())
        for t, witnesses in self._new_witnesses(channel, self.negative[channel]).items():
            self.covered[channel][t] = True
            self.witnesses[channel][t] = witnesses

    def preview_negative(self, channel, q):
        """Return a new boolean mask of hypothetical newly certified cells."""
        q = self._point(q)
        gain = np.zeros(len(self.triangles), bool)
        if any(np.array_equal(q, p) for p in self.negative[channel]):
            return gain
        found = self._new_witnesses(channel, self.negative[channel]+[q])
        if found:
            gain[list(found)] = True
        return gain

    def needed_stations(self, channels):
        needed = set()
        for c in channels:
            candidates = {k for t in np.flatnonzero(~self.covered[c])
                          for k in self.pre_witnesses[t]}
            # Exact equality is deliberate: the proof has zero hull slack at
            # some boundaries, so a nearby point cannot stand in for a station.
            needed.update(k for k in candidates
                          if not any(np.array_equal(self.stations[k], q) for q in self.negative[c]))
        return sorted(needed)

    def geometry_certificate(self):
        return dict(basis='directional_cell_cover',
                    geometry_sha256=CERTIFICATE_SHA256,
                    station_count=len(self.stations), cell_count=len(self.triangles),
                    stations=self.stations.tolist(), vertices=self.vertices.tolist(),
                    triangles=self.indices.tolist(),
                    cell_geometry='exact_midpoint_descendants_of_binary_double_stations',
                    cell_vertex_role='proof subdivision vertices; not measurement stations')
