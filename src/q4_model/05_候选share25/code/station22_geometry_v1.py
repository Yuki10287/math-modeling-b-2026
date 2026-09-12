"""Experimental 14-outer / 7-inner / origin cover; actual-evidence only.

Receiver stations and exact proof-mesh vertices have distinct index spaces.
Coordinates and pre-witness sets are frozen by the independently audited hash.
"""
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

Q4 = next(p for p in Path(__file__).resolve().parents if (p/'polar_cover.py').is_file())
sys.path.insert(0, str(Q4))
from directional_cover import DirectionalCover, hull, inside

CERTIFICATE = Path(__file__).resolve().parent.parent/'results/station_geometry_dense_v1/candidate_3_certificate.json'
CERTIFICATE_SHA256 = 'ad7b777685f2b3cb3e590b8d6bc96da32dbe74426869ff313da9e1415546bb7f'


def cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


@lru_cache(maxsize=1)
def canonical():
    raw = CERTIFICATE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CERTIFICATE_SHA256
    data = json.loads(raw)
    points = tuple(tuple(float(x) for x in p) for p in data['stations'])
    exact = tuple(tuple(F(x) for x in p) for p in points)
    parents = [tuple(exact[i] for i in ids) for ids in data['initial_triangles']]
    cells = []
    for cell in data['cells']:
        triangle = parents[cell['parent']]
        for digit in cell['branch']:
            code = int(digit)
            i, j, k = code % 3, (code+1) % 3, (code+2) % 3
            mid = tuple((a+b)/2 for a, b in zip(triangle[i], triangle[j]))
            triangle = (triangle[i], mid, triangle[k]) if code < 3 else (mid, triangle[j], triangle[k])
        cells.append(triangle)
    assert len(points) == 22 and len(cells) == 274
    return points, tuple(cells), tuple(tuple(c['witnesses']) for c in data['cells'])


@lru_cache(maxsize=8192)
def exact_hull(keys):
    points = sorted(set(tuple(F(x) for x in p) for p in keys))
    chains = []
    for ordered in (points, points[::-1]):
        chain = []
        for p in ordered:
            while len(chain) > 1 and cross(chain[-2], chain[-1], p) <= 0:
                chain.pop()
            chain.append(p)
        chains.append(chain)
    return tuple(chains[0][:-1]+chains[1][:-1])


@lru_cache(maxsize=32768)
def exact_witness(cell_id, keys):
    poly = exact_hull(keys)
    if len(poly) < 3:
        return False
    points = tuple(tuple(F(x) for x in p) for p in keys)
    for v in canonical()[1][cell_id]:
        if any(cross(a, b, v) < 0 for a, b in zip(poly, poly[1:]+poly[:1])):
            return False
        if any(sum((a-b)**2 for a, b in zip(v, p)) > 1000**2 for p in points):
            return False
    return True


class Station22Cover(DirectionalCover):
    def __init__(self):
        stations, self.exact_cells, self.static_witnesses = canonical()
        self.stations = np.array(stations)
        vertices, indices = {}, []
        for triangle in self.exact_cells:
            row = []
            for p in triangle:
                if p not in vertices:
                    vertices[p] = len(vertices)
                row.append(vertices[p])
            indices.append(row)
        self.vertices = np.array(list(vertices), float)
        self.indices = np.array(indices, int)
        self.triangles = self.vertices[self.indices]
        self.spacing = 1000.  # Legacy return field; wrapper replaces geometry metadata.
        self.negative = {c: [] for c in range(1, 21)}
        self.covered = {c: np.zeros(len(self.triangles), bool) for c in range(1, 21)}
        self.witnesses = {c: {} for c in range(1, 21)}

    def observe_negative(self, channel, q):
        q = np.asarray(q, float)
        assert q.shape == (2,) and np.isfinite(q).all()
        if any(np.array_equal(q, p) for p in self.negative[channel]):
            return
        self.negative[channel].append(q.copy())
        points = np.array(self.negative[channel])
        if len(points) < 3:
            return
        actual_ids = {tuple(point): i for i, point in enumerate(points)}
        distance = np.linalg.norm(self.triangles[:, :, None, :]-points, axis=3)
        eligible = np.max(distance, axis=1) <= 1000.-1e-6
        for t in np.flatnonzero(~self.covered[channel] & (eligible.sum(axis=1) >= 3)):
            ids = [actual_ids.get(tuple(self.stations[k])) for k in self.static_witnesses[t]]
            if all(i is not None for i in ids):
                self.covered[channel][t] = True
                self.witnesses[channel][int(t)] = ids
                continue
            ids = np.flatnonzero(eligible[t])
            w = points[ids]
            if np.all(inside(hull(w), self.triangles[t])) and exact_witness(int(t), tuple(sorted(map(tuple, w)))):
                self.covered[channel][t] = True
                self.witnesses[channel][int(t)] = ids.tolist()

    def needed_stations(self, channels):
        needed = set()
        for c in channels:
            for t in np.flatnonzero(~self.covered[c]):
                for k in self.static_witnesses[t]:
                    if not any(np.array_equal(self.stations[k], p) for p in self.negative[c]):
                        needed.add(k)
        return sorted(needed)

    def geometry_certificate(self):
        return dict(geometry_sha256=CERTIFICATE_SHA256, station_count=22, cell_count=274,
            cell_geometry='exact_midpoint_descendants_of_binary_double_stations',
            stations=self.stations.tolist(), vertices=self.vertices.tolist(), triangles=self.indices.tolist())
