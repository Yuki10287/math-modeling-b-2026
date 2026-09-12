"""Continuous, all-orientation discovery certificates for mixed source types."""
import math
import numpy as np


def hull(points):
    pts = sorted(set(map(tuple, np.asarray(points, float).reshape(-1, 2))))
    if len(pts) < 3:
        return np.asarray(pts).reshape(-1, 2)
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])


def inside(P, points, tolerance=1e-8):
    if len(P) < 3:
        return np.zeros(len(points), bool)
    edges = np.roll(P, -1, axis=0)-P
    v = np.asarray(points)[:, None, :]-P
    signed = (edges[None, :, 0]*v[:, :, 1]-edges[None, :, 1]*v[:, :, 0])
    return np.all(signed >= -tolerance*np.linalg.norm(edges, axis=1), axis=1)


def triangle_distance(T):
    if inside(T, np.zeros((1, 2)))[0]:
        return 0.
    edges = np.roll(T, -1, axis=0)-T
    alpha = np.clip(-np.sum(T*edges, axis=1)/np.sum(edges*edges, axis=1), 0, 1)
    return float(np.min(np.linalg.norm(T+alpha[:, None]*edges, axis=1)))


class DirectionalCover:
    def __init__(self, spacing=950.):
        if not 100 <= spacing < 1000:
            raise ValueError('triangle edges must be safely below 1000 m')
        # The disk has lattice coordinates |i|,|j| <= 2*1800/(sqrt(3)*spacing).
        n = math.ceil(2*1800/(math.sqrt(3)*spacing))+1
        def point(key):
            i, j = key
            return np.array([spacing*(i+.5*j), spacing*math.sqrt(3)/2*j])
        cells = []
        for i in range(-n, n):
            for j in range(-n, n):
                for keys in [((i,j),(i+1,j),(i,j+1)),
                             ((i+1,j+1),(i,j+1),(i+1,j))]:
                    if triangle_distance(np.array([point(k) for k in keys])) <= 1800+1e-7:
                        cells.append(keys)
        keys = sorted(set(k for cell in cells for k in cell))
        ids = {k: i for i, k in enumerate(keys)}
        self.stations = np.array([point(k) for k in keys])
        self.indices = np.array([[ids[k] for k in cell] for cell in cells])
        self.triangles = self.stations[self.indices]
        self.spacing = spacing
        self.negative = {c: [] for c in range(1, 21)}
        self.covered = {c: np.zeros(len(cells), bool) for c in range(1, 21)}
        self.witnesses = {c: {} for c in range(1, 21)}

    def observe_negative(self, channel, q):
        q = np.asarray(q, float)
        if any(np.linalg.norm(q-p) <= 1e-8 for p in self.negative[channel]):
            return
        self.negative[channel].append(q.copy())
        points = np.array(self.negative[channel])
        if len(points) < 3:
            return
        # Every eligible witness lies within 1000 m of EVERY point of a triangle.
        distances = np.linalg.norm(self.triangles[:, :, None, :]-points, axis=3)
        eligible = np.max(distances, axis=1) <= 1000-1e-6
        for t in np.flatnonzero(~self.covered[channel] & (eligible.sum(axis=1) >= 3)):
            witnesses = np.flatnonzero(eligible[t])
            if np.all(inside(hull(points[witnesses]), self.triangles[t])):
                self.covered[channel][t] = True
                self.witnesses[channel][int(t)] = witnesses.tolist()

    def complete(self, channel):
        return bool(np.all(self.covered[channel]))

    def needed_stations(self, channels):
        needed = set()
        for c in channels:
            remaining = self.indices[~self.covered[c]]
            for k in set(remaining.ravel().tolist()):
                if not any(np.linalg.norm(self.stations[k]-q) < 1e-7 for q in self.negative[c]):
                    needed.add(k)
        return sorted(needed)

    def certificate(self, channel):
        return dict(complete=self.complete(channel), triangles=len(self.triangles),
                    covered=int(self.covered[channel].sum()),
                    negative_points=[q.tolist() for q in self.negative[channel]],
                    triangle_witnesses=self.witnesses[channel])

