"""Joint service/coverage proposals; hypothetical scans never become evidence.

Scan locations live in intersections of equal-radius disks around residual
cell centres. Coordinate descent shortens the current open route, retaining
the previous feasible route as a fallback. This is a finite route heuristic,
not a global optimum or a probability model for the official source layout.
"""
from __future__ import annotations

import itertools
import numpy as np

import geometry as core
from scan_planning import build_scan_tasks
from v1_solver import rolling_route


def project_disks(target, centers, radius, fallback):
    """Euclidean projection by constraint generation in two dimensions.

    A boundary minimizer for finitely many disks is a radial projection on
    one circle or an intersection of two circles. Solve the active subset,
    add the most violated full constraint, and recheck all constraints. On
    numerical difficulty return the supplied, independently checked point.
    """
    target, centers, fallback = map(lambda x: np.asarray(x, float), (target, centers, fallback))
    if not len(centers):
        return target.copy()
    limit = radius + 1e-8
    if np.linalg.norm(centers - fallback, axis=1).max() > limit:
        raise ValueError('projection fallback is not feasible')
    active, q = [], target.copy()
    for _ in range(24):
        distances = np.linalg.norm(centers - q, axis=1)
        j = int(np.argmax(distances))
        if distances[j] <= limit:
            # Move minutely toward the feasible seed so squared-distance
            # masks do not lose tangent cells through rounding differences.
            for fraction in (1e-9, 1e-7, 1e-5, 1e-3):
                inside = (1-fraction)*q+fraction*fallback
                if np.max(np.sum((centers-inside)**2, axis=1)) <= radius**2:
                    return inside
            return fallback.copy()
        if j in active:
            break
        active.append(j)
        small = centers[active]
        candidates = [fallback]
        for c in small:
            delta = target-c
            d = np.linalg.norm(delta)
            candidates.append(c + delta * min(1., radius/max(d, 1e-15)))
        for a, b in itertools.combinations(small, 2):
            delta = b-a
            d = np.linalg.norm(delta)
            if d < 1e-10 or d > 2*radius:
                continue
            middle = (a+b)/2
            side = np.array([-delta[1], delta[0]])/d
            height = np.sqrt(max(0., radius*radius-d*d/4))
            candidates.extend((middle+height*side, middle-height*side))
        candidates = np.asarray(candidates)
        valid = np.max(np.linalg.norm(candidates[:, None]-small[None], axis=2), axis=1) <= limit
        choices = candidates[valid]
        if not len(choices):
            break
        q = choices[np.argmin(np.sum((choices-target)**2, axis=1))]
    return fallback.copy()


def shortest_visit(before, after, centers, radius, fallback):
    """Feasible candidate minimization of entry plus exit distance."""
    before = np.asarray(before, float)
    if after is None:
        return project_disks(before, centers, radius, fallback)
    after = np.asarray(after, float)
    cost = lambda q: float(np.linalg.norm(q-before)+np.linalg.norm(q-after))
    candidates = [np.asarray(fallback, float)]
    for fraction in (0., .5, 1.):
        candidates.append(project_disks((1-fraction)*before+fraction*after, centers, radius, fallback))
    best = min(candidates, key=cost)
    # Projected descent refines a feasible solution; only improving steps
    # are accepted, so an early stop cannot lose fallback feasibility/cost.
    for _ in range(8):
        a, b = best-before, best-after
        da, db = np.linalg.norm(a), np.linalg.norm(b)
        if min(da, db) < 1e-7:
            break
        direction = a/da+b/db
        if np.linalg.norm(direction) < 1e-7:
            break
        trial = project_disks(best-direction/(1/da+1/db), centers, radius, best)
        if cost(trial) >= cost(best)-1e-5:
            break
        best = trial
    return best


def residual_groups(tracker, channels, scans):
    if not channels:
        return [np.empty((0, 2)) for _ in scans]
    indices = [tracker._index(c) for c in channels]
    required = np.any(~tracker._excluded[indices], axis=0)
    centers = tracker.centers[required]
    if not len(centers):
        return [np.empty((0, 2)) for _ in scans]
    points = np.array([s['position'] for s in scans])
    masks = np.array([tracker._mask(q)[required] for q in points])
    if not len(points) or not np.all(np.any(masks, axis=0)):
        raise ValueError('scan route does not cover residual evidence')
    distances = np.sum((centers[:, None]-points[None])**2, axis=2)
    distances[~masks.T] = np.inf
    assignment = np.argmin(distances, axis=1)
    return [centers[assignment == i] for i in range(len(scans))]


def route_cost(start, route):
    if not route:
        return 0.
    points = np.vstack((start, [x['position'] for x in route]))
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def joint_route(tracker, channels, start, sources):
    scans = build_scan_tasks(tracker, channels, start, sources) if channels else []
    if not scans and not sources:
        return [], 0.
    route, _ = rolling_route(np.asarray(start), scans+sources)
    radius = tracker.signal_r-tracker.cell_half_diagonal-1e-9
    for _ in range(2):
        scans = [x for x in route if x['kind'] == 'scan']
        groups = residual_groups(tracker, channels, scans)
        group_by_key = {s['key']: g for s, g in zip(scans, groups)}
        updated = [dict(x) for x in route]
        for i, task in enumerate(updated):
            if task['kind'] != 'scan':
                continue
            group = group_by_key[task['key']]
            before = start if i == 0 else updated[i-1]['position']
            after = updated[i+1]['position'] if i+1 < len(updated) else None
            q = shortest_visit(before, after, group, radius, task['position'])
            task['position'] = q.tolist()
        ordered, _ = rolling_route(np.asarray(start), updated)
        if route_cost(start, ordered) > route_cost(start, updated):
            ordered = updated
        if route_cost(start, ordered) >= route_cost(start, route)-1e-6:
            break
        route = ordered
    # Reassigning through the real mask independently verifies union coverage.
    residual_groups(tracker, channels, [x for x in route if x['kind'] == 'scan'])
    return route, route_cost(start, route)


class ServiceContext:
    """Value of buying all unknown channels at one proposed service stop.

    The no-signal branch is a coverage-cost surrogate, not assumed feedback.
    A positive branch creates a new source task only after its actual reply.
    Each saved residual cell must be covered by the hypothetical stop or by
    a remaining planned scan. No CoverageTracker state is modified here.
    """
    def __init__(self, model, active):
        self.channels = model.unknown()
        self.tracker = model.coverage
        self.start = core.mec(model.beliefs[active]['P'])[0]
        sources = [dict(kind='source', key=c, position=core.mec(b['P'])[0].tolist())
                   for c, b in sorted(model.beliefs.items()) if c != active]
        self.route, _ = joint_route(self.tracker, self.channels, self.start, sources)
        scans = [x for x in self.route if x['kind'] == 'scan']
        groups = residual_groups(self.tracker, self.channels, scans)
        self.groups = {s['key']: g for s, g in zip(scans, groups)}
        self.radius = self.tracker.signal_r-self.tracker.cell_half_diagonal-1e-9
        self.reference = route_cost(self.start, self.route)/5 + 6*len(self.channels)*len(scans)

    def value(self, q):
        if not self.channels or not self.route:
            return 0., []
        eligible = [c for c in self.channels if self.tracker.channel_gain(q, c)]
        if not eligible:
            return 0., []
        route = []
        for task in self.route:
            if task['kind'] == 'scan':
                group = self.groups[task['key']]
                group = group[np.linalg.norm(group-q, axis=1) > self.radius]
                if not len(group):
                    continue
                task = dict(task, group=group)
            route.append(dict(task))
        for i, task in enumerate(route):
            if task['kind'] != 'scan':
                continue
            before = self.start if i == 0 else route[i-1]['position']
            after = route[i+1]['position'] if i+1 < len(route) else None
            # One exact projection is sufficient for ranking numerous local
            # proposals; the executed global plan uses shortest_visit.
            target = before if after is None else (np.asarray(before)+after)/2
            task['position'] = project_disks(target, task['group'], self.radius, task['position']).tolist()
        after_cost = route_cost(self.start, route)/5 + 6*len(self.channels)*sum(x['kind']=='scan' for x in route)
        saving = self.reference-after_cost-6*len(eligible)
        return max(0., saving), eligible if saving > 0 else []

    def candidates(self, P, s, witness, base):
        scans = [x for x in self.route if x['kind'] == 'scan']
        scans = sorted(scans, key=lambda x: np.linalg.norm(np.asarray(x['position'])-self.start))[:2]
        result = []
        for task in scans:
            group = self.groups[task['key']]
            for anchor in (self.start, base):
                q = project_disks(anchor, group, self.radius, task['position'])
                for fraction in (1., .5):
                    p = fraction*q+(1-fraction)*np.asarray(base)
                    if core.reception_certified(P, p, witness) and all(np.linalg.norm(p-old)>.1 for old in result):
                        result.append(p)
        return result
