"""Persistent Q3 outer beliefs, including conservative negative feedback."""
from __future__ import annotations
import math
import numpy as np
import geometry as core
from coverage_model import CoverageTracker
from recovery import _measure_feedback, _clear_feedback


def convex_hull(points):
    points = sorted(set(map(tuple, np.asarray(points, float))))
    if len(points) < 3:
        return np.asarray(points, float).reshape(-1, 2)
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1]+upper[:-1])


def exclude_disk_hull(P, q, radius, sides=32):
    """Outer relaxation of P minus a disk, preserving ALL remaining branches.

    Subtract an inscribed polygon (strictly smaller than the excluded disk),
    split along every face, retain every outside piece, then take their hull.
    Holes may be lost; no still-feasible component is selected away.
    """
    P, q = np.asarray(P, float), np.asarray(q, float)
    c, r = core.mec(P)
    if np.linalg.norm(c-q) >= radius+r+1e-6:
        return P.copy()
    remaining, parts = P.copy(), []
    inset = radius*math.cos(math.pi/sides)-1e-7
    for angle in np.arange(sides)*2*math.pi/sides:
        n = np.array([math.cos(angle), math.sin(angle)])
        b = float(n@q+inset)
        outside = core.clip(remaining, -n, -b)
        if len(outside):
            parts.extend(outside)
        remaining = core.clip(remaining, n, b)
        if not len(remaining):
            break
    return convex_hull(parts) if parts else np.empty((0, 2))


class FeedbackModel:
    """The solver's only environment access is this four-member public API."""
    def __init__(self, api, trace, *, use_negatives=True, range_cuts=False, search_fraction=.15):
        self.api, self.trace = api, trace
        self.coverage = CoverageTracker()
        self.beliefs, self.cleared, self.discovered = {}, set(), set()
        self.positions = {c: [] for c in range(1, 21)}
        self.use_negatives = use_negatives
        self.range_cuts = range_cuts
        self.search_fraction = search_fraction

    def _range_cut(self, P, negative, positive):
        # Fixed unknown R: ||g-positive|| <= R < ||g-negative||.
        # Squaring cancels g@g; closing the strict half-plane is conservative.
        q, w = np.asarray(negative), np.asarray(positive)
        return core.clip(P, 2*(q-w), float(q@q-w@w))

    @property
    def position(self):
        return self.api.position

    @property
    def channel(self):
        return self.api.channel

    def unknown(self):
        if len(self.discovered) == 16:
            # Distinct positive channels already attain the stated upper bound;
            # the other channels cannot contain an additional source.
            return []
        return [c for c in range(1, 21) if c not in self.discovered
                and not self.coverage.complete(c)]

    def recorded(self, channel, q):
        return any(np.linalg.norm(p-q) <= .1 for p in self.positions[channel])

    def _save(self, channel, P, reason):
        item = self.beliefs[channel]
        if not len(P) or not np.isfinite(P).all():
            item['conflict'] = reason
            self.trace.append(dict(phase='belief_conflict', channel=channel, reason=reason))
            return
        item['P'] = P
        self.trace.append(dict(phase='belief_update', channel=channel, reason=reason,
                               position=np.asarray(self.position).tolist(), polygon=P.tolist()))

    def measure(self, q, channel):
        q = np.asarray(q, float)
        result = self.api.measure(q, channel)
        kind = _measure_feedback(result)  # rejected/malformed actions never become evidence
        if channel in self.cleared:
            # Post-clear silence says nothing about the source's former range.
            return result
        self.positions[channel].append(q.copy())
        if kind == 'no_signal':
            self.coverage.observe(channel, q)
            if self.use_negatives and channel in self.beliefs:
                P = exclude_disk_hull(self.beliefs[channel]['P'], q, 1000.)
                if self.range_cuts:
                    for w in self.beliefs[channel]['positives']:
                        P = self._range_cut(P, q, w)
                self._save(channel, P, 'no_signal_minimum_radius')
            return result
        self.discovered.add(channel)
        if channel not in self.beliefs:
            P = (core.disk_clip(core.circle_polygon(), q, 5.) if kind == 'near'
                 else core.initial_belief(q, result['svd_deg']))
            self.beliefs[channel] = dict(P=P, anchor=q.copy(), first=result.copy(),
                witness=q.copy(), state=None, near=q.copy() if kind == 'near' else None,
                conflict=None, positives=[q.copy()])
            if self.use_negatives:
                for p in self.coverage.certificate(channel)['negative_points']:
                    P = exclude_disk_hull(P, p, 1000.)
                    if self.range_cuts:
                        P = self._range_cut(P, p, q)
                    if not len(P):
                        break
        else:
            item = self.beliefs[channel]
            P = item['P']
            P = (core.disk_clip(P, q, 5.) if kind == 'near'
                 else core.disk_clip(core.wedge(P, q, result['svd_deg']), q))
            item['witness'] = q.copy()
            item['positives'].append(q.copy())
            item['near'] = q.copy() if kind == 'near' else item['near']
            # A new positive bearing makes the old optical plan unnecessary.
            item['state'] = None
            if self.range_cuts:
                for p in self.coverage.certificate(channel)['negative_points']:
                    P = self._range_cut(P, p, q)
        self._save(channel, P, kind)
        return result

    def clear(self, q, channel):
        q = np.asarray(q, float)
        result = self.api.clear(q, channel)
        kind = _clear_feedback(result)
        if kind == 'success':
            self.cleared.add(channel)
            self.beliefs.pop(channel, None)
        elif self.use_negatives and channel in self.beliefs:
            self._save(channel, exclude_disk_hull(self.beliefs[channel]['P'], q, 20.),
                       'failed_optical_clear')
        return result

    def certificate(self):
        absent = [c for c in range(1, 21) if c not in self.discovered
                  and self.coverage.complete(c)]
        count = len(self.cleared) == 16
        covered = len(self.cleared)+len(absent) == 20
        absent_by_count = sorted(set(range(1, 21))-self.discovered) if len(self.discovered) == 16 else []
        return dict(valid=count or covered,
            basis='count_upper_bound' if count else 'adaptive_coverage' if covered else 'incomplete',
            cleared_channels=sorted(self.cleared), absent_channels=absent,
            absent_by_count_channels=absent_by_count,
            unresolved_channels=sorted(set(range(1, 21))-self.cleared-set(absent)-set(absent_by_count)),
            channel_coverage={str(c): self.coverage.certificate(c) for c in range(1, 21)},
            assumptions='Q3; fixed omnidirectional sources; unique channels; target disk 1800 m; R in [1000,1500]; at most 16 sources')
