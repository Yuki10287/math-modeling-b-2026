"""Local Q3 action model: area-risk rollout and certified optical partition.

All decisions use a convex outer belief and public observations only.  Area
sampling is an explicit design prior, not a claim about the official generator.
It changes ranking, never feasibility or the completion certificate.
"""
from __future__ import annotations

import math
import numpy as np
import geometry as core


RADIUS = 20.0 - 1e-5


class OpticalCoverExhausted(RuntimeError):
    """Every disk in a proven cover failed: feedback contradicts the belief."""


def _poly(P):
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1:] != (2,) or not len(P) or not np.isfinite(P).all():
        raise ValueError('belief must be a finite nonempty n by 2 array')
    return P


def _area(P):
    if len(P) < 3:
        return 0.0
    return abs(float(np.dot(P[:, 0], np.roll(P[:, 1], -1))
                     - np.dot(P[:, 1], np.roll(P[:, 0], -1)))) / 2


def area_scenarios(P, count=12):
    """Deterministic equal-area quadrature over a convex polygon.

    No vertices are treated as equally probable with the much larger interior.
    Degenerate polygons use length-uniform points on the diameter instead.
    """
    P = _poly(P)
    center = P.mean(axis=0)
    triangles = [(center, P[i], P[(i + 1) % len(P)]) for i in range(len(P))]
    masses = np.array([abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])) / 2
                       for a, b, c in triangles])
    if masses.sum() < 1e-7:
        _, (i, j) = core.diameter(P)
        f = (np.arange(count) + .5) / count
        return (1 - f[:, None]) * P[i] + f[:, None] * P[j]
    cumulative = np.cumsum(masses) / masses.sum()
    points = []
    for k in range(count):
        z = (k + .5) / count
        index = min(int(np.searchsorted(cumulative, z)), len(P) - 1)
        before = 0 if index == 0 else cumulative[index - 1]
        frac = (z - before) / max(cumulative[index] - before, 1e-15)
        a, b, c = triangles[index]
        u = math.sqrt(float(np.clip(frac, 0, 1)))
        v = ((k + 1) * .6180339887498949) % 1
        points.append((1 - u) * a + u * ((1 - v) * b + v * c))
    return np.array(points)


def partition_cover(P, *, max_parts=32):
    """Partition a tight oriented rectangle, with recursive safety fallback.

    Choose a rectangular tiling with cell half diagonal <= RADIUS.  Intersect
    every cell with P, then fit its actual MEC.  This avoids needless powers-of-
    two subdivisions of a long narrow region.  Every retained
    piece is contained in its MEC, so covering the vertices covers its whole
    convex interior.  This is a continuous cover, not a source sample test.
    """
    P = _poly(P)
    _, (i, j) = core.diameter(P)
    axes = [np.array([1., 0.]), P[j] - P[i]]
    axes.extend(np.roll(P, -1, axis=0) - P)
    candidates = []
    for axis in axes:
        length = np.linalg.norm(axis)
        if length < 1e-9:
            continue
        u = axis / length
        v = np.array([-u[1], u[0]])
        x, y = P @ u, P @ v
        L, W = float(np.ptp(x)), float(np.ptp(y))
        for nx in range(1, max_parts + 1):
            dx = L / nx / 2
            if dx >= RADIUS:
                continue
            max_y = 2 * math.sqrt(RADIUS**2 - dx**2)
            ny = max(1, int(math.ceil(W / max_y)))
            if nx * ny <= max_parts:
                # Cell count controls failed clear overhead; aspect term breaks
                # ties in favor of a snake with short transverse movement.
                rank = (nx * ny, (ny - 1) * L + (nx - 1) * W)
                candidates.append((rank, u, v, x.min(), x.max(), y.min(), y.max(), nx, ny))
    if candidates:
        candidate = min(candidates, key=lambda x: x[0])
        _, u, v, xmin, xmax, ymin, ymax, nx, ny = candidate
        xs, ys = np.linspace(xmin, xmax, nx + 1), np.linspace(ymin, ymax, ny + 1)
        leaves = []
        for ix in range(nx):
            for iy in range(ny):
                part = P.copy()
                for n, b in ((u, xs[ix + 1]), (-u, -xs[ix]), (v, ys[iy + 1]), (-v, -ys[iy])):
                    part = core.clip(part, n, float(b))
                if len(part):
                    c, r = core.mec(part)
                    if r > RADIUS + 1e-7:
                        raise ValueError('rectangular optical cover exceeds its cell certificate')
                    leaves.append((part, c, r))
        return leaves
    pending, leaves = [P], []
    while pending:
        part = pending.pop()
        c, r = core.mec(part)
        if r <= RADIUS:
            leaves.append((part, c, r))
            continue
        if len(pending) + len(leaves) + 2 > max_parts:
            return None
        diameter, (i, j) = core.diameter(part)
        if diameter < 1e-9:
            raise ValueError('nonprogressing optical partition')
        axis = (part[j] - part[i]) / diameter
        midpoint = float(axis @ ((part[j] + part[i]) / 2))
        left = core.clip(part, axis, midpoint)
        right = core.clip(part, -axis, -midpoint)
        if not len(left) or not len(right):
            raise ValueError('empty optical partition')
        pending.extend([right, left])
    return leaves


def _near_cover(part, start):
    q = core.nearest_certified_clear(part, start)
    if q is None:
        raise ValueError('partition leaf exceeds the clearance radius')
    if np.max(np.linalg.norm(part - q, axis=1)) > 20 - 1e-7:
        raise ValueError('invalid optical cover certificate')
    return q


def _cover_route(leaves, start, order):
    path, previous = [], np.asarray(start, float)
    for i in order:
        q = _near_cover(leaves[i][0], previous)
        path.append(q)
        previous = q
    return np.array(path)


def cover_is_complete(P, path, radius=20.0):
    """Independent exact polygon/Voronoi coverage certificate.

    On each disk center's restricted Voronoi cell, that center is nearest.
    Squared distance is convex and its maximum on a convex polygon is attained
    at a vertex.  Therefore all such vertices in their nearest disk imply every
    point of P lies in the union of disks.
    """
    P = _poly(P)
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1:] != (2,) or not len(path) or not np.isfinite(path).all():
        return False
    for i, q in enumerate(path):
        cell = P.copy()
        for j, other in enumerate(path):
            if i != j:
                cell = core.clip(cell, 2 * (other - q), float(other @ other - q @ q))
            if not len(cell):
                break
        if len(cell) and np.max(np.linalg.norm(cell - q, axis=1)) > radius + 1e-7:
            return False
    return True


def optical_cost(path, start, samples):
    points = np.vstack((start, path))
    cumulative = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)) / 5
    costs = []
    for g in samples:
        hits = np.flatnonzero(np.linalg.norm(path - g, axis=1) <= 20 + 1e-7)
        if not len(hits):
            raise ValueError('optical path misses a belief sample')
        j = int(hits[0])
        costs.append(float(cumulative[j] + 3 * j + 5))
    return np.array(costs), float(cumulative[-1] + 3 * (len(path) - 1) + 5)


def best_optical_plan(P, s, *, risk=.2, max_parts=16):
    leaves = partition_cover(P, max_parts=max_parts)
    if leaves is None:
        return None
    _, (i, j) = core.diameter(P)
    axis = P[j] - P[i]
    centers = np.array([x[1] for x in leaves])
    sorted_order = np.argsort(centers @ axis).tolist()
    orders = [sorted_order, sorted_order[::-1]]
    # A near-first visit usually clears sooner under the explicit area prior.
    for first in np.argsort(np.linalg.norm(centers - s, axis=1))[:min(3, len(leaves))]:
        order = [int(first)]
        rest = set(range(len(leaves))) - set(order)
        while rest:
            nxt = min(rest, key=lambda k: (np.linalg.norm(centers[k] - centers[order[-1]]), k))
            order.append(nxt)
            rest.remove(nxt)
        orders.append(order)
    samples = area_scenarios(P, 24)
    best = None
    for order in orders:
        path = _cover_route(leaves, s, order)
        variants = [path]
        if np.any(np.linalg.norm(samples - s, axis=1) <= 20) and np.linalg.norm(path[0] - s) > 1e-7:
            variants.append(np.vstack((s, path)))
        for path in variants:
            values, worst = optical_cost(path, s, samples)
            score = (1 - risk) * float(values.mean()) + risk * worst
            rank = (score, worst, len(path))
            if best is None or rank < best[0]:
                best = (rank, path, values)
    rank, path, values = best
    if not cover_is_complete(P, path):
        raise ValueError('independent Voronoi cover verification failed')
    return dict(path=path, score=rank[0], worst_s=rank[1], mean_s=float(values.mean()),
                parts=len(path), certified=True)


def _measure_score(P, s, q, samples, risk=.2):
    values = []
    for g in samples:
        delta = g - q
        if np.linalg.norm(delta) <= 5:
            values.append(5.)
            continue
        bearing = math.degrees(math.atan2(delta[1], delta[0]))
        error_values = []
        for error in (-1., 0., 1.):
            next_P = core.wedge(P, q, bearing + error)
            if not len(next_P):
                raise ValueError('empty predicted observation')
            # Original rollout retained: design change is probability/risk
            # aggregation, so measured improvement can be attributed clearly.
            error_values.append(core.continuation_cost(next_P, q, g))
        values.append(float(np.mean(error_values)))
    movement = float(np.linalg.norm(q - s)) / 5 + 5
    return movement + (1 - risk) * float(np.mean(values)) + risk * float(np.max(values))


def _conservative_measure(P, s, options):
    """Original baseline minimax proxy over an already-filtered candidate set."""
    samples = core.scenario_points(P)
    best = None
    for q in options:
        worst = 0.
        for g in samples:
            delta = g - q
            if np.linalg.norm(delta) <= 5:
                worst = max(worst, 5.)
                continue
            bearing = math.degrees(math.atan2(delta[1], delta[0]))
            for error in (-1., 0., 1.):
                next_P = core.wedge(P, q, bearing + error)
                if not len(next_P):
                    raise ValueError('empty predicted observation')
                worst = max(worst, core.continuation_cost(next_P, q, g))
        movement = float(np.linalg.norm(q - s)) / 5 + 5
        rank = (float(worst + movement), movement)
        if best is None or rank < best[0]:
            best = (rank, q)
    if best is None:
        raise ValueError('no unused reception-certified candidate')
    return best[1], best[0][0]


def short_candidates(P, s, witness, options=None):
    """Add optical-scale lateral baselines to range-scaled candidates.

    20/40/80 m are one/two/four optical radii.  A large radial uncertainty does
    not by itself require all cross-range movement to be hundreds of meters.
    These are extra candidates only: the same time model ranks every action,
    and the same hard reception certificate filters them before execution.
    """
    options = list(core.candidates(P, s, witness) if options is None else options)
    c, r = core.mec(P)
    _, (i, j) = core.diameter(P)
    axis = P[j] - P[i]
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    perp = np.array([-axis[1], axis[0]])
    for shift in (20., 40., 80.):
        if shift >= .3 * r:
            continue
        for sign in (-1., 1.):
            for along in (0., -.25 * r, .25 * r):
                q = c + along * axis + sign * shift * perp
                if np.linalg.norm(q - s) > .1 and core.reception_certified(P, q, witness):
                    if all(np.linalg.norm(q - p) > .1 for p in options):
                        options.append(q)
    return options


def choose_action(P, s, witness, channel, *, current_channel=None, state=None,
                  mode='hybrid', risk=.2, optical_max_radius=180.0, observed_positions=None,
                  candidate_mode='standard'):
    """Return kind/q/rationale and next state; never call the environment.

    Pass returned state into the next call after an unsuccessful optical clear.
    For measure feedback, discard state and update P with the bounded bearing.
    A whole chosen optical path remains committed and cannot cycle on a failure.
    The optional optical mode keeps the original conservative measure ranking.
    """
    if mode not in ('hybrid', 'optical', 'baseline'):
        raise ValueError('unknown local mode')
    if candidate_mode not in ('standard', 'short'):
        raise ValueError('unknown candidate mode')
    P, s, witness = _poly(P), np.asarray(s, float), np.asarray(witness, float)
    if s.shape != (2,) or witness.shape != (2,) or not np.isfinite([s, witness]).all():
        raise ValueError('invalid local position')
    if state is not None and state.get('exhausted'):
        raise OpticalCoverExhausted('all disks in the certified optical cover failed')
    if state is not None and state.get('remaining'):
        path = state['remaining']
        return dict(kind='clear', q=np.array(path[0]), rationale='continue_certified_optical_cover',
                    state=dict(remaining=path[1:], cover_size=state['cover_size'], exhausted=len(path) == 1),
                    cover_size=state['cover_size'], certified=not path[1:])
    q = core.nearest_certified_clear(P, s)
    if q is not None:
        return dict(kind='clear', q=q, rationale='whole_belief_in_one_disk', state=None, certified=True)
    observed = [witness] + ([] if observed_positions is None else [np.asarray(q, float) for q in observed_positions])
    if any(q.shape != (2,) or not np.isfinite(q).all() for q in observed):
        raise ValueError('invalid historical measurement position')
    used = lambda q: any(np.linalg.norm(q - old) <= .1 for old in observed)
    options = core.candidates(P, s, witness)
    if candidate_mode == 'short':
        options = short_candidates(P, s, witness, options)
    options = [q for q in options if not used(q)]
    if mode == 'hybrid':
        # The operator may have moved while servicing other channels.  A fresh
        # bearing here has zero movement cost.  The old candidate set excluded
        # all zero-movement observations even when this channel was never read
        # here.  witness prevents repeating the latest fixed-error observation.
        if not used(s) and core.reception_certified(P, s, witness):
            options.insert(0, s.copy())
        if not options:
            raise ValueError('no unused reception-certified candidate')
        samples = area_scenarios(P)
        ranked = [(float(_measure_score(P, s, q, samples, risk)), i, q) for i, q in enumerate(options)]
        measure_score, _, measure_q = min(ranked, key=lambda x: (x[0], x[1]))
    else:
        measure_q, measure_score = _conservative_measure(P, s, options)
        samples = core.scenario_points(P)
        if mode == 'optical':
            measure_score = _measure_score(P, s, measure_q, samples, risk=1.)
    measure_score += int(current_channel is not None and current_channel != channel)
    if mode != 'baseline' and core.mec(P)[1] <= optical_max_radius:
        plan = best_optical_plan(P, s, risk=risk)
        if plan is not None and plan['score'] <= measure_score:
            path = plan['path']
            return dict(kind='clear', q=path[0], rationale='certified_optical_partition_cheaper',
                        state=dict(remaining=path[1:].tolist(), cover_size=len(path), exhausted=len(path) == 1),
                        cover_size=len(path), certified=len(path) == 1,
                        optical_path=path.tolist(),
                        estimated_s=plan['score'], worst_cover_s=plan['worst_s'],
                        alternative_measure_s=measure_score)
    return dict(kind='measure', q=measure_q, rationale='area_risk_rollout' if mode == 'hybrid' else 'baseline_time_rollout',
                state=None, estimated_s=measure_score)
