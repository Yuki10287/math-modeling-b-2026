"""Local sensing and complete optical cover with a shared continuation proxy.

Scenario masses rank actions; neither these masses nor the continuation proxy
certify a source position. The full rectangular optical cover is retained.
"""
import math
import numpy as np
from localization import choose_measure, optical_plan, samples
from shared import core


def _continuations(points):
    points = np.asarray(points, dtype=float)
    if not points.size:
        return np.empty((0, 2))
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('continuation_points must be finite two-dimensional points')
    return points


def _continuation_s(points, continuation):
    """Movement to the nearest pending task, with no return-to-start charge."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if not len(continuation):
        return np.zeros(len(points))
    return np.linalg.norm(points[:, None, :] - continuation, axis=2).min(axis=1) / 5


def _context_optical(P, start, continuation, quadrature=None):
    if not len(continuation):
        return optical_plan(P, start, quadrature)
    _, (i, j) = core.diameter(P)
    u = P[j] - P[i]
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-10 else np.array([1., 0.])
    rotation = np.column_stack((u, [-u[1], u[0]]))
    origin = P.mean(axis=0)
    local = (P - origin) @ rotation
    lo, hi = local.min(axis=0), local.max(axis=0)
    side = math.sqrt(2) * (20 - 1e-4)
    nx, ny = np.maximum(1, np.ceil((hi - lo) / side).astype(int))
    x = lo[0] + (np.arange(nx) + .5) * (hi[0] - lo[0]) / nx
    y = lo[1] + (np.arange(ny) + .5) * (hi[1] - lo[1]) / ny
    points = samples(P) if quadrature is None or not len(quadrature) else quadrature
    best = None
    for ys in (y, y[::-1]):
        for reverse in (False, True):
            line = []
            for k, yy in enumerate(ys):
                xs = x[::-1] if bool(k % 2) ^ reverse else x
                line.extend((xx, yy) for xx in xs)
            path = origin + np.array(line) @ rotation.T
            cumulative = np.cumsum(np.linalg.norm(
                np.diff(np.vstack((start, path)), axis=0), axis=1)) / 5
            hits = np.linalg.norm(points[:, None, :] - path, axis=2) <= 20
            if not hits.any(axis=1).all():
                raise RuntimeError('optical rectangle failed sample sanity check')
            first = hits.argmax(axis=1)
            costs = cumulative[first] + 3 * first + 5
            mean = float(costs.mean())
            worst = float(cumulative[-1] + 3 * (len(path) - 1) + 5)
            onward = _continuation_s(path, continuation)
            onward_mean = float(onward[first].mean())
            onward_worst = float(onward[-1])
            # Prefix travel plus onward travel cannot exceed full-path travel
            # plus its onward leg (triangle inequality). The .8/.2 weighting
            # is the existing design heuristic, not a calibrated probability.
            local_score = .8 * mean + .2 * worst
            continuation_s = .8 * onward_mean + .2 * onward_worst
            score = local_score + continuation_s
            if best is None or score < best['score']:
                best = dict(path=path, score=score, local_score=local_score,
                            continuation_s=continuation_s,
                            continuation_mean_s=onward_mean,
                            continuation_worst_s=onward_worst,
                            worst_s=worst, mean_s=mean, origin=origin,
                            rotation=rotation, lower=lo, upper=hi,
                            cells=[int(nx), int(ny)],
                            cell_radius=float(np.linalg.norm((hi-lo)/[nx, ny])/2))
    return best


def choose_context_optical(belief, start, continuation_points):
    """Return a complete optical plan; score includes the same onward leg as sensing."""
    return _context_optical(belief.P, np.asarray(start, dtype=float),
                            _continuations(continuation_points))


def _candidates(belief, start, continuation):
    center, radius = core.mec(belief.P)
    _, (i, j) = core.diameter(belief.P)
    u = belief.P[j] - belief.P[i]
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-9 else np.array([1., 0.])
    v = np.array([-u[1], u[0]])
    candidates = [center, (center + start) / 2]
    candidates.extend(center + along*u + offset*v for along in (0., -.35*radius)
                      for offset in (-100., -40., 40., 100.))
    # A projection onto the next-task segment and one compromise view add at
    # most two candidates. Their value is assessed by the ordinary scenarios.
    target = continuation[np.argmin(np.linalg.norm(continuation-start, axis=1))]
    direction = target - start
    length2 = float(direction @ direction)
    fraction = float(np.clip((center-start) @ direction / length2, 0, 1)) if length2 else 0.
    projection = start + fraction*direction
    candidates.extend((projection, (projection+center)/2))
    unique = []
    for q in candidates:
        if any(np.linalg.norm(q-p) <= .1 for p in belief.measured):
            continue
        if not any(np.linalg.norm(q-p) <= 1e-7 for p in unique):
            unique.append(q)
    return unique


def choose_context_measure(belief, start, channel, current_channel, continuation_points):
    """Choose a paid view with optical and onward movement in every outcome.

    Only observations already in ``belief`` and public pending-task positions
    enter this calculation. A negative branch retains the entire polygon.
    """
    continuation = _continuations(continuation_points)
    start = np.asarray(start, dtype=float)
    if not len(continuation):
        return choose_measure(belief, start, channel, current_channel)
    gs, scenarios = belief.scenarios()
    if not scenarios:
        return None
    total_weight = sum(row[4] for row in scenarios)
    best = None
    candidates = _candidates(belief, start, continuation)
    for q in candidates:
        hit_weights = np.zeros(len(gs))
        for index, g, n, radius, weight in scenarios:
            if np.linalg.norm(q-g) <= radius and (n is None or (q-g) @ n >= 0):
                hit_weights[index] += weight/total_weight
        probability = float(hit_weights.sum())
        if probability < .05:
            continue
        negative = _context_optical(belief.P, q, continuation, gs)
        positive_cost = positive_onward = 0.
        for k in np.flatnonzero(hit_weights):
            g = gs[k]
            if np.linalg.norm(g-q) <= 5:
                onward = float(_continuation_s([q], continuation)[0])
                cost = 5. + onward
            else:
                angle = math.degrees(math.atan2(g[1]-q[1], g[0]-q[0])) % 360
                values, onward_values = [], []
                for error in (-1., 0., 1.):
                    bearing = round((angle+error) % 360, 2) % 360
                    P = core.disk_clip(core.wedge(belief.P, q, bearing), q)
                    if not len(P):
                        raise RuntimeError('empty predictive branch')
                    tail = _context_optical(P, q, continuation)
                    values.append(tail['score'])
                    onward_values.append(tail['continuation_s'])
                cost, onward = float(np.mean(values)), float(np.mean(onward_values))
            positive_cost += hit_weights[k]*cost
            positive_onward += hit_weights[k]*onward
        immediate = float(np.linalg.norm(q-start))/5 + 5 + int(channel != current_channel)
        score = immediate + positive_cost + (1-probability)*negative['score']
        if best is None or score < best['score']:
            continuation_s = positive_onward + (1-probability)*negative['continuation_s']
            best = dict(q=q, score=score, signal_mass=probability,
                        scenarios=len(scenarios), continuation_s=continuation_s,
                        local_score=score-continuation_s,
                        candidate_count=len(candidates))
    return best
