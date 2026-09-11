"""Finite positive/negative scan rollouts for experimental Q3 ordering.

This changes only which already-planned scan is executed next.  It never
records a planned observation, alters a belief, or supplies a certificate.

Design prior (not an official layout distribution): N is uniform on 10..16,
channel subsets given N are uniform, positions are uniform-area cell-centre
quadrature, and R has equal mass at 1000, 1250, 1500 m. Accepted negative
observations condition this prior. Positive channels are known present.

Twelve common deterministic scenarios compare every possible first scan.
Negative branches retain a channel until real-style whole-cell coverage;
positive branches remove it from scanning and add a service visit. Service
visits use the scenario source position immediately after discovery. This
is an optimistic localization/travel proxy, not a realizable locator or an
accurate full-time forecast. Batch service, fixed future scan positions, and
finite quadrature are further approximations. Only the first scan survives;
the real solver replans after actual feedback.
"""
from __future__ import annotations

import math
import numpy as np

import bootstrap  # installs the frozen module import path
from v1_solver import rolling_route

SCENARIOS = 12
RADIUS_NODES = (1000., 1250., 1500.)


def _mask(tracker, point):
    # Do not even populate the tracker's mask cache while planning.
    radius = tracker.signal_r - tracker.cell_half_diagonal - 1e-9
    return np.sum((tracker.centers - point) ** 2, axis=1) <= radius ** 2


def _suffix_sums(weights):
    """Elementary symmetric sums for weighted fixed-cardinality subsets."""
    size = len(weights)
    result = np.zeros((size + 1, size + 1))
    result[:, 0] = 1.
    for i in range(size - 1, -1, -1):
        result[i, 1:] = result[i + 1, 1:] + weights[i] * result[i + 1, :-1]
    return result


def _worlds(model, channels):
    tracker = model.coverage
    in_disk = np.sum(tracker.centers ** 2, axis=1) <= tracker.target_r ** 2
    denominator = int(np.count_nonzero(in_disk)) * len(RADIUS_NODES)
    banks, weights = [], []
    for channel in channels:
        residual = ~tracker._excluded[tracker._index(channel)] & in_disk
        points = tracker.centers[residual]
        distances = [np.sum((points - np.asarray(q)) ** 2, axis=1)
                     for q in tracker._points[channel]]
        bank = []
        for radius in RADIUS_NODES:
            keep = np.ones(len(points), dtype=bool)
            for squared in distances:
                keep &= squared > radius ** 2
            bank.append(points[keep])
        total = sum(len(points) for points in bank)
        weights.append(total / max(denominator, 1))
        banks.append(bank)

    weights = np.asarray(weights)
    suffix = _suffix_sums(weights)
    discovered = len(model.discovered)
    counts = np.arange(len(channels) + 1)
    mass = np.array([suffix[0, k] / math.comb(20, discovered + int(k))
                     if 10 <= discovered + k <= 16 else 0. for k in counts])
    if not np.isfinite(mass).all() or mass.sum() <= 0:
        # The finite prior can miss a thin feasible set. It cannot overrule
        # the real geometric model; fall back to the existing scan order.
        return None
    mass /= mass.sum()
    cumulative = np.cumsum(mass)
    random = np.random.default_rng(13016020)
    worlds = []
    for scenario in range(SCENARIOS):
        remaining = int(np.searchsorted(cumulative, (scenario + .5) / SCENARIOS))
        world = {}
        for i, channel in enumerate(channels):
            if remaining == 0:
                break
            divisor = suffix[i, remaining]
            probability = (weights[i] * suffix[i + 1, remaining - 1] / divisor
                           if divisor > 0 else 0.)
            if random.random() >= min(1., probability):
                continue
            remaining -= 1
            bank = banks[i]
            index = int(random.integers(sum(len(points) for points in bank)))
            for radius, points in zip(RADIUS_NODES, bank):
                if index < len(points):
                    world[channel] = (points[index].copy(), radius)
                    break
                index -= len(points)
        if remaining:
            return None
        worlds.append(world)
    return worlds


def _batch_visit(start, points, next_point):
    """Small deterministic service tour; hypothetical source visits only."""
    if not points:
        return np.asarray(start), 0.
    points = np.asarray(points)
    distance = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    starts = np.argsort(np.linalg.norm(points - start, axis=1))[:3]
    best = None
    for first in starts:
        path, remaining = [int(first)], set(range(len(points))) - {int(first)}
        while remaining:
            index = min(remaining, key=lambda j: (distance[path[-1], j], j))
            path.append(index)
            remaining.remove(index)
        travel = float(np.linalg.norm(points[path[0]] - start))
        travel += sum(float(distance[a, b]) for a, b in zip(path, path[1:]))
        onward = (float(np.linalg.norm(points[path[-1]] - next_point))
                  if next_point is not None else 0.)
        rank = travel + onward
        if best is None or rank < best[0]:
            best = rank, points[path[-1]], travel
    return best[1], best[2]


def _rollout(start, order, points, gains, channels, world, already_found):
    active = np.ones(len(channels), dtype=bool)
    cursor = np.asarray(start).copy()
    measured, distance, found = 0, 0., already_found
    for step, index in enumerate(order):
        if found == 16:
            break
        eligible = active & gains[step]
        if not np.any(eligible):
            continue
        q = points[index]
        distance += float(np.linalg.norm(q - cursor))
        cursor = q.copy()
        discovered = []
        for i in np.flatnonzero(eligible):
            if found == 16:
                break
            measured += 1
            channel = channels[i]
            source = world.get(channel)
            if source is not None and np.linalg.norm(source[0] - q) <= source[1]:
                active[i] = False
                discovered.append(source[0])
                found += 1
        next_point = None
        if found < 16:
            for later, future in enumerate(order[step + 1:], start=step + 1):
                if np.any(active & gains[later]):
                    next_point = points[future]
                    break
        cursor, travel = _batch_visit(cursor, discovered, next_point)
        distance += travel
    # Six seconds per scan is a scan/switch proxy; actual receiver ordering
    # can save switches. Clearing/localization fixed terms are not ranked.
    return distance / 5. + 6. * measured


def rank_scan_tasks(model, tasks, route):
    """Return (proxy seconds, original task) pairs, or [] when inapplicable."""
    if (not route or route[0]['kind'] != 'scan' or model.beliefs
            or any(task['kind'] != 'scan' for task in tasks) or len(tasks) < 2):
        return []
    channels = model.unknown()
    if not channels:
        return []
    points = np.asarray([task['position'] for task in tasks], dtype=float)
    if points.shape != (len(tasks), 2) or not np.isfinite(points).all():
        raise ValueError('scan points must be finite two-dimensional points')
    tracker = model.coverage
    residual = ~tracker._excluded[[tracker._index(c) for c in channels]]
    masks = np.asarray([_mask(tracker, q) for q in points])
    if not np.all(np.any(masks, axis=0) | ~np.any(residual, axis=0)):
        return []  # A partial task list is unsuitable for a completion rollout.
    worlds = _worlds(model, channels)
    if worlds is None:
        return []
    start = np.asarray(model.position, dtype=float)
    ranks = []
    for first, task in enumerate(tasks):
        rest = [dict(item, preview_index=i) for i, item in enumerate(tasks) if i != first]
        tail, _ = rolling_route(points[first], rest) if rest else ([], 0.)
        order = [first] + [item['preview_index'] for item in tail]
        # Every still-active channel has been negative at its earlier scans;
        # precompute that branch's whole-cell gains once for all scenarios.
        remaining = residual.copy()
        gains = []
        for index in order:
            gains.append(np.any(remaining & masks[index], axis=1))
            remaining &= ~masks[index]
        costs = [_rollout(start, order, points, gains, channels, world,
                          len(model.discovered)) for world in worlds]
        ranks.append((float(np.mean(costs)), task))
    return sorted(ranks, key=lambda entry: entry[0])


def choose_scan_task(model, tasks, route):
    """Choose an existing scan task without modifying any public evidence."""
    ranked = rank_scan_tasks(model, tasks, route)
    if not ranked:
        return None
    best = ranked[0][1]
    return best if best['key'] != route[0]['key'] else None
