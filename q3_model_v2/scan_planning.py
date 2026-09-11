"""Residual-set covering scan plans using only accepted public evidence.

The plan covers the union of every unknown channel's residual grid cells.
It is a prospective route, NOT evidence that the scans have happened. The
caller must measure unknown channels at the chosen locations and update its
CoverageTracker exclusively from accepted feedback.

Model: finite candidate set cover with a joint source/scan route cost. We
construct movable scan stations by assigning residual cells to one of the
original seven covering stations, then moving that station toward a route
anchor while retaining whole-cell coverage of its assigned group. Original
stations remain candidates and a feasible fallback. No source truth is read.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from v1_solver import rolling_route


def _fixed_stations() -> np.ndarray:
    return np.vstack(([0., 0.], [[1200. * math.cos(k * math.pi / 3),
                                 1200. * math.sin(k * math.pi / 3)] for k in range(6)]))


def _project_group(station, anchor, centers, inset_radius):
    """Move along a segment within the intersection of cell-centred disks.

    A candidate q covers a complete cell if ||q-cell_center|| <= inset_radius.
    The segment starts at a known feasible station, so every quadratic
    constraint allows t in [0, t_max]. Their smallest t_max is a certificate.
    """
    delta = np.asarray(anchor, dtype=float) - station
    quadratic = float(np.dot(delta, delta))
    if quadratic < 1e-18 or len(centers) == 0:
        return station.copy()
    offsets = station - centers
    linear = 2 * (offsets @ delta)
    constant = np.sum(offsets * offsets, axis=1) - inset_radius ** 2
    if np.max(constant) > 1e-5:
        raise ValueError("group station does not cover its assigned cells")
    discriminant = np.maximum(linear * linear - 4 * quadratic * constant, 0.)
    upper = (-linear + np.sqrt(discriminant)) / (2 * quadratic)
    t = min(1., max(0., float(np.min(upper))))
    # Retreat a tiny amount toward the feasible start for floating-point slack.
    return station + max(0., t - 1e-10) * delta


def _task(q) -> dict:
    q = np.asarray(q, dtype=float)
    return dict(kind="scan", key=f"cover:{q[0]:.9f}:{q[1]:.9f}", position=q.tolist())


def build_scan_tasks(tracker, unknown_channels: Iterable[int], current_position,
                     source_tasks: list[dict]) -> list[dict]:
    """Return <= 7 prospective scans covering all remaining channel/cells.

    Candidate selection is a heuristic. We explicitly verify candidate-mask
    union coverage and compare its route proxy against the fixed-station
    fallback before returning it. Source locations are existing belief proxy
    positions, not actual source positions. Real future cost may differ.
    """
    channels = tuple(sorted(set(int(c) for c in unknown_channels)))
    if not channels:
        return []
    position = np.asarray(current_position, dtype=float)
    if position.shape != (2,) or not np.all(np.isfinite(position)):
        raise ValueError("current_position must be finite and two-dimensional")
    # Read-only use of the companion model's cell state; planning never writes
    # the evidence arrays or calls observe(). We work on fresh residual arrays.
    row_indices = [tracker._index(c) for c in channels]
    residual_by_channel = ~tracker._excluded[row_indices]
    weights = np.sum(residual_by_channel, axis=0)
    required = weights > 0
    if not np.any(required):
        return []
    centers = tracker.centers[required]
    weights = weights[required].astype(float)
    fixed = _fixed_stations()
    fixed_masks = np.array([tracker._mask(q)[required] for q in fixed])
    if not np.all(np.any(fixed_masks, axis=0)):
        raise ValueError("the original seven stations do not cover this tracker configuration")

    # Assign each cell to its nearest station; if customized floating geometry
    # makes that station fail the exact mask, select a certified alternative.
    distances = np.sum((centers[:, None, :] - fixed[None, :, :]) ** 2, axis=2)
    distances[~fixed_masks.T] = np.inf
    assignment = np.argmin(distances, axis=1)
    source_positions = [np.asarray(task["position"], dtype=float) for task in source_tasks]
    if any(q.shape != (2,) or not np.all(np.isfinite(q)) for q in source_positions):
        raise ValueError("source proxy positions must be finite two-dimensional points")
    anchors = [position] + source_positions
    candidates = []
    masks = []

    def add(q):
        q = np.asarray(q, dtype=float)
        for i, old in enumerate(candidates):
            if np.linalg.norm(q - old) < 1e-7:
                return i
        mask = tracker._mask(q)[required]
        if not np.any(mask):
            return None
        if len(candidates) >= 40:
            return None
        candidates.append(q.copy())
        masks.append(mask.copy())
        return len(candidates) - 1

    fixed_plan = [index for q in fixed if (index := add(q)) is not None]
    toward_current = []
    toward_sources = []
    inset = tracker.signal_r - tracker.cell_half_diagonal - 1e-7
    for i, station in enumerate(fixed):
        group = centers[assignment == i]
        if not len(group):
            continue
        projected = _project_group(station, position, group, inset)
        index = add(projected)
        toward_current.append(index if index is not None else add(station))
        closest_anchor = min(anchors, key=lambda q: np.linalg.norm(q - station))
        projected = _project_group(station, closest_anchor, group, inset)
        index = add(projected)
        toward_sources.append(index if index is not None else add(station))
    add(position)
    for q in source_positions[:16]:
        add(q)
    add(np.average(centers, axis=0, weights=weights))
    mask_array = np.asarray(masks)
    candidates = np.asarray(candidates)

    def feasible(plan):
        return bool(plan) and bool(np.all(np.any(mask_array[plan], axis=0)))

    anchor_distance = np.min(np.linalg.norm(candidates[:, None, :] - np.asarray(anchors)[None, :, :], axis=2), axis=1)

    def prune(plan):
        plan = list(dict.fromkeys(index for index in plan if index is not None))
        # Remove the farthest redundant point first; retain a full set cover.
        for index in sorted(plan, key=lambda i: (-anchor_distance[i], i)):
            rest = [i for i in plan if i != index]
            if feasible(rest):
                plan = rest
        return plan

    plans = [prune(fixed_plan), prune(toward_current), prune(toward_sources)]
    scan_time = 6. * len(channels)
    # Two greedy costs: nearby source tasks may make a scan almost free in
    # distance, while the sequential version discourages a scattered cover.
    for use_route_anchors in (True, False):
        remaining = np.ones(len(centers), dtype=bool)
        plan = []
        cursor = position.copy()
        while np.any(remaining):
            gain = np.sum(mask_array[:, remaining] * weights[remaining], axis=1)
            travel = anchor_distance if use_route_anchors else np.linalg.norm(candidates - cursor, axis=1)
            score = gain / (scan_time + travel / 5. + 1.)
            if plan:
                score[plan] = -1.
            index = int(np.argmax(score))
            if gain[index] <= 0:
                # With the original stations present this cannot happen.
                plan = plans[0]
                break
            plan.append(index)
            remaining &= ~mask_array[index]
            cursor = candidates[index]
        plans.append(prune(plan))

    best = None
    for plan in plans:
        if len(plan) > 7 or not feasible(plan):
            continue
        scans = [_task(candidates[i]) for i in plan]
        route, distance = rolling_route(position, scans + list(source_tasks))
        # Source work is common to every candidate; only scan and travel
        # increments distinguish this proxy objective. It is not a forecast
        # of unknown future discoveries or a claim of global optimality.
        rank = (distance / 5. + scan_time * len(scans), len(scans), tuple(plan))
        if best is None or rank < best[0]:
            ordered_scans = [task for task in route if task["kind"] == "scan"]
            best = (rank, ordered_scans, plan)
    if best is None:
        raise RuntimeError("failed to construct a certified residual covering plan")
    assert feasible(best[2])
    return best[1]

