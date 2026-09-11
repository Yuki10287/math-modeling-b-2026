"""Independent local-only model validation for Q3 v2.

The layouts are deterministic stress cases, not the reserved seed-4000 test
population. No HTTP, official simulator, or network is used. The solver sees
only position/channel and public measure/clear operations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np

from environment import LocalArena


class PublicAPI:
    """The only public API surface supplied to the decision algorithm."""
    __slots__ = ("__measure", "__clear", "__position", "__channel")

    def __init__(self, arena):
        self.__measure, self.__clear = arena.measure, arena.clear
        self.__position = lambda: arena.position.copy()
        self.__channel = lambda: arena.channel

    @property
    def position(self):
        return self.__position()

    @property
    def channel(self):
        return self.__channel()

    def measure(self, q, channel):
        return self.__measure(q, channel)

    def clear(self, q, channel):
        return self.__clear(q, channel)


class BoundedArena(LocalArena):
    def __init__(self, *args, max_actions=6000, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_actions = max_actions

    def _budget(self):
        if len(self.events) >= self._max_actions:
            raise RuntimeError("local validation action limit reached")

    def _error(self, q, channel):
        if self._field == "quantized_edge":
            # Explicit additional robustness condition: full analog +1 degree
            # followed by LocalArena's independent two-decimal rounding.
            return 1.0
        return super()._error(q, channel)

    def measure(self, q, channel):
        self._budget()
        return super().measure(q, channel)

    def clear(self, q, channel):
        self._budget()
        return super().clear(q, channel)


def source(channel, x, y, radius=1000.):
    return dict(channel=int(channel), position=[float(x), float(y)], radius=float(radius))


def stress_cases():
    """Explicit nonuniform layouts; no random test-set samples are generated."""
    origin = [source(c, 0, 0) for c in range(1, 11)]
    coincident = [source(c, 1490, 350) for c in range(1, 13)]
    boundary = [source(c + 1, 1800 * math.cos((c + .5) * math.pi / 8),
                       1800 * math.sin((c + .5) * math.pi / 8)) for c in range(16)]
    last_far = [source(c + 1, -950 + 7 * c, 160 + (c % 3) * 9) for c in range(9)]
    last_far.append(source(20, 1798, 0))
    two_clusters = [source(c + 1, (-1 if c < 7 else 1) * (1330 + (c % 7) * 6),
                           (c % 7 - 3) * 12) for c in range(14)]
    aligned = [source(c + 1, 810 + 98 * c, 0) for c in range(10)]
    thresholds = [source(c + 1, *q) for c, q in enumerate([
        (5, 0), (5.00001, 0), (20, 0), (20.00001, 0), (999.99999, 0),
        (1000, 0), (1000.00001, 0), (-1000, 0), (0, 1800), (0, -1800)])]
    dense = [source(c + 1, 440 + 4 * (c % 4), -510 + 4 * (c // 4),
                    1000 if c % 2 else 1500) for c in range(16)]
    theta = math.radians(10.006)
    quantized = [source(c, 1499 * math.cos(theta), 1499 * math.sin(theta), 1500)
                 for c in range(1, 11)]
    return [dict(name=name, sources=items, field=field, seed=-901 - index)
            for index, (name, items, field) in enumerate([
                ("origin_coincident_10", origin, "constant"),
                ("remote_coincident_12", coincident, "extreme"),
                ("boundary_radius1000_16", boundary, "extreme"),
                ("cluster_plus_last_channel20", last_far, "constant"),
                ("two_dense_clusters_14", two_clusters, "hash"),
                ("collinear_sources_10", aligned, "extreme"),
                ("distance_thresholds_10", thresholds, "constant"),
                ("nearby_sources_mixed_radii_16", dense, "smooth"),
                ("quantization_robustness_10", quantized, "quantized_edge")])]


def independent_time_audit(arena):
    """Recompute all action costs without LocalArena.evaluation or its counters."""
    point, tuned = np.zeros(2), 1
    distance, duration, max_error = 0., 0., 0.
    counts = dict(measure=0, switch=0, clear_success=0, clear_fail=0)
    for index, event in enumerate(arena.events, 1):
        q = np.asarray(event["position"], float)
        assert q.shape == (2,) and np.isfinite(q).all() and np.max(np.abs(q)) <= 2_000_000
        leg = float(np.sqrt(np.sum((q - point) ** 2)))
        distance += leg
        duration += leg / 5.
        point = q
        if event["action"] == "measure":
            switched = int(tuned != event["channel"])
            counts["switch"] += switched
            counts["measure"] += 1
            duration += switched + 5
            tuned = event["channel"]
        else:
            assert event["action"] == "clear"
            success = event["clear_result"] == "success"
            counts["clear_success" if success else "clear_fail"] += 1
            duration += 5 if success else 3
        error = abs(duration - event["time_s"])
        max_error = max(max_error, error)
        assert error < 1e-5, (index, duration, event["time_s"])
    assert abs(duration - arena.time_s) < 1e-5
    assert abs(distance - arena.distance_m) < 1e-5
    assert tuned == arena.channel
    assert np.allclose(point, arena.position, atol=1e-9, rtol=0)
    parts = dict(move_s=distance / 5., measure_s=counts["measure"] * 5.,
                 switch_s=float(counts["switch"]), clear_success_s=counts["clear_success"] * 5.,
                 clear_fail_s=counts["clear_fail"] * 3.)
    assert abs(sum(parts.values()) - duration) < 1e-5
    return dict(total_s=duration, distance_m=distance, actions=len(arena.events), counts=counts,
                parts_s=parts, max_step_error_s=max_error,
                average_s_per_cleared=duration / counts["clear_success"] if counts["clear_success"] else None)


def observation_audit(arena):
    removed, repeats = set(), {}
    maximum_error, repeated = 0., 0
    per_channel = {c: dict(measure=0, direction=0, near=0, no_signal=0,
                          clear_success=0, clear_fail=0, negative_points=[]) for c in range(1, 21)}
    for index, event in enumerate(arena.events, 1):
        channel = event["channel"]
        assert channel in range(1, 21)
        item = arena._sources.get(channel)
        q = np.asarray(event["position"], float)
        distance = math.inf if item is None else float(np.linalg.norm(q - item["position"]))
        stats = per_channel[channel]
        if event["action"] == "clear":
            expected = item is not None and channel not in removed and distance <= 20
            assert (event["clear_result"] == "success") == expected, (index, distance, event)
            stats["clear_success" if expected else "clear_fail"] += 1
            if expected:
                removed.add(channel)
            continue
        stats["measure"] += 1
        kind = event["measure_result"]
        stats[kind] += 1
        if item is None or channel in removed or distance > item["radius"]:
            assert kind == "no_signal", (index, distance, event)
            stats["negative_points"].append(q.tolist())
        elif distance <= 5:
            assert kind == "near", (index, distance, event)
        else:
            assert kind == "direction", (index, distance, event)
            delta = np.asarray(item["position"]) - q
            angle = math.degrees(math.atan2(delta[1], delta[0])) % 360
            error = abs((event["svd_deg"] - angle + 180) % 360 - 180)
            maximum_error = max(maximum_error, error)
            allowed = 1.005 if arena._field == "quantized_edge" else 1.0
            assert error <= allowed + 1e-9
        key = (channel, tuple(q), channel in removed)
        response = (kind, event.get("svd_deg"))
        if key in repeats:
            assert repeats[key] == response, (index, "repeated-place measurement changed")
            repeated += 1
        repeats[key] = response
    assert removed == arena._removed
    return dict(max_observed_absolute_error_deg=maximum_error, repeated_place_checks=repeated,
                feedback_error_bound_deg=1.005 if arena._field == "quantized_edge" else 1.0,
                per_channel={str(k): v for k, v in per_channel.items()})


def independent_cells(cell_size=20., target_radius=1800.):
    """All closed squares intersecting the target disk, including boundary slivers."""
    half = cell_size / 2.
    extent = math.ceil(target_radius / cell_size)
    centers = []
    for i in range(-extent, extent):
        x = (i + .5) * cell_size
        for j in range(-extent, extent):
            y = (j + .5) * cell_size
            nearest_x, nearest_y = max(abs(x) - half, 0.), max(abs(y) - half, 0.)
            if nearest_x ** 2 + nearest_y ** 2 <= target_radius ** 2 + 1e-8:
                centers.append((x, y))
    return np.asarray(centers)


def independent_negative_cover(points, centers, cell_size=20., radius=1000.):
    """Exact four-corner bound for each whole square, independently of tracker code."""
    covered = np.zeros(len(centers), dtype=bool)
    for q in points:
        # The farthest corner is displaced by |center-q|+half in each axis.
        far = np.abs(centers - np.asarray(q)) + cell_size / 2.
        covered |= np.sum(far ** 2, axis=1) <= radius ** 2 + 1e-8
    return covered


def find_channel_certificates(value):
    found = {}
    if isinstance(value, dict):
        if "channel" in value and "negative_points" in value and "cell_size_m" in value:
            found[int(value["channel"])] = value
        for item in value.values():
            found.update(find_channel_certificates(item))
    elif isinstance(value, list):
        for item in value:
            found.update(find_channel_certificates(item))
    return found


def certificate_audit(arena, result, observations, centers):
    declared = result["completion_certificate"]
    assert result.get("complete") is True and declared.get("valid") is True, result
    assert set(declared["cleared_channels"]) == arena._removed
    absent = set(range(1, 21)) - set(arena._sources)
    claimed_absent = set(declared["absent_channels"])
    assert claimed_absent <= absent
    by_count = len(arena._removed) == 16
    actual_cover = {}
    declared_channels = find_channel_certificates(declared)
    for channel in sorted(absent):
        points = observations["per_channel"][str(channel)]["negative_points"]
        covered = independent_negative_cover(points, centers)
        actual_cover[str(channel)] = dict(negative_points=len(points),
                                          covered_cells=int(covered.sum()), total_cells=len(centers),
                                          complete=bool(covered.all()))
        if not by_count:
            assert covered.all(), (channel, "missing continuous no-signal coverage", int((~covered).sum()))
        item = declared_channels.get(channel)
        if item is not None:
            assert item["cell_size_m"] == 20 and item["target_radius_m"] == 1800
            assert item["minimum_reception_radius_m"] == 1000
            assert item["total_cells"] == len(centers)
            assert 0 <= item["excluded_cells"] <= int(covered.sum())
            assert item["uncovered_cells"] == len(centers) - item["excluded_cells"]
            public_points = {tuple(q) for q in points}
            assert {tuple(q) for q in item["negative_points"]} == public_points
            if item["complete"]:
                assert covered.all()
    if not by_count:
        assert claimed_absent == absent
        assert set(arena._removed) | claimed_absent == set(range(1, 21))
        assert all(channel in declared_channels for channel in absent), "Missing per-channel continuous certificates"
    return dict(independent_basis="count_16" if by_count else "whole_square_negative_disk_coverage",
                per_absent_channel=actual_cover, declared_channel_certificates=len(declared_channels))


def _contains_convex(polygon, truth, tolerance=1e-5):
    P = np.asarray(polygon, float)
    assert P.ndim == 2 and P.shape[1] == 2 and len(P) and np.isfinite(P).all()
    if len(P) == 1:
        return np.linalg.norm(P[0] - truth) <= tolerance
    if len(P) == 2:
        delta = P[1] - P[0]
        t = np.clip((truth - P[0]) @ delta / max(delta @ delta, 1e-30), 0, 1)
        return np.linalg.norm(truth - (P[0] + t * delta)) <= tolerance
    edges, offsets = np.roll(P, -1, axis=0) - P, truth - P
    lengths = np.linalg.norm(edges, axis=1)
    keep = lengths > 1e-7
    if not keep.any():
        return np.linalg.norm(P[0] - truth) <= tolerance
    cross = edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0]
    distances = cross[keep] / lengths[keep]
    return bool(distances.min() >= -tolerance or distances.max() <= tolerance)


def trace_truth_audit(arena, trace):
    checked, optical_plans = 0, 0
    conflicts = [row for row in trace if row.get("phase") == "belief_conflict"]
    assert not conflicts, ("consistent local observations created empty or invalid beliefs", conflicts)
    for index, row in enumerate(trace):
        if "polygon" in row and "channel" in row:
            truth = np.asarray(arena._sources[row["channel"]]["position"], float)
            assert _contains_convex(row["polygon"], truth), (index, row.get("phase"), row["channel"])
            checked += 1
            path = row.get("optical_path")
            if path is not None:
                assert independent_disk_cover(row["polygon"], path)["complete"]
                optical_plans += 1
    return dict(polygons_checked=checked, truth_exclusion_violations=0,
                full_optical_paths_checked=optical_plans, belief_conflicts=0)


def _clip_independent(polygon, normal, rhs):
    output = []
    if not len(polygon):
        return np.empty((0, 2))
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        da, db = float(a @ normal - rhs), float(b @ normal - rhs)
        inside_a, inside_b = da <= 1e-9, db <= 1e-9
        if inside_a:
            output.append(a)
        if inside_a != inside_b:
            output.append(a + da / (da - db) * (b - a))
    return np.asarray(output).reshape(-1, 2)


def independent_disk_cover(polygon, path, radius=20.):
    """Whole polygon coverage by disks via independently coded Voronoi clipping."""
    P, path = np.asarray(polygon, float), np.asarray(path, float)
    assert len(P) and len(path)
    max_distance, nonempty = 0., 0
    for i, q in enumerate(path):
        cell = P.copy()
        for j, other in enumerate(path):
            if i == j:
                continue
            # |x-q|^2 <= |x-other|^2 gives this halfplane.
            cell = _clip_independent(cell, 2 * (other - q), float(other @ other - q @ q))
            if not len(cell):
                break
        if len(cell):
            nonempty += 1
            max_distance = max(max_distance, float(np.linalg.norm(cell - q, axis=1).max()))
    return dict(complete=bool(nonempty and max_distance <= radius + 1e-7),
                max_nearest_distance_m=max_distance, nonempty_voronoi_cells=nonempty)


def negative_hull_checks():
    """Property checks use random geometry points, never test-set source seeds."""
    from belief_model import exclude_disk_hull, FeedbackModel
    import geometry
    rng = np.random.default_rng(317)
    layouts = [
        ("two_remaining_branches", [[-40, -10], [40, -10], [40, 10], [-40, 10]], [0, 0], 20),
        ("polygon_contains_disk", [[-40, -40], [40, -40], [40, 40], [-40, 40]], [0, 0], 20),
        ("entirely_outside", [[50, 50], [60, 50], [60, 60], [50, 60]], [0, 0], 20),
        ("entirely_inside", [[-5, -5], [5, -5], [5, 5], [-5, 5]], [0, 0], 20),
        ("one_remaining_branch", [[-10, -10], [35, -10], [35, 10], [-10, 10]], [0, 0], 20),
        ("tangent", [[-10, -10], [10, -10], [10, 10], [-10, 10]], [30, 0], 20),
        ("translated", [[1420, 220], [1580, 220], [1580, 380], [1420, 380]], [1530, 300], 50),
        ("degenerate_segment", [[-50, 0], [50, 0]], [0, 0], 20),
        ("single_point_outside", [[25, 0]], [0, 0], 20),
        ("single_point_inside", [[5, 0]], [0, 0], 20),
    ]
    rows = []
    for name, vertices, q, radius in layouts:
        P, q = np.asarray(vertices, float), np.asarray(q, float)
        hull = exclude_disk_hull(P, q, radius)
        interior = rng.dirichlet(np.ones(len(P)), size=1500) @ P
        fractions = np.linspace(0, 1, 101)[:, None]
        boundary = np.vstack([a + fractions * (b - a) for a, b in zip(P, np.roll(P, -1, axis=0))])
        points = np.vstack((P, interior, boundary))
        feasible = points[np.linalg.norm(points - q, axis=1) > radius + 1e-8]
        if len(feasible):
            assert len(hull), name
            assert all(_contains_convex(hull, point) for point in feasible), (name, "feasible point excluded")
        if name in ("entirely_inside", "single_point_inside"):
            assert not len(hull)
        if name in ("two_remaining_branches", "polygon_contains_disk"):
            assert all(_contains_convex(hull, point) for point in P), (name, "branch lost")
        rows.append(dict(name=name, points_checked=len(points), feasible_points_checked=len(feasible),
                         result_vertices=len(hull), feasible_exclusion_violations=0))
    # A positive w and a negative q refer to the same fixed-radius source:
    # |g-w| <= R < |g-q| implies 2(q-w).g < |q|^2-|w|^2.
    slacks = []
    range_model = FeedbackModel(None, [], range_cuts=True)
    for _ in range(1000):
        truth = rng.uniform(-1500, 1500, size=2)
        radius = rng.uniform(1000, 1500)
        angles = rng.uniform(0, 2 * math.pi, size=2)
        w = truth + radius * rng.uniform(0, .999) * np.array([math.cos(angles[0]), math.sin(angles[0])])
        q = truth + (radius + rng.uniform(.01, 800)) * np.array([math.cos(angles[1]), math.sin(angles[1])])
        normal = 2 * (q - w)
        rhs = float(q @ q - w @ w)
        slack = rhs - float(normal @ truth)
        assert slack > 0
        square = truth + np.array([[-40, -40], [40, -40], [40, 40], [-40, 40]])
        assert _contains_convex(geometry.clip(square, normal, rhs), truth)
        assert _contains_convex(range_model._range_cut(square, q, w), truth)
        slacks.append(slack)
    return dict(geometry_rng_seed=317, source_test_population_used=False, cases=rows,
                set_inclusion_reason="An inscribed exclusion polygon lies inside the forbidden disk. Retaining every outside halfplane piece preserves P minus that polygon, hence P minus the disk. Its convex hull is an outer relaxation that preserves all remaining branches; it may restore holes.",
                fixed_radius_halfplane=dict(pair_checks=1000, actual_model_helper_checked=True,
                    minimum_positive_slack_m2=min(slacks),
                    inequality="2*(q-w) dot g <= q dot q - w dot w", verified_sign=True,
                    assumptions="Same stationary omnidirectional source and fixed R; positive w and negative q must both precede successful removal."))


def feedback_guard_checks():
    from belief_model import FeedbackModel
    from recovery import APIResponseError

    class ReplyAPI:
        position = np.zeros(2)
        channel = 1
        reply = None

        def measure(self, q, channel):
            return self.reply

        def clear(self, q, channel):
            return self.reply

    def snapshot(model):
        return dict(certificate=model.certificate(), discovered=sorted(model.discovered),
                    positions={str(c): [p.tolist() for p in points] for c, points in model.positions.items()},
                    polygons={str(c): item["P"].tolist() for c, item in model.beliefs.items()})

    malformed = [
        ("measure", {"accepted": False, "measure_result": "no_signal"}),
        ("measure", {"accepted": "true", "measure_result": "no_signal"}),
        ("measure", {"accepted": True, "measure_result": "unknown"}),
        ("measure", {"accepted": True}),
        ("measure", []),
        ("measure", {"measure_result": "direction"}),
        ("measure", {"measure_result": "direction", "svd_deg": math.nan}),
        ("measure", {"measure_result": "direction", "svd_deg": math.inf}),
        ("measure", {"measure_result": "direction", "svd_deg": True}),
        ("measure", {"measure_result": "direction", "svd_deg": 360}),
        ("clear", {"accepted": False, "clear_result": "success"}),
        ("clear", {"accepted": True, "clear_result": "unknown"}),
        ("clear", {"accepted": True}),
        ("clear", []),
    ]
    for action, reply in malformed:
        api, trace = ReplyAPI(), []
        model = FeedbackModel(api, trace)
        api.reply = {"measure_result": "direction", "svd_deg": 20.0}
        model.measure([0, 0], 3)
        before = snapshot(model)
        trace_before = len(trace)
        api.reply = reply
        try:
            getattr(model, action)([100, 100], 3)
        except APIResponseError:
            pass
        else:
            raise AssertionError((action, reply, "malformed feedback accepted"))
        assert snapshot(model) == before
        assert len(trace) == trace_before
        assert not model.certificate()["valid"]
    api, trace = ReplyAPI(), []
    model = FeedbackModel(api, trace, range_cuts=True)
    api.reply = {"measure_result": "direction", "svd_deg": 20.0}
    model.measure([0, 0], 3)
    api.reply = {"clear_result": "success"}
    model.clear([0, 0], 3)
    before, trace_before = snapshot(model), len(trace)
    api.reply = {"measure_result": "no_signal"}
    model.measure([1500, 0], 3)
    assert snapshot(model) == before and len(trace) == trace_before
    return dict(rejected_or_malformed_cases=len(malformed), unchanged_belief_and_certificate=True,
                no_rejected_action_recorded_as_measurement=True,
                post_removal_silence_not_used_as_range_evidence=True)


def shared_clear_and_count_checks():
    from belief_model import FeedbackModel
    from solver import clear_at_stop
    arena = BoundedArena([source(c, 0, 0) for c in range(1, 17)], seed=-910, field="constant")
    trace = []
    model = FeedbackModel(PublicAPI(arena), trace, range_cuts=True)
    for channel in range(1, 17):
        model.measure(np.zeros(2), channel)
    assert len(model.discovered) == 16 and not model.cleared
    assert model.unknown() == []
    assert model.certificate()["valid"] is False, "16 discovered sources are not 16 cleared sources"
    active = model.beliefs[1]
    active_polygon = active["P"].copy()
    tuned_before, position_before = arena.channel, arena.position.copy()
    clear_at_stop(model, 1, position_before)
    assert model.beliefs[1] is active and np.array_equal(model.beliefs[1]["P"], active_polygon)
    assert model.cleared == set(range(2, 17)) and arena._removed == set(range(2, 17))
    assert set(model.beliefs) == {1}
    assert model.certificate()["valid"] is False
    assert np.array_equal(arena.position, position_before) and arena.channel == tuned_before
    assert sum(row.get("phase") == "shared_clear" for row in trace) == 15
    model.clear(np.zeros(2), 1)
    assert model.certificate()["valid"] is True
    assert model.certificate()["basis"] == "count_upper_bound"
    timing = independent_time_audit(arena)
    observation_audit(arena)
    trace_truth_audit(arena, trace)
    assert timing["distance_m"] == 0 and timing["total_s"] == 175
    return dict(sixteen_discovered_does_not_imply_complete=True,
                fifteen_other_near_sources_cleared_at_same_stop=True,
                active_source_and_its_belief_preserved=True, no_motion_or_tuning_change_from_clear=True,
                completed_only_after_sixteenth_success=True, independent_total_s=timing["total_s"])


def mathematical_checks():
    from local_policy import best_optical_plan
    import geometry
    import baseline_solver
    centers = independent_cells()
    points = [[0, 0]] + [[1200 * math.cos(i * math.pi / 3), 1200 * math.sin(i * math.pi / 3)] for i in range(6)]
    assert independent_negative_cover(points, centers).all()
    assert not independent_negative_cover([[0, 0]], centers).all()
    square = [[-20, -20], [20, -20], [20, 20], [-20, 20]]
    vertex_counterexample = independent_disk_cover(square, square)
    assert not vertex_counterexample["complete"], "Vertex-only union coverage is unsound"
    plans = []
    for name, polygon, start in [
            ("long_thin_rectangle", [[-65, -5], [65, -5], [65, 5], [-65, 5]], [-90, 0]),
            ("wide_square", [[-28, -28], [28, -28], [28, 28], [-28, 28]], [90, 40]),
            ("triangle", [[-30, -10], [55, 0], [-30, 20]], [-50, -30])]:
        plan = best_optical_plan(np.asarray(polygon, float), np.asarray(start, float))
        assert plan is not None
        independent = independent_disk_cover(polygon, plan["path"])
        assert independent["complete"], (name, independent)
        plans.append(dict(name=name, parts=plan["parts"], **independent))
    true_bearing, analog_error = 10.006, 1.0
    rounded = round(true_bearing + analog_error, 2)
    returned_error = rounded - true_bearing
    assert 1 < returned_error <= 1.005 + 1e-12
    theta = math.radians(true_bearing)
    truth = 1499 * np.asarray([math.cos(theta), math.sin(theta)])
    enlarged = geometry.initial_belief(np.zeros(2), rounded)
    original = baseline_solver.initial_belief(np.zeros(2), rounded)
    assert _contains_convex(enlarged, truth)
    assert not _contains_convex(original, truth)
    return dict(continuous_cells=len(centers), original_seven_stations_also_cover_all_whole_cells=True,
                vertex_only_disk_union_counterexample=vertex_counterexample, optical_plans=plans,
                negative_hull=negative_hull_checks(), feedback_guards=feedback_guard_checks(),
                shared_clear_and_count=shared_clear_and_count_checks(),
                quantization_counterexample=dict(true_bearing_deg=true_bearing, analog_error_deg=analog_error,
                    returned_two_decimal_bearing_deg=rounded, returned_error_deg=returned_error,
                    conservative_halfwidth_deg=1.005,
                    v2_contains_truth=True, v1_contains_truth=False,
                    v2_actual_halfwidth_deg=math.degrees(geometry.ANGLE_EPS),
                    preserved_v1_halfwidth_deg=math.degrees(baseline_solver.ANGLE_EPS),
                    interpretation="Extra robustness scenario only: analog error <=1 degree followed by rounding; not proof that the official bound excludes quantization."))


def run_case(solve_multi, case, centers, *, max_actions=6000):
    sources = case["sources"]
    assert 10 <= len(sources) <= 16
    assert len({s["channel"] for s in sources}) == len(sources)
    assert all(np.linalg.norm(s["position"]) <= 1800 + 1e-8 and 1000 <= s["radius"] <= 1500 for s in sources)
    arena = BoundedArena(sources, seed=case["seed"], field=case["field"], max_actions=max_actions)
    api = PublicAPI(arena)
    forbidden = ("_sources", "_removed", "evaluation", "events", "_arena", "__dict__", "time_s")
    assert all(not hasattr(api, name) for name in forbidden)
    trace = []
    start = time.perf_counter()
    result = solve_multi(api, trace=trace)
    elapsed = time.perf_counter() - start
    assert len(arena._removed) == len(sources), (case["name"], result, arena.evaluation())
    timing = independent_time_audit(arena)
    observations = observation_audit(arena)
    certificate = certificate_audit(arena, result, observations, centers)
    geometry = trace_truth_audit(arena, trace)
    return dict(name=case["name"], passed=True, field=case["field"], seed=case["seed"], sources=sources,
                source_count=len(sources), cleared=len(arena._removed), all_cleared=True,
                local_solver_wall_s=elapsed, result=result, independent_timing=timing,
                public_observations=observations, independent_certificate=certificate,
                geometry=geometry, truth_interface_hidden=list(forbidden), trace_rows=len(trace))


def source_hashes():
    names = ("solver.py", "belief_model.py", "geometry.py", "local_policy.py", "coverage_model.py",
             "scan_planning.py", "recovery.py", "baseline_solver.py", "v1_solver.py", "environment.py", "validate_model.py")
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=[c["name"] for c in stress_cases()])
    parser.add_argument("--max-actions", type=int, default=6000)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "validation.json")
    args = parser.parse_args()
    initial_hashes = source_hashes()
    from solver import solve_multi
    selected = [case for case in stress_cases() if not args.case or case["name"] in args.case]
    report = dict(local_only=True, network_used=False, official_simulator_used=False,
                  reserved_seed4000_population_used=False, mathematical_checks=mathematical_checks(), cases=[],
                  source_sha256_before=initial_hashes)
    centers = independent_cells()
    for case in selected:
        print("validate", case["name"], flush=True)
        try:
            row = run_case(solve_multi, case, centers, max_actions=args.max_actions)
        except Exception as exc:
            row = dict(name=case["name"], passed=False, error_type=type(exc).__name__,
                       error=str(exc), traceback=traceback.format_exc())
        report["cases"].append(row)
        final_hashes = source_hashes()
        report["source_snapshot_stable"] = initial_hashes == final_hashes
        report["passed"] = all(row["passed"] for row in report["cases"]) and report["source_snapshot_stable"]
        report["completed_cases"] = len(report["cases"])
        report["requested_cases"] = len(selected)
        report["source_sha256"] = final_hashes
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"case": case["name"], "passed": row["passed"], "error": row.get("error")}, ensure_ascii=False), flush=True)
    raise SystemExit(0 if report["passed"] and len(report["cases"]) == len(selected) else 1)


if __name__ == "__main__":
    main()
