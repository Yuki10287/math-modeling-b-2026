"""Independent, local-only checks for Q3 strategies. Never contacts the official simulator.

The solver receives a narrow adapter; source truth is retained by this validator.
Run with --policy midpoint for a quick structural stress run, or --policy time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import numpy as np

from environment import LocalArena, audit_trace


class PublicAPI:
    """Only the operation methods and the two required public state fields exist."""

    __slots__ = ("__measure", "__clear", "__position", "__channel")

    def __init__(self, arena):
        self.__measure = arena.measure
        self.__clear = arena.clear
        self.__position = lambda: arena.position.copy()
        self.__channel = lambda: arena.channel

    @property
    def position(self):
        return self.__position()

    @property
    def channel(self):
        return self.__channel()

    def measure(self, position, channel):
        return self.__measure(position, channel)

    def clear(self, position, channel):
        return self.__clear(position, channel)


def source(channel, x, y, radius=1000.0):
    return dict(channel=int(channel), position=[float(x), float(y)], radius=float(radius))


def coverage_check(points):
    """Analytic disk coverage plus an independently sampled cross-check.

    For rho <= 1000, the origin covers the point. For 1000 <= rho <= 1800,
    a hexagon vertex is at angular distance <= pi/6. Its distance squared is
    convex in rho, so the maximum occurs at one of the two annulus endpoints.
    """
    points = np.asarray(points, float)
    angles = np.arange(6) * math.pi / 3
    expected = np.vstack(([0.0, 0.0], 1200 * np.column_stack((np.cos(angles), np.sin(angles)))))
    assert points.shape == expected.shape
    assert all(np.min(np.linalg.norm(points - p, axis=1)) < 1e-8 for p in expected)
    endpoint_distances = [math.sqrt(r * r + 1200**2 - 2 * r * 1200 * math.cos(math.pi / 6)) for r in (1000, 1800)]
    assert max(endpoint_distances) < 1000
    rho = np.linspace(0, 1800, 181)
    theta = np.linspace(0, 2 * math.pi, 1441)
    sample = (rho[:, None, None] * np.column_stack((np.cos(theta), np.sin(theta)))[None, :, :]).reshape(-1, 2)
    sampled_max = 0.0
    for chunk in np.array_split(sample, 32):
        sampled_max = max(sampled_max, float(np.min(np.linalg.norm(chunk[:, None, :] - points[None, :, :], axis=2), axis=1).max()))
    assert sampled_max <= 1000 + 1e-8
    return dict(continuous_coverage_proved=True, annulus_endpoint_distances_m=endpoint_distances,
                sampled_max_nearest_distance_m=sampled_max, samples=len(sample))


def independent_time_audit(arena):
    """Recompute action timing without calling LocalArena.evaluation()."""
    position = np.zeros(2)
    tuned_channel = 1
    duration = 0.0
    distance = 0.0
    counts = dict(measure=0, switch=0, clear_success=0, clear_fail=0)
    for index, event in enumerate(arena.events):
        next_position = np.asarray(event['position'], float)
        leg = float(np.linalg.norm(next_position - position))
        duration += leg / 5
        distance += leg
        position = next_position
        if event['action'] == 'measure':
            if tuned_channel != event['channel']:
                duration += 1
                counts['switch'] += 1
                tuned_channel = event['channel']
            duration += 5
            counts['measure'] += 1
        elif event['action'] == 'clear':
            success = event['clear_result'] == 'success'
            duration += 5 if success else 3
            counts['clear_success' if success else 'clear_fail'] += 1
        else:
            raise AssertionError(f"Unknown action at {index}: {event['action']}")
        assert abs(duration - event['time_s']) <= 1e-6 * max(1, duration / 10000), (index, duration, event['time_s'])
    assert abs(duration - arena.time_s) <= 1e-6 * max(1, duration / 10000)
    assert abs(distance - arena.distance_m) <= 1e-6
    assert counts == arena.counts
    assert tuned_channel == arena.channel
    assert np.allclose(position, arena.position, atol=1e-9, rtol=0)
    return dict(total_s=duration, distance_m=distance, counts=counts,
                average_s=duration / max(counts['clear_success'], 1), actions=len(arena.events))


def station_index(position, stations):
    distances = np.linalg.norm(stations - np.asarray(position, float), axis=1)
    return int(np.argmin(distances)) if distances.min() < 1e-6 else None


def absence_audit(arena, stations, result):
    """Independently establish a negative observation at every station per absent channel."""
    negative = {c: set() for c in range(1, 21)}
    seen_positive = set()
    for event in arena.events:
        if event['action'] != 'measure':
            continue
        if event['measure_result'] != 'no_signal':
            seen_positive.add(event['channel'])
        index = station_index(event['position'], stations)
        if index is not None and event['measure_result'] == 'no_signal':
            negative[event['channel']].add(index)
    absent = sorted(set(range(1, 21)) - set(arena._sources))
    count_certificate = len(arena._removed) == 16
    missing = {str(c): sorted(set(range(len(stations))) - negative[c]) for c in absent
               if negative[c] != set(range(len(stations)))}
    if not count_certificate:
        assert not missing, f"Absent channels lack coverage certificate: {missing}"
    assert set(arena._sources) <= seen_positive, "Some actual source was never detected"
    if result.get('status') == 'complete_by_count':
        assert count_certificate
    declared = result['completion_certificate']
    assert result['complete'] is True and declared['valid'] is True
    assert set(declared['cleared_channels']) == arena._removed
    assert set(declared['absent_channels']) <= set(absent)
    for channel in range(1, 21):
        assert set(declared['negative_scan_indices'][str(channel)]) == negative[channel]
    if not count_certificate:
        assert set(declared['absent_channels']) == set(absent)
    return dict(certificate='count_16' if count_certificate else 'per_channel_coverage',
                absent_channels=absent, negative_station_counts={str(c): len(negative[c]) for c in absent},
                absent_channels_missing_stations=missing)


def observation_audit(arena):
    """Verify recorded public feedback against local truth and the stated angular bound."""
    removed = set()
    maximum_error = 0.0
    for event in arena.events:
        channel = event['channel']
        item = arena._sources.get(channel)
        position = np.asarray(event['position'], float)
        distance = math.inf if item is None else float(np.linalg.norm(position - item['position']))
        if event['action'] == 'clear':
            expected = item is not None and channel not in removed and distance <= 20
            assert (event['clear_result'] == 'success') == expected
            if expected:
                removed.add(channel)
        elif item is None or channel in removed or distance > item['radius']:
            assert event['measure_result'] == 'no_signal'
        elif distance <= 5:
            assert event['measure_result'] == 'near'
        else:
            assert event['measure_result'] == 'direction'
            delta = np.asarray(item['position']) - position
            angle = math.degrees(math.atan2(delta[1], delta[0])) % 360
            error = abs((event['svd_deg'] - angle + 180) % 360 - 180)
            maximum_error = max(maximum_error, error)
            assert error <= 1 + 1e-9
    assert removed == arena._removed
    return dict(max_observed_absolute_error_deg=maximum_error)


def trace_truth_audit(arena, trace):
    violations = audit_trace(arena, trace)
    assert not violations, violations
    checked = 0
    for row in trace:
        if 'polygon' not in row or 'channel' not in row:
            continue
        polygon = np.asarray(row['polygon'], float)
        assert len(polygon) and polygon.shape[1] == 2 and np.all(np.isfinite(polygon))
        truth = np.asarray(arena._sources[row['channel']]['position'], float)
        if len(polygon) == 1:
            assert np.linalg.norm(truth - polygon[0]) <= 1e-6
        elif len(polygon) == 2:
            delta = polygon[1] - polygon[0]
            t = np.clip((truth - polygon[0]) @ delta / max(delta @ delta, 1e-30), 0, 1)
            assert np.linalg.norm(truth - (polygon[0] + t * delta)) <= 1e-6
        else:
            edges = np.roll(polygon, -1, axis=0) - polygon
            offsets = truth - polygon
            # Clipping can emit consecutive copies differing at 1e-12 m. Such
            # edges have no stable normal and are redundant for containment.
            lengths = np.linalg.norm(edges, axis=1)
            usable = lengths > 1e-7
            assert usable.any()
            signed_distance = (edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0])[usable] / lengths[usable]
            assert signed_distance.min() >= -1e-5, (row.get('phase'), row['channel'], signed_distance.min())
        checked += 1
    return dict(polygons_checked=checked, truth_exclusion_violations=0)


def run_case(solve_multi, stations, name, sources, schedule, policy, field='extreme', seed=701):
    assert 10 <= len(sources) <= 16
    assert len({s['channel'] for s in sources}) == len(sources)
    assert all(np.linalg.norm(s['position']) <= 1800 + 1e-9 and s['radius'] >= 1000 for s in sources)
    arena = LocalArena(sources, seed=seed, field=field)
    api = PublicAPI(arena)
    for forbidden in ('_sources', '_removed', 'evaluation', 'events', '_arena', '__dict__'):
        assert not hasattr(api, forbidden), forbidden
    trace = []
    started = time.perf_counter()
    result = solve_multi(api, policy=policy, schedule=schedule, trace=trace)
    elapsed = time.perf_counter() - started
    assert len(arena._removed) == len(sources), (name, schedule, result, arena.evaluation())
    assert str(result['status']).startswith('complete'), result
    assert result['cleared'] == len(sources)
    assert arena.counts['clear_fail'] == 0
    audited = independent_time_audit(arena)
    certificate = absence_audit(arena, stations, result)
    geometry = trace_truth_audit(arena, trace)
    observed = observation_audit(arena)
    visited = []
    for event in arena.events:
        if event['action'] != 'measure':
            continue
        index = station_index(event['position'], stations)
        if index is not None and index not in visited:
            visited.append(index)
    near_count = sum(e.get('measure_result') == 'near' for e in arena.events)
    return dict(case=name, schedule=schedule, policy=policy, field=field, seed=seed,
                passed=True, source_count=len(sources), cleared=len(arena._removed),
                all_cleared=True, solver_result=result, runtime_s=elapsed,
                timing=audited, coverage_certificate=certificate, trace_audit=geometry,
                observation_audit=observed, near_responses=near_count,
                station_visit_order=visited), arena


def stress_cases():
    channels = [1, 3, 5, 7, 9, 11, 13, 15, 17, 20]
    origin = [source(c, 0, 0) for c in channels]
    distances = [0, 4.999, 5, 5.001, 19.999, 20, 20.001, 999.999, 1200, 1799.999]
    near = [source(c, r * math.cos(i * 0.61), r * math.sin(i * 0.61)) for i, (c, r) in enumerate(zip(channels, distances))]
    boundary_point = 1800 * np.array([math.cos(math.pi / 6), math.sin(math.pi / 6)])
    colocated = [source(c, *boundary_point) for c in range(1, 17)]
    ring = [source(c, 1800 * math.cos((c - 1) * 2 * math.pi / 16 + math.pi / 30),
                   1800 * math.sin((c - 1) * 2 * math.pi / 16 + math.pi / 30)) for c in range(1, 17)]
    return [('ten_sources_at_origin', origin, 'smooth'),
            ('near_and_distance_boundaries', near, 'constant'),
            ('sixteen_colocated_worst_boundary', colocated, 'extreme'),
            ('sixteen_minimum_radius_boundary_ring', ring, 'hash')]


def injected_failure_checks(solve_multi, stations):
    """Exercise scan-proxy and source-initialization recovery end to end.

    Nine sources immediately return near; channel 20 at 10 m is the only normal
    bearing. Therefore the first initial_belief call is its scan proxy, and the
    second is its service initialization. Only the geometry function is patched.
    """
    import baseline_solver as core

    sources = [source(c, 0, 0) for c in range(1, 10)] + [source(20, 10, 0)]
    original_initial_belief = core.initial_belief
    checks = []

    def failing_geometry(number_of_failures):
        counter = {'calls': 0}

        def initial_belief(*args, **kwargs):
            counter['calls'] += 1
            if counter['calls'] <= number_of_failures:
                raise FloatingPointError('injected initial geometry failure')
            return original_initial_belief(*args, **kwargs)

        return initial_belief, counter

    for schedule in ('safe', 'dynamic'):
        for failures in (1, 2):
            arena = LocalArena(sources, seed=701, field='constant')
            trace = []
            replacement, counter = failing_geometry(failures)
            with patch.object(core, 'initial_belief', replacement):
                result = solve_multi(PublicAPI(arena), policy='time', schedule=schedule, trace=trace)
            assert result['complete'] is True and len(arena._removed) == 10
            proxy = [row for row in trace if row.get('phase') == 'proxy_recovery']
            fallback = [row for row in trace if row.get('phase') == 'fallback_start']
            assert len(proxy) == 1 and proxy[0]['channel'] == 20
            assert len(fallback) == failures - 1
            assert result['fallback_count'] == failures - 1
            if fallback:
                assert fallback[0]['reason'].startswith('initial_geometry:')
                completed = [row for row in trace if row.get('phase') == 'fallback_end']
                assert len(completed) == 1 and completed[0]['success'] is True
                assert 1 <= completed[0]['calls'] <= 152
            checks.append(dict(case='injected_initial_geometry', schedule=schedule,
                               injected_failures=failures, geometry_calls=counter['calls'],
                               passed=True, cleared=10, proxy_recoveries=len(proxy),
                               fallback_count=len(fallback), timing=independent_time_audit(arena),
                               coverage_certificate=absence_audit(arena, stations, result),
                               trace_audit=trace_truth_audit(arena, trace),
                               observation_audit=observation_audit(arena)))

    class InjectedTransportError(RuntimeError):
        pass

    class FailingAPI(PublicAPI):
        __slots__ = ('mode',)

        def __init__(self, arena, mode):
            super().__init__(arena)
            self.mode = mode

        def measure(self, position, channel):
            if channel == 20 and self.mode == 'measure_transport':
                raise InjectedTransportError('injected measure transport failure')
            if channel == 20 and self.mode == 'measure_rejected':
                return dict(accepted=False, measure_result='direction', svd_deg=0.0)
            return super().measure(position, channel)

        def clear(self, position, channel):
            if channel == 20 and self.mode == 'clear_transport':
                raise InjectedTransportError('injected clear transport failure')
            if channel == 20 and self.mode == 'clear_rejected':
                return dict(accepted=False, clear_result='success')
            return super().clear(position, channel)

    for schedule in ('safe', 'dynamic'):
        for mode in ('measure_transport', 'measure_rejected', 'clear_transport', 'clear_rejected'):
            arena = LocalArena(sources, seed=701, field='constant')
            trace = []
            # Clear failures are injected inside a genuine fallback, verifying
            # that recovery does not turn a transport/rejection failure into a miss.
            geometry_failures = 2 if mode.startswith('clear') else 0
            replacement, _ = failing_geometry(geometry_failures)
            caught = None
            with patch.object(core, 'initial_belief', replacement):
                try:
                    solve_multi(FailingAPI(arena, mode), policy='time', schedule=schedule, trace=trace)
                except RuntimeError as exc:
                    caught = exc
            assert caught is not None, (schedule, mode, 'API failure was swallowed')
            if mode.endswith('transport'):
                assert isinstance(caught, InjectedTransportError), repr(caught)
            else:
                assert '未被接受' in str(caught), repr(caught)
            assert 20 not in arena._removed
            assert not any(row.get('phase') == 'fallback_end' and row.get('success') for row in trace)
            fallback_count = sum(row.get('phase') == 'fallback_start' for row in trace)
            assert fallback_count == (1 if mode.startswith('clear') else 0)
            checks.append(dict(case='api_failure_propagates', schedule=schedule, mode=mode,
                               passed=True, exception_type=type(caught).__name__,
                               successful_target_clear=False, fallback_count=fallback_count,
                               timing=independent_time_audit(arena)))

    for schedule in ('baseline', 'safe', 'dynamic'):
        arena = LocalArena(sources)
        try:
            solve_multi(PublicAPI(arena), schedule=schedule, max_local_steps=True)
        except ValueError:
            pass
        else:
            raise AssertionError('Boolean max_local_steps was accepted')
        assert not arena.events
        checks.append(dict(case='boolean_step_limit_rejected', schedule=schedule, passed=True, actions=0))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=['time', 'geometry', 'midpoint'], default='time')
    parser.add_argument('--schedules', nargs='+', choices=['baseline', 'safe', 'dynamic'], default=['baseline', 'safe', 'dynamic'])
    parser.add_argument('--output', default=str(Path(__file__).with_name('validation-results.json')))
    args = parser.parse_args()
    from solver import solve_multi
    from baseline_solver import coverage_points
    stations = np.asarray(coverage_points(), float)
    report = dict(official_simulator=False, validation_kind='independent_local_adversarial_cases',
                  policy=args.policy, python_version=sys.version.split()[0], numpy_version=np.__version__,
                  source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ['validate.py', 'solver.py', 'recovery.py', 'baseline_solver.py', 'environment.py']},
                  coverage=coverage_check(stations), results=[], errors=[])
    print('Running numeric-recovery and API-failure injection checks', flush=True)
    try:
        report['injection_checks'] = injected_failure_checks(solve_multi, stations)
        print(f"Passed {len(report['injection_checks'])} injected-failure checks", flush=True)
    except Exception as exc:
        report['errors'].append(dict(case='injected_failure_checks', exception=repr(exc), traceback=traceback.format_exc()))
        print(f'FAILED injection checks: {exc}', flush=True)
    for schedule in args.schedules:
        for name, sources, field in stress_cases():
            print(f'Running {schedule}/{name}/{args.policy}', flush=True)
            try:
                row, arena = run_case(solve_multi, stations, name, sources, schedule, args.policy, field)
                report['results'].append(row)
                if name == 'ten_sources_at_origin':
                    # Until the remote signal is found, all stations produce the same
                    # negative feedback as this probe; place one source at its last ring station.
                    last = row['station_visit_order'][-1]
                    late_sources = [dict(s) for s in sources[:-1]] + [source(sources[-1]['channel'], *(1.5 * stations[last]))]
                    late_row, late_arena = run_case(solve_multi, stations, 'last_boundary_source', late_sources, schedule, args.policy, 'extreme')
                    target_channel = sources[-1]['channel']
                    first_positive = next(e for e in late_arena.events if e['action'] == 'measure' and e['channel'] == target_channel and e['measure_result'] != 'no_signal')
                    discovered_at = station_index(first_positive['position'], stations)
                    late_row['late_source_discovery_station'] = discovered_at
                    late_row['late_source_found_at_last_station'] = discovered_at == late_row['station_visit_order'][-1]
                    assert late_row['late_source_found_at_last_station'], late_row['station_visit_order']
                    report['results'].append(late_row)
                print(f"Passed {schedule}/{name}: {row['cleared']}/{row['source_count']}, {row['runtime_s']:.2f}s runtime", flush=True)
            except Exception as exc:
                report['errors'].append(dict(case=name, schedule=schedule, exception=repr(exc), traceback=traceback.format_exc()))
                print(f'FAILED {schedule}/{name}: {exc}', flush=True)
            report['all_passed'] = not report['errors']
            Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    report['all_passed'] = (not report['errors'] and len(report['results']) == 5 * len(args.schedules)
                            and len(report.get('injection_checks', [])) == 15)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(all_passed=report['all_passed'], passed_cases=len(report['results']), errors=len(report['errors']), output=args.output), ensure_ascii=False), flush=True)
    raise SystemExit(0 if report['all_passed'] else 1)


if __name__ == '__main__':
    main()
