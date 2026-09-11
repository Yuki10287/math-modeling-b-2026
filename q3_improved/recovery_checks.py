"""Local checks for recovery guarantees, baseline preservation and API isolation."""
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np

import baseline_solver as baseline
from environment import LocalArena
import recovery


def source(channel, position):
    return dict(channel=channel, position=np.asarray(position).tolist(), radius=1500.0)


def check_grid():
    # Continuous rectangle covering bound, independent of any sampled simulator.
    lateral_bound = 1500 * math.sin(math.radians(1))
    assert lateral_bound < 30
    assert math.hypot(10, 15) < 20
    checked = 0
    for angle in (0, 90, 180, 359.99):
        a = math.radians(angle)
        u = np.array([math.cos(a), math.sin(a)])
        v = np.array([-u[1], u[0]])
        anchor = np.array([200., -100.])
        for x0, y0 in ((0, -15), (0, 15), (1500, -15), (1500, 15)):
            current = anchor + x0*u + y0*v
            path = recovery.optical_grid(anchor, angle, current)
            assert path.shape == (152, 2)
            assert len(np.unique(np.round(path, 8), axis=0)) == 152
            assert np.linalg.norm(path[0]-current) < 1e-9
            assert abs(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()-3030) < 1e-8
            # Sample all forward half-grid boundaries and both lateral extremes;
            # the continuous proof above supplies the gap between samples.
            rect = np.array([(x, y) for x in np.arange(0, 1501, 10)
                             for y in (-lateral_bound, 0, lateral_bound)])
            world = anchor + rect[:, :1]*u + rect[:, 1:]*v
            distances = np.linalg.norm(world[:, None, :]-path[None, :, :], axis=2)
            assert distances.min(axis=1).max() <= math.sqrt(325)+1e-8
            checked += 1
    return dict(route_orientations_checked=checked, points=152,
                internal_path_m=3030, continuous_cover_bound_m=math.sqrt(325))


def check_forced_recovery():
    checked = 0
    largest_calls = 0
    anchor = np.array([200., -100.])
    for bearing in (0., 90., 179.99, 359.99):
        for error in (-1., 0., 1.):
            for radius in (5.001, 20., 500., 1000., 1499.999, 1500.):
                a = math.radians(bearing+error)
                g = anchor+radius*np.array([math.cos(a), math.sin(a)])
                env = LocalArena([source(3, g)], start=anchor+[80., -70.])
                env.channel = 7
                trace = []
                first = dict(measure_result="direction", svd_deg=bearing)
                assert recovery.locate_source(env, 3, anchor, first, trace=trace, max_steps=0)
                assert env.evaluation()["all_cleared"]
                assert env.channel == 7 and env.counts["switch"] == 0
                assert env.counts["measure"] == 0
                assert all(row["action"] == "clear" for row in env.events)
                assert trace[0]["phase"] == "fallback_start"
                assert trace[-1]["fallback_result"] == "success"
                assert 1 <= trace[-1]["calls"] <= 152
                assert env.time_s <= trace[0]["expected_max_virtual_s"]+1e-8
                largest_calls = max(largest_calls, trace[-1]["calls"])
                checked += 1
    # An impossible/stale original detection must not be marked as complete.
    env = LocalArena([], start=(0., -15.))
    trace = []
    assert not recovery.locate_source(env, 3, np.zeros(2),
        dict(measure_result="direction", svd_deg=0.), trace=trace, max_steps=0)
    assert env.counts["clear_fail"] == 152 and env.counts["clear_success"] == 0
    assert abs(env.distance_m-3030.) < 1e-8
    assert abs(env.time_s-(3030/5+152*3)) < 1e-8
    assert trace[-1]["fallback_result"] == "model_inconsistent"
    assert 3030/5+151*3+5 == recovery.GRID_MAX_INTERNAL_TIME_S
    return dict(legal_forced_cases=checked, largest_clear_count=largest_calls,
                exhausted_calls=152, exhausted_failure_virtual_s=env.time_s,
                worst_success_virtual_bound_s=1064)


def check_baseline_equivalence():
    compared = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        a = rng.uniform(0, 2*math.pi)
        g = rng.uniform(6, 1499)*np.array([math.cos(a), math.sin(a)])
        for field in ("smooth", "hash", "extreme"):
            for policy in ("geometry", "time"):
                envs = [LocalArena([source(3, g)], seed, field) for _ in range(2)]
                first = [env.measure([0., 0.], 3) for env in envs]
                # Recreate a history-based call after the robot has moved away.
                for env in envs:
                    env.measure([120., -75.], 8)
                traces = [[], []]
                old = baseline.solve_source_from_history(envs[0], 3, np.zeros(2), first[0], policy, traces[0])
                new = recovery.locate_source(envs[1], 3, np.zeros(2), first[1], policy, traces[1])
                assert old is True and new is True
                assert envs[0].events == envs[1].events
                assert traces[0] == traces[1]
                assert envs[0].evaluation() == envs[1].evaluation()
                compared += 1
    # near at the first observation preserves the one-clear baseline behavior.
    env = LocalArena([source(3, [3., 4.])])
    trace = []
    assert recovery.locate_source(env, 3, np.zeros(2), {"measure_result": "near"}, trace=trace)
    assert len(env.events) == 1 and env.events[0]["clear_result"] == "success"
    assert trace == []
    return dict(normal_runs_exactly_equal=compared, first_near_checked=True)


def check_model_failures():
    cases = []
    first = {"measure_result": "direction", "svd_deg": 0.}
    for function in ("initial_belief", "mec", "choose_measure", "update_belief"):
        env = LocalArena([source(3, [1300., 0.])])
        trace = []
        with patch.object(baseline, function, side_effect=RuntimeError("injected geometry failure")):
            assert recovery.locate_source(env, 3, np.zeros(2), first, trace=trace)
        assert trace[-1]["fallback_result"] == "success"
        cases.append(function)
    env = LocalArena([source(3, [1300., 0.])])
    trace = []
    with patch.object(baseline, "nearest_certified_clear", return_value=np.zeros(2)):
        assert recovery.locate_source(env, 3, np.zeros(2), first, trace=trace)
    assert any(row.get("reason") == "certified_clear_failed" for row in trace)
    cases.append("certified_clear_failed")
    return dict(recovered_model_failure_paths=cases)


def check_api_isolation():
    marker = RuntimeError("synthetic transport failure")
    first = {"measure_result": "direction", "svd_deg": 0.}
    checked = []

    class ExplodingArena(LocalArena):
        def measure(self, q, channel):
            raise marker
        def clear(self, q, channel):
            raise marker

    for name, steps, clear_point in (("normal_measure", 30, None),
                                      ("normal_clear", 30, np.zeros(2)),
                                      ("fallback_clear", 0, None)):
        env = ExplodingArena([source(3, [1300., 0.])])
        trace = []
        try:
            with patch.object(baseline, "nearest_certified_clear", return_value=clear_point):
                recovery.locate_source(env, 3, np.zeros(2), first, trace=trace, max_steps=steps)
        except RuntimeError as exc:
            assert exc is marker
        else:
            raise AssertionError("Transport exception swallowed")
        assert not any(row["phase"] == "fallback_end" for row in trace)
        if steps:
            assert not any(row["phase"] == "fallback_start" for row in trace)
        checked.append(name)

    class RejectedArena(LocalArena):
        def measure(self, q, channel):
            return {"accepted": False, "virtual_time_s": 0}
        def clear(self, q, channel):
            return {"accepted": False, "virtual_time_s": 0}

    for steps in (0, 30):
        env = RejectedArena([source(3, [1300., 0.])])
        try:
            recovery.locate_source(env, 3, np.zeros(2), first, max_steps=steps)
        except recovery.APIResponseError:
            pass
        else:
            raise AssertionError("Rejected API response treated as failed geometry")
        assert env.time_s == 0
        checked.append(f"rejected_response_steps_{steps}")
    return dict(api_failure_paths_propagated=checked)


def main():
    result = dict(local_only=True, grid=check_grid(), recovery=check_forced_recovery(),
                  baseline=check_baseline_equivalence(), model=check_model_failures(),
                  api=check_api_isolation())
    path = Path(__file__).with_name("recovery_check_results.json")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
