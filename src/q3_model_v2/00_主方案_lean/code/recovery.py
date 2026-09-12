"""Question 3 localization with a finite optical fallback.

The usual localization path is the unchanged omnidirectional baseline.  Its
reception certificate does not apply to directional sources in Question 4.
The fallback uses only the original positive bearing, never a possibly damaged
later polygon.  It needs the stated 1500 m range and total bearing error <= 1 deg.
For v2 the same grid also covers the enlarged 1.0050001 degree observation
bound, since 1500*sin(1.0050001 deg) < 30 m. It bounds actions and virtual
time; it cannot bound HTTP or real running time.
"""
from __future__ import annotations

import math

import numpy as np

import baseline_solver as baseline


GRID_MAX_CALLS = 152
GRID_INTERNAL_DISTANCE_M = 3030.0
GRID_MAX_INTERNAL_TIME_S = 1064.0
GRID_COVER_RADIUS_M = math.sqrt(10.0**2 + 15.0**2)
_GEOMETRY_ERRORS = (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError)


class APIResponseError(RuntimeError):
    """An unaccepted or malformed response; never treated as a geometry failure."""


def _feedback(reply, key, allowed):
    if not isinstance(reply, dict):
        raise APIResponseError("API response is not an object")
    if "accepted" in reply and reply["accepted"] is not True:
        raise APIResponseError("API did not accept the action")
    value = reply.get(key)
    if value not in allowed:
        raise APIResponseError(f"Invalid {key}: {value!r}")
    if key == "measure_result" and value == "direction":
        angle = reply.get("svd_deg")
        if isinstance(angle, bool) or not isinstance(angle, (float, int)):
            raise APIResponseError("Missing or invalid bearing")
        if not math.isfinite(angle) or not 0 <= angle < 360:
            raise APIResponseError("Bearing is outside [0, 360)")
    return value


def _clear_feedback(reply):
    return _feedback(reply, "clear_result", ("success", "no_target_in_range"))


def _measure_feedback(reply):
    return _feedback(reply, "measure_result", ("near", "direction", "no_signal"))


def _point(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("Nonfinite or malformed geometry point")
    return q


def _polygon(P):
    if P.ndim != 2 or P.shape[1] != 2 or len(P) == 0 or not np.isfinite(P).all():
        raise ValueError("Empty or nonfinite belief")
    return P


def optical_grid(first_position, bearing_deg, current_position=None):
    """Return all 152 points in one of four nearest-start snake orientations.

    In bearing coordinates the source has x in [0, 1500] and |y| <=
    1500*sin(1 deg) < 30.  The two rows y = -15, +15 and x = 0,20,...,1500
    give |dx| <= 10 and |dy| <= 15, hence distance <= sqrt(325) < 20.
    Each row has 76 points INCLUDING BOTH endpoints.  Snake length is
    1500 + 30 + 1500 = 3030 m.  Flipping either axis preserves both facts.
    """
    anchor = _point(first_position)
    a = math.radians(float(bearing_deg))
    u = np.array([math.cos(a), math.sin(a)])
    v = np.array([-u[1], u[0]])
    x = np.arange(76, dtype=float) * 20.0
    paths = []
    for first_y in (-15.0, 15.0):
        for reverse_x in (False, True):
            first_x = x[::-1] if reverse_x else x
            local = np.vstack((
                np.column_stack((first_x, np.full(76, first_y))),
                np.column_stack((first_x[::-1], np.full(76, -first_y))),
            ))
            paths.append(anchor + local[:, :1] * u + local[:, 1:] * v)
    current = anchor if current_position is None else _point(current_position)
    return min(paths, key=lambda path: float(np.linalg.norm(path[0] - current)))


def _recover(api, channel, first_position, bearing, trace, reason):
    start = _point(api.position).copy()
    path = optical_grid(first_position, bearing, start)
    approach = float(np.linalg.norm(path[0] - start))
    before = getattr(api, "time_s", None)
    trace.append(dict(
        phase="fallback_start", channel=channel, reason=reason,
        position=start.tolist(), first_position=np.asarray(first_position).tolist(),
        bearing_deg=float(bearing), expected_max_calls=GRID_MAX_CALLS,
        expected_max_internal_path_m=GRID_INTERNAL_DISTANCE_M,
        approach_distance_m=approach,
        expected_max_virtual_s=GRID_MAX_INTERNAL_TIME_S + approach / 5.0,
        cover_radius_m=GRID_COVER_RADIUS_M,
    ))
    distance = 0.0
    previous = start
    success = False
    calls = 0
    for calls, q in enumerate(path, start=1):
        # API calls and response validation deliberately sit outside all geometry
        # exception handlers: a lost/rejected request is not a failed clear.
        result = _clear_feedback(api.clear(q, channel))
        distance += float(np.linalg.norm(q - previous))
        previous = q
        trace.append(dict(
            phase="fallback_result", channel=channel, reason=reason, calls=calls,
            expected_max_calls=GRID_MAX_CALLS, position=q.tolist(), clear_result=result,
        ))
        if result == "success":
            success = True
            break
    end = dict(
        phase="fallback_end", channel=channel, reason=reason, calls=calls,
        expected_max_calls=GRID_MAX_CALLS, success=success, distance_m=distance,
        fallback_result="success" if success else "model_inconsistent",
    )
    after = getattr(api, "time_s", None)
    if before is not None and after is not None:
        end["virtual_time_s"] = float(after - before)
    if not success:
        end["detail"] = "All 152 covering clear attempts failed; original bearing/range/state is inconsistent."
    trace.append(end)
    return success


def locate_source(api, channel, first_position, first, policy="time", trace=None, max_steps=30):
    """Clear a previously detected source, preserving normal baseline actions.

    Numerical/model failures or the local iteration limit invoke the optical
    fallback.  API/transport failures propagate to the caller.  A False result
    means no successful clearance, and must never be counted as completion.
    """
    if policy not in ("time", "geometry", "midpoint"):
        raise ValueError(f"Unknown policy: {policy}")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    trace = [] if trace is None else trace
    anchor = _point(first_position).copy()
    initial_result = _measure_feedback(first)
    if initial_result == "no_signal":
        raise ValueError("Localization requires an original positive detection")
    if initial_result == "near":
        # This original observation already certifies a successful optical clear.
        success = _clear_feedback(api.clear(anchor, channel)) == "success"
        if not success:
            trace.append(dict(phase="localization_failure", channel=channel,
                              reason="near_clear_failed", model_inconsistent=True))
        return success
    bearing = first["svd_deg"]
    try:
        P = _polygon(baseline.initial_belief(anchor, bearing))
        witness = anchor.copy()
    except _GEOMETRY_ERRORS as exc:
        return _recover(api, channel, anchor, bearing, trace, f"initial_geometry: {exc}")

    for step in range(max_steps):
        try:
            s = _point(api.position).copy()
            c, r = baseline.mec(P)
            if not math.isfinite(r):
                raise ValueError("Nonfinite belief radius")
            trace.append(dict(channel=channel, step=step, position=s.tolist(),
                              radius_m=r, polygon=P.tolist(), phase="localize"))
            q = baseline.nearest_certified_clear(P, s)
            if q is not None:
                q = _point(q)
        except _GEOMETRY_ERRORS as exc:
            return _recover(api, channel, anchor, bearing, trace, f"clear_geometry: {exc}")
        if q is not None:
            if _clear_feedback(api.clear(q, channel)) == "success":
                return True
            return _recover(api, channel, anchor, bearing, trace, "certified_clear_failed")
        try:
            q = _point(baseline.choose_measure(P, s, policy, witness))
        except _GEOMETRY_ERRORS as exc:
            return _recover(api, channel, anchor, bearing, trace, f"measure_geometry: {exc}")
        result = api.measure(q, channel)
        result_kind = _measure_feedback(result)
        if result_kind == "near":
            if _clear_feedback(api.clear(q, channel)) == "success":
                return True
            return _recover(api, channel, anchor, bearing, trace, "near_clear_failed")
        if result_kind == "no_signal":
            return _recover(api, channel, anchor, bearing, trace, "certified_reception_failed")
        try:
            P = _polygon(baseline.update_belief(P, q, result["svd_deg"]))
            witness = q.copy()
        except _GEOMETRY_ERRORS as exc:
            return _recover(api, channel, anchor, bearing, trace, f"update_geometry: {exc}")
    return _recover(api, channel, anchor, bearing, trace, "localization_limit")
