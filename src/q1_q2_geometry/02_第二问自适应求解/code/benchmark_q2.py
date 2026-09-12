"""Frozen, local-only second-observation benchmark for question 2.

The planner receives only public observations.  Source truth is used exclusively
by this file to produce feedback and to audit the returned geometry.  There are
no network, official simulator, clearance, or task-time operations here.

Usage: python benchmark_q2.py --output-dir results/q2_second_observation_01
The output directory must not already exist.  The experiment configuration is
written before importing and calling the planner, and is never tuned at runtime.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import platform
import random
import statistics
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


SEEDS = tuple(range(52000, 52060))
FIELDS = ("smooth", "extreme", "constant")
POLICIES = ("geometry", "midpoint", "nearest")
DISTANCE_GROUPS = ("near", "middle", "far")
LENGTH_TOLERANCE_M = 1e-6
PAIR_TOLERANCE_M = 1e-6


def json_value(value):
    """Convert common numeric containers without adding solver dependencies."""
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_value(value), ensure_ascii=False,
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    folder = Path(__file__).resolve().parent
    return {path.name: sha256(path) for path in sorted(folder.glob("*.py"))}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def generate_layouts():
    """Each independent seed draws its own source, R, bearing and first range."""
    layouts = []
    for index, seed in enumerate(SEEDS):
        rng = random.Random(seed)
        group = DISTANCE_GROUPS[index // 20]
        source_radius = 1800.0 * math.sqrt(rng.random())
        source_angle = rng.uniform(0.0, 2.0 * math.pi)
        source = [source_radius * math.cos(source_angle),
                  source_radius * math.sin(source_angle)]
        receive_radius = rng.uniform(1000.0, 1500.0)
        bounds = {"near": (6.0, 300.0), "middle": (300.0, 1000.0),
                  "far": (1000.0, receive_radius)}[group]
        first_range = rng.uniform(*bounds)
        first_angle = rng.uniform(0.0, 2.0 * math.pi)
        first_position = [source[0] + first_range * math.cos(first_angle),
                          source[1] + first_range * math.sin(first_angle)]
        layouts.append({
            "seed": seed, "distance_group": group,
            "source_position": source, "receive_radius_m": receive_radius,
            "first_position": first_position, "first_distance_m": first_range,
            "error_field_parameters": {
                "phase_x_m": rng.uniform(-1000.0, 1000.0),
                "phase_y_m": rng.uniform(-1000.0, 1000.0),
                "scale_x_m": 173.0, "scale_y_m": 211.0,
                "amplitude_deg": 0.99,
            },
        })
    return layouts


def spatial_error(position, layout, field):
    """A deterministic function of the location, shared by all three policies."""
    params = layout["error_field_parameters"]
    value = (math.sin((position[0] + params["phase_x_m"]) / params["scale_x_m"])
             * math.cos((position[1] + params["phase_y_m"]) / params["scale_y_m"]))
    if field == "smooth":
        return params["amplitude_deg"] * value
    if field == "extreme":
        return params["amplitude_deg"] if value >= 0.0 else -params["amplitude_deg"]
    if field == "constant":
        return params["amplitude_deg"]
    raise ValueError(f"Unknown fixed spatial error field: {field}")


def observe(position, layout, field):
    """Return separate public feedback and private benchmark audit metadata."""
    position = [float(coordinate) for coordinate in position]
    source = layout["source_position"]
    delta = [source[axis] - position[axis] for axis in (0, 1)]
    distance = math.hypot(*delta)
    observation = {"position": position}
    private = {"distance_to_source_m": distance,
               "receive_radius_m": layout["receive_radius_m"]}
    if distance > layout["receive_radius_m"]:
        observation["status"] = "no_signal"
    elif distance <= 5.0:
        observation["status"] = "near"
    else:
        true_angle = math.degrees(math.atan2(delta[1], delta[0])) % 360.0
        error = spatial_error(position, layout, field)
        angle = round((true_angle + error) % 360.0, 2) % 360.0
        observation.update(status="direction", angle=angle)
        private.update(true_bearing_deg=true_angle, prequantization_error_deg=error,
                       actual_output_error_deg=(angle - true_angle + 180.0) % 360.0 - 180.0)
    return observation, private


def cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def convex_hull(vertices):
    """Independent monotone chain; does not call any planner geometry helper."""
    points = sorted(set((float(point[0]), float(point[1])) for point in vertices))
    if len(points) <= 1:
        return points
    lower, upper = [], []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def segment_distance(point, a, b):
    length2 = sum((b[axis] - a[axis]) ** 2 for axis in (0, 1))
    if length2 == 0.0:
        return math.dist(point, a)
    fraction = sum((point[axis] - a[axis]) * (b[axis] - a[axis])
                   for axis in (0, 1)) / length2
    fraction = min(1.0, max(0.0, fraction))
    nearest = [a[axis] + fraction * (b[axis] - a[axis]) for axis in (0, 1)]
    return math.dist(point, nearest)


def hull_contains(hull, point, tolerance=LENGTH_TOLERANCE_M):
    if not hull:
        return False
    if len(hull) == 1:
        return math.dist(hull[0], point) <= tolerance
    if len(hull) == 2:
        return segment_distance(point, *hull) <= tolerance
    for index, start in enumerate(hull):
        end = hull[(index + 1) % len(hull)]
        if cross(start, end, point) < -tolerance * math.dist(start, end):
            return False
    return True


def independent_region_audit(region, truth):
    vertices = region["vertices"]
    hull = convex_hull(vertices)
    if not hull:
        raise ValueError("Reported region has no vertices")
    if not all(math.isfinite(value) for point in hull for value in point):
        raise ValueError("Reported region has non-finite vertices")
    actual_diameter = max(math.dist(a, b) for a in hull for b in hull)
    diameter = float(region["diameter_m"])
    circle = region["enclosing_circle"]
    center, radius = circle["center"], float(circle["radius_m"])
    reported_circle_cover = max(math.dist(point, center) for point in hull)
    finite_metrics = all(math.isfinite(value) for value in [diameter, radius, *center])
    return {
        "truth_in_region": hull_contains(hull, truth),
        "diameter_matches_independent_vertex_pairs": abs(diameter - actual_diameter) <= LENGTH_TOLERANCE_M,
        "independent_diameter_m": actual_diameter,
        "reported_circle_covers_all_vertices": reported_circle_cover <= radius + LENGTH_TOLERANCE_M,
        "truth_in_reported_circle": math.dist(truth, center) <= radius + LENGTH_TOLERANCE_M,
        "finite_nonnegative_metrics": finite_metrics and diameter >= 0.0 and radius >= 0.0,
        "reported_circle_excess_m": reported_circle_cover - radius,
    }


def clip_halfplane(hull, normal, bound):
    """Independent halfplane clipping of the convex hull, including degeneracy."""
    if not hull:
        return []
    signed = lambda point: sum(normal[axis] * point[axis] for axis in (0, 1)) - bound
    result = []
    previous, old = hull[-1], signed(hull[-1])
    for point in hull:
        current = signed(point)
        if (old <= 0.0) != (current <= 0.0):
            fraction = old / (old - current)
            result.append([previous[axis] + fraction * (point[axis] - previous[axis])
                           for axis in (0, 1)])
        if current <= 0.0:
            result.append(point)
        previous, old = point, current
    return convex_hull(result)


def independent_reception_audit(region, point, witness):
    hull = convex_hull(region["vertices"])
    normal = [2.0 * (point[axis] - witness[axis]) for axis in (0, 1)]
    bound = sum(value * value for value in point) - sum(value * value for value in witness)
    farther = clip_halfplane(hull, normal, bound)
    maximum = max((math.dist(vertex, point) for vertex in farther), default=None)
    return {"fixed_radius_reception_sufficient_check": maximum is None or maximum <= 1000.0 + LENGTH_TOLERANCE_M,
            "farther_subregion_max_distance_m": maximum,
            "farther_subregion_vertices": farther,
            "simple_1000m_check": max(math.dist(vertex, point) for vertex in hull) <= 1000.0 + LENGTH_TOLERANCE_M}


def audit_passed(audit):
    boolean_checks = [value for value in audit.values() if isinstance(value, bool)]
    return bool(boolean_checks) and all(boolean_checks)


def run_case(solver, layout, field, policy, first, first_private):
    identifier = f"{layout['seed']}-{field}-{policy}"
    row = {"case_id": identifier, "layout_seed": layout["seed"],
           "distance_group": layout["distance_group"], "error_field": field,
           "policy": policy, "first_observation": copy.deepcopy(first),
           "first_private_audit": copy.deepcopy(first_private), "passed": False}
    try:
        # Never pass layout, source coordinates, actual R or error function to the planner.
        plan = json_value(solver.plan_second_measurement(copy.deepcopy(first), policy=policy))
        row["plan"] = plan
        if plan.get("status") != "ready":
            raise ValueError(f"Planner did not return ready: {plan.get('status')}")
        point = plan["selected_point"]
        first_region = plan["region"]
        row["first_region_audit"] = independent_region_audit(first_region, layout["source_position"])
        reception = independent_reception_audit(first_region, point, first["position"])
        row["reception_audit"] = reception
        second, second_private = observe(point, layout, field)
        row.update(second_observation=second, second_private_audit=second_private,
                   second_signal_received=second["status"] in ("direction", "near"))
        updated = json_value(solver.update_second_measurement(copy.deepcopy(plan), copy.deepcopy(second)))
        row["updated"] = updated
        if second["status"] == "no_signal" or updated.get("status") not in ("direction", "near"):
            raise ValueError(f"Second feedback/update is unusable: {second['status']}/{updated.get('status')}")
        if updated["status"] != second["status"]:
            raise ValueError("Update status does not match the actual second observation")
        second_region = updated["region"]
        row["second_region_audit"] = independent_region_audit(second_region, layout["source_position"])
        first_hull = convex_hull(first_region["vertices"])
        diameter1, diameter2 = float(first_region["diameter_m"]), float(second_region["diameter_m"])
        ratio = diameter2 / diameter1 if diameter1 > LENGTH_TOLERANCE_M else None
        row["metrics"] = {
            "first_diameter_m": diameter1, "second_diameter_m": diameter2,
            "first_radius_m": float(first_region["enclosing_circle"]["radius_m"]),
            "second_radius_m": float(second_region["enclosing_circle"]["radius_m"]),
            "diameter_ratio": ratio, "candidate_count": len(plan["candidates"]),
            "second_step_distance_m": math.dist(first["position"], point),
        }
        update_ratio = updated.get("diameter_ratio")
        row["transition_audit"] = {
            "second_region_subset_first": all(hull_contains(first_hull, point) for point in second_region["vertices"]),
            "diameter_nonincreasing": diameter2 <= diameter1 + LENGTH_TOLERANCE_M,
            "radius_nonincreasing": float(second_region["enclosing_circle"]["radius_m"]) <= float(first_region["enclosing_circle"]["radius_m"]) + LENGTH_TOLERANCE_M,
            "reported_ratio_matches": ratio is None and update_ratio is None or
                                      ratio is not None and update_ratio is not None and
                                      abs(float(update_ratio) - ratio) <= 1e-9,
            "first_error_within_bound": abs(first_private["actual_output_error_deg"]) <= 1.0,
            "second_error_within_bound": second["status"] == "near" or abs(second_private["actual_output_error_deg"]) <= 1.0,
        }
        row["passed"] = (audit_passed(row["first_region_audit"])
                         and audit_passed(row["second_region_audit"])
                         and audit_passed(row["transition_audit"])
                         and reception["fixed_radius_reception_sufficient_check"]
                         and row["second_signal_received"])
    except Exception as exception:
        row.update(error_type=type(exception).__name__, error=str(exception),
                   traceback=traceback.format_exc())
    return row


def quantile(values, probability):
    values = sorted(values)
    if not values:
        return None
    index = probability * (len(values) - 1)
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (index - low) * (values[high] - values[low])


def describe(values):
    values = [value for value in values if value is not None]
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    return {"count": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "p90": quantile(values, 0.9),
            "min": min(values), "max": max(values)}


def policy_summary(rows):
    return {
        "attempted": len(rows), "passed": sum(row["passed"] for row in rows),
        "normal_second_feedback": sum(row.get("second_observation", {}).get("status") == "direction" for row in rows),
        "near_second_feedback": sum(row.get("second_observation", {}).get("status") == "near" for row in rows),
        "no_signal_second_feedback": sum(row.get("second_observation", {}).get("status") == "no_signal" for row in rows),
        "true_source_retained_after_second": sum(row.get("second_region_audit", {}).get("truth_in_region", False) for row in rows),
        "reception_certificates_passed": sum(row.get("reception_audit", {}).get("fixed_radius_reception_sufficient_check", False) for row in rows),
        "metric_denominator_note": "Metric summaries include cases with usable returned second-region metrics; audit failures are reported separately and are never treated as successes.",
        "metrics": {name: describe([row.get("metrics", {}).get(name) for row in rows])
                    for name in ("first_diameter_m", "second_diameter_m", "first_radius_m", "second_radius_m",
                                 "diameter_ratio", "candidate_count", "second_step_distance_m")},
        "worst_second_diameter_cases": [compact_case(row) for row in sorted(
            [row for row in rows if "metrics" in row],
            key=lambda row: row["metrics"]["second_diameter_m"], reverse=True)[:5]],
        "failed_case_ids": [row["case_id"] for row in rows if not row["passed"]],
    }


def compact_case(row):
    return {key: row[key] for key in ("case_id", "layout_seed", "distance_group", "error_field", "policy", "passed", "metrics") if key in row}


def paired_summary(rows, candidate, baseline):
    lookup = {(row["layout_seed"], row["error_field"], row["policy"]): row for row in rows}
    pairs = []
    missing = []
    conditions = sorted(set((row["layout_seed"], row["error_field"]) for row in rows))
    for seed, field in conditions:
        improved, original = lookup.get((seed, field, candidate)), lookup.get((seed, field, baseline))
        if improved is None or original is None or "metrics" not in improved or "metrics" not in original:
            missing.append({"layout_seed": seed, "error_field": field})
            continue
        difference = original["metrics"]["second_diameter_m"] - improved["metrics"]["second_diameter_m"]
        pairs.append({"layout_seed": seed, "error_field": field,
                      "distance_group": improved["distance_group"],
                      "candidate_case_id": improved["case_id"], "baseline_case_id": original["case_id"],
                      "candidate_diameter_m": improved["metrics"]["second_diameter_m"],
                      "baseline_diameter_m": original["metrics"]["second_diameter_m"],
                      "baseline_minus_candidate_diameter_m": difference,
                      "both_audits_passed": improved["passed"] and original["passed"]})
    differences = [pair["baseline_minus_candidate_diameter_m"] for pair in pairs]
    layout_means = [{"layout_seed": seed, "fields_included": len(subset),
                     "mean_baseline_minus_candidate_diameter_m": statistics.mean(subset)}
                    for seed in sorted(set(pair["layout_seed"] for pair in pairs))
                    if (subset := [pair["baseline_minus_candidate_diameter_m"] for pair in pairs if pair["layout_seed"] == seed])]
    return {"candidate": candidate, "baseline": baseline, "available_pairs": len(pairs),
            "pairs_with_both_audits_passed": sum(pair["both_audits_passed"] for pair in pairs),
            "missing_pairs": missing, "comparison_tolerance_m": PAIR_TOLERANCE_M,
            "candidate_smaller": sum(value > PAIR_TOLERANCE_M for value in differences),
            "equal_within_tolerance": sum(abs(value) <= PAIR_TOLERANCE_M for value in differences),
            "candidate_larger": sum(value < -PAIR_TOLERANCE_M for value in differences),
            "diameter_improvement_m": describe(differences),
            "independent_layout_average_improvement_m": describe([row["mean_baseline_minus_candidate_diameter_m"] for row in layout_means]),
            "layout_means": layout_means,
            "worst_candidate_regressions": sorted(pairs, key=lambda pair: pair["baseline_minus_candidate_diameter_m"])[:5],
            "largest_candidate_improvements": sorted(pairs, key=lambda pair: pair["baseline_minus_candidate_diameter_m"], reverse=True)[:5],
            "pairs": pairs}


def summarize(rows):
    expected = len(SEEDS) * len(FIELDS) * len(POLICIES)
    candidate_sets_match = []
    for seed in SEEDS:
        for field in FIELDS:
            subset = [row for row in rows if row["layout_seed"] == seed and row["error_field"] == field]
            sets = [sorted(tuple(candidate["point"]) for candidate in row["plan"]["candidates"])
                    for row in subset if "plan" in row and "candidates" in row["plan"]]
            if len(sets) != len(POLICIES) or any(points != sets[0] for points in sets):
                candidate_sets_match.append({"layout_seed": seed, "error_field": field})
    return {
        "scope": "Local question 2 benchmark ending immediately after the second observation.",
        "independent_layouts": len(SEEDS), "paired_layout_error_conditions": len(SEEDS) * len(FIELDS),
        "policies_per_condition": len(POLICIES), "expected_policy_runs": expected,
        "actual_policy_runs": len(rows),
        "sample_unit_note": "There are 60 independent layouts and 180 paired layout/error conditions. The 540 policy runs are not 540 independent samples; the three error fields on each layout are also dependent.",
        "official_simulator_contacted": False, "all_audits_passed": len(rows) == expected and all(row["passed"] for row in rows) and not candidate_sets_match,
        "candidate_set_mismatch_conditions": candidate_sets_match,
        "policies": {policy: policy_summary([row for row in rows if row["policy"] == policy]) for policy in POLICIES},
        "by_distance_group": {group: {policy: policy_summary([row for row in rows if row["distance_group"] == group and row["policy"] == policy]) for policy in POLICIES} for group in DISTANCE_GROUPS},
        "by_error_field": {field: {policy: policy_summary([row for row in rows if row["error_field"] == field and row["policy"] == policy]) for policy in POLICIES} for field in FIELDS},
        "paired_comparisons": {f"{candidate}_vs_{baseline}": paired_summary(rows, candidate, baseline)
                               for candidate, baseline in (("geometry", "midpoint"), ("geometry", "nearest"), ("midpoint", "nearest"))},
        "failure_case_ids": [row["case_id"] for row in rows if not row["passed"]],
        "interpretation_limits": [
            "These scenarios are a frozen constructed local test distribution, not the official source or error distribution.",
            "No repeated sampling at the same location or averaging of independent observation noise is used.",
            "The nearest baseline is a geometric distance rule on the same reliable candidate set; no movement speed or task cost is used.",
            "Audit containment tolerances are in metres, and the independent convex hull audit does not call solver geometry routines.",
            "No separate fixed boundary-stress suite is included in these means.",
            "The enclosing-circle audit certifies containment, not independent global minimality of the reported circle.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New output directory; existing directories are refused.")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        parser.error(f"Output path already exists; refusing to overwrite frozen evidence: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "cases").mkdir()
    before_hashes = source_hashes()
    if "q2_solver.py" not in before_hashes:
        parser.error("q2_solver.py must be available before running this benchmark")
    configuration = {
        "protocol": "q2-second-observation-v1", "layout_seeds": list(SEEDS),
        "independent_layout_count": 60, "distance_groups": {
            "near": {"count": 20, "first_distance_interval_m": [6, 300]},
            "middle": {"count": 20, "first_distance_interval_m": [300, 1000]},
            "far": {"count": 20, "first_distance_interval_m": [1000, "R"]}},
        "source_distribution": "Uniform by area in the radius-1800 target disk, independently drawn for each layout seed.",
        "receive_radius_distribution_m": [1000, 1500],
        "first_station": "Uniform angular direction relative to source and uniform distance within its fixed stratum; may lie outside the target disk.",
        "policies": list(POLICIES), "error_fields": list(FIELDS),
        "smooth_error_formula_deg": "0.99*sin((x+phase_x)/173)*cos((y+phase_y)/211)",
        "extreme_error_formula_deg": "+0.99 when the smooth field is nonnegative, otherwise -0.99",
        "constant_error_formula_deg": "+0.99 at every location",
        "output_quantization": "Python round(angle modulo 360, 2), then modulo 360",
        "near_distance_m": 5, "solver_angle_half_width_deg": 1,
        "solver_circle_sides": 128, "solver_api_parameters": "Defaults except policy; no truth passed to either solver API.",
        "length_tolerance_m": LENGTH_TOLERANCE_M, "pair_tolerance_m": PAIR_TOLERANCE_M,
        "layouts": generate_layouts(), "source_hashes_before": before_hashes,
        "frozen_before_planning": True,
    }
    write_json(output_dir / "configuration.json", configuration)
    manifest = {"protocol": configuration["protocol"], "status": "running", "started_utc": utc_now(),
                "python_version": platform.python_version(), "platform": platform.platform(),
                "official_simulator_contacted": False, "source_hashes_before": before_hashes,
                "configuration_sha256": sha256(output_dir / "configuration.json")}
    write_json(output_dir / "manifest.json", manifest)
    rows = []
    try:
        solver = importlib.import_module("q2_solver")
        for layout_index, layout in enumerate(configuration["layouts"], 1):
            for field in FIELDS:
                first, private = observe(layout["first_position"], layout, field)
                if first["status"] != "direction":
                    raise AssertionError("Frozen layout generator must produce a normal first observation")
                for policy in POLICIES:
                    row = run_case(solver, layout, field, policy, first, private)
                    rows.append(row)
                    write_json(output_dir / "cases" / f"{row['case_id']}.json", row)
            print(json.dumps({"completed_layouts": layout_index, "total_layouts": len(SEEDS),
                              "policy_runs": len(rows), "audit_failures": sum(not row["passed"] for row in rows)}, ensure_ascii=False), flush=True)
        summary = summarize(rows)
        write_json(output_dir / "summary.json", summary)
        manifest["status"] = "complete" if summary["all_audits_passed"] else "complete_with_audit_failures"
        manifest["all_audits_passed"] = summary["all_audits_passed"]
    except BaseException as exception:
        manifest.update(status="interrupted" if isinstance(exception, KeyboardInterrupt) else "failed",
                        error_type=type(exception).__name__, error=str(exception), traceback=traceback.format_exc())
        raise
    finally:
        after_hashes = source_hashes()
        manifest.update(finished_utc=utc_now(), source_hashes_after=after_hashes,
                        source_snapshot_stable=before_hashes == after_hashes,
                        completed_policy_runs=len(rows))
        if not manifest["source_snapshot_stable"]:
            manifest["status"] = "invalid_source_snapshot_changed"
        manifest["data_file_sha256"] = {path.relative_to(output_dir).as_posix(): sha256(path)
                                        for path in sorted(output_dir.rglob("*.json"))
                                        if path.name != "manifest.json"}
        write_json(output_dir / "manifest.json", manifest)
    return 0 if manifest.get("all_audits_passed") and manifest["source_snapshot_stable"] else 2


if __name__ == "__main__":
    sys.exit(main())
