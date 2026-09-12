"""Independently audit saved Q3 final-version practice logs; never use HTTP.

Run from the repository root, with the original private logs available::

    python -X utf8 src/q3_model_v2/05_官方演练复盘/code/audit_final_practice_batch10.py

Uses only the standard library, reads frozen hashes rather than importing the
solver, and rebuilds coverage from accepted no_signal responses. Source counts
come from the user's simulator-end-screen transcription in input_manifest.json.
These are practice results, not formal tests. No hidden coordinates are inferred.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[4]
MODEL = ROOT / "src/q3_model_v2"
GROUP = MODEL / "05_官方演练复盘"
RUNTIME_FILES = (
    "baseline_solver.py", "belief_model.py", "coverage_model.py", "geometry.py",
    "local_policy.py", "official_client.py", "recovery.py", "scan_planning.py",
    "solver.py", "v1_solver.py",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def independent_cells():
    """Closed 20 m squares intersecting the closed 1800 m target disk.

    A square intersects the disk iff the closest point of the square to the
    origin is no farther than 1800 m. Boundary-touching squares are included.
    """
    cells = []
    for iy in range(-90, 90):
        for ix in range(-90, 90):
            x, y = ix * 20 + 10, iy * 20 + 10
            if max(abs(x) - 10, 0) ** 2 + max(abs(y) - 10, 0) ** 2 <= 1800**2:
                cells.append((x, y))
    return cells


def independent_cover(points, cells):
    """Two whole-square tests, both independent of the runtime implementation.

    First apply a center-distance plus half-diagonal bound. Also check the
    exact farthest square corner from every observation. Each square must fit
    in ONE observed disk; covering merely its corners with different disks is
    insufficient. A closed 1000 m no-signal disk is a valid exclusion under Q3.
    """
    conservative_uncovered = exact_uncovered = 0
    worst_bound = worst_corner = 0.0
    for x, y in cells:
        best_bound = best_corner = math.inf
        for px, py in points:
            dx, dy = abs(x - px), abs(y - py)
            best_bound = min(best_bound, math.hypot(dx, dy) + math.sqrt(200))
            best_corner = min(best_corner, math.hypot(dx + 10, dy + 10))
        conservative_uncovered += best_bound > 1000 - 1e-9
        exact_uncovered += best_corner > 1000 - 1e-9
        worst_bound = max(worst_bound, best_bound)
        worst_corner = max(worst_corner, best_corner)
    return {
        "accepted_distinct_no_signal_points": len(points),
        "total_intersecting_cells": len(cells),
        "center_plus_half_diagonal_uncovered_cells": conservative_uncovered,
        "exact_farthest_corner_uncovered_cells": exact_uncovered,
        "max_best_center_plus_half_diagonal_m": worst_bound if points else None,
        "max_best_farthest_corner_m": worst_corner if points else None,
        "continuous_absence_proved": conservative_uncovered == exact_uncovered == 0,
    }


def audit_case(path, expected_total, current_hashes, frozen_hashes, cells):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
               if line.strip()]
    summaries = [r for r in records if r.get("type") == "session_summary"]
    if len(summaries) != 1:
        raise ValueError(f"{path.name}: expected exactly one session summary")
    summary = summaries[0]
    failures = []

    def check(condition, name):
        if not condition:
            failures.append(name)

    seen_requests = {}
    duplicate_ids = conflicts = rejected = 0
    accepted = []
    for row in records:
        if "path" not in row or not isinstance(row.get("response"), dict):
            continue
        request_id = row["request"].get("request_id")
        check(bool(request_id), "missing_request_id")
        if request_id in seen_requests:
            duplicate_ids += 1
            earlier = seen_requests[request_id]
            if row != earlier:
                conflicts += 1
            continue
        seen_requests[request_id] = row
        if row["response"].get("accepted") is True:
            accepted.append(row)
        else:
            rejected += 1

    errors = [r for r in records if r.get("type") == "client_error"]
    retry_records = [r for r in records if "retry" in str(r.get("type", "")).lower()]
    transport_errors = [r for r in errors if r.get("phase") == "transport"]
    later_attempt_errors = [r for r in errors if isinstance(r.get("attempt"), int) and r["attempt"] > 1]
    timeout_records = [r for r in errors if "timeout" in str(r).lower()]
    check(conflicts == 0, "conflicting_duplicate_request_ids")
    check(not errors and rejected == 0, "client_error_or_rejected_response")
    enters = [r for r in accepted if r["path"] == "/enter"]
    exits = [r for r in accepted if r["path"] == "/exit"]
    check(len(enters) == len(exits) == 1, "enter_exit_count")
    check(accepted[0]["path"] == "/enter" and accepted[-1]["path"] == "/exit", "session_order")
    if len(enters) != 1 or len(exits) != 1:
        raise ValueError(f"{path.name}: cannot audit missing or repeated enter/exit")

    position, current_channel, observed_time = (0.0, 0.0), 1, 0.0
    expected_microseconds = 0
    max_fee_error = max_microsecond_error = 0
    distance_m = 0.0
    fees = Counter()
    outcomes = Counter()
    clears = []
    negative_points = defaultdict(list)
    measure_seen, bearing_seen = set(), set()
    repeat_measure = repeat_direction = after_clear = 0
    last_clear_time, last_clear_action = None, None
    action_count = 0
    for row in accepted:
        action, request, response = row["path"], row["request"], row["response"]
        new_time = float(response["virtual_time_s"])
        if action in ("/enter", "/exit"):
            check(new_time == observed_time, "enter_or_exit_advances_virtual_clock")
            continue
        check(action in ("/measure", "/clear"), "unexpected_action")
        action_count += 1
        p = request["position"]
        q = (float(p["x"]), float(p["y"]))
        channel = request["channel"]
        check(all(math.isfinite(x) and abs(x) <= 2_000_000 for x in q), "invalid_position")
        check(channel in range(1, 21), "invalid_channel")
        distance = math.dist(position, q)
        distance_m += distance
        move_fee = distance / 5
        fees["movement_s"] += move_fee
        fee = move_fee
        after_clear += channel in clears
        if action == "/measure":
            outcome = response.get("measure_result")
            outcomes[f"measure_{outcome}"] += 1
            check(outcome in ("direction", "near", "no_signal"), "unknown_measure_result")
            switch = int(channel != current_channel)
            fees["measurement_s"] += 5
            fees["switch_s"] += switch
            fee += 5 + switch
            current_channel = channel
            key = (channel, *q)
            repeat_measure += key in measure_seen
            measure_seen.add(key)
            if outcome == "direction":
                check(isinstance(response.get("svd_deg"), (int, float)) and
                      math.isfinite(response["svd_deg"]) and 0 <= response["svd_deg"] < 360,
                      "invalid_direction_value")
                repeat_direction += key in bearing_seen
                bearing_seen.add(key)
            if outcome == "no_signal" and q not in negative_points[channel]:
                negative_points[channel].append(q)
        else:
            success = response.get("clear_result") == "success"
            outcomes[f"clear_{response.get('clear_result')}"] += 1
            check(response.get("clear_result") in ("success", "no_target_in_range"), "unknown_clear_result")
            fee += 5 if success else 3
            fees["successful_clear_s" if success else "failed_clear_s"] += 5 if success else 3
            fees["movement_to_successful_clear_s" if success else "movement_to_failed_clear_s"] += move_fee
            if success:
                clears.append(channel)
                last_clear_time, last_clear_action = new_time, action_count
        max_fee_error = max(max_fee_error, abs(new_time - observed_time - fee))
        expected_microseconds += round(fee * 1_000_000)
        max_microsecond_error = max(max_microsecond_error,
                                   abs(round(new_time * 1_000_000) - expected_microseconds))
        check(new_time >= observed_time, "virtual_clock_not_monotone")
        position, observed_time = q, new_time
    check(max_microsecond_error == 0, "microsecond_accumulated_fee_mismatch")
    check(len(set(clears)) == len(clears), "duplicate_successful_channel")
    check(len(clears) == expected_total, "user_reported_total_mismatch")
    check(summary.get("total_virtual_s") == observed_time, "summary_virtual_time_mismatch")
    check(summary.get("successful_clear_count") == summary.get("cleared") == len(clears), "summary_clear_count_mismatch")
    check(summary.get("complete") is True and summary.get("status") == "complete", "summary_incomplete")
    check(summary.get("exit_accepted") is True and exits[0]["response"].get("exit_reason") == "user_exit", "abnormal_exit")
    check(summary.get("policy") == "time" and summary.get("schedule") == "lean", "unexpected_policy")
    check(summary.get("source_sha256") == current_hashes == frozen_hashes, "source_hash_mismatch")

    cleared = set(clears)
    missing = set(range(1, 21)) - cleared
    certificate = summary.get("completion_certificate", {})
    check(certificate.get("valid") is True, "summary_certificate_invalid")
    check(certificate.get("unresolved_channels") == [], "unresolved_channels")
    check(set(certificate.get("cleared_channels", [])) == cleared, "certificate_cleared_mismatch")
    coverage = {}
    if len(cleared) == 16:
        independent_basis = "count_upper_bound"
        termination_valid = True
        check(certificate.get("basis") == independent_basis, "termination_basis_mismatch")
        check(set(certificate.get("absent_by_count_channels", [])) == missing, "absent_by_count_mismatch")
        check(certificate.get("absent_channels") == [], "unexpected_geometric_absence_claim")
    else:
        independent_basis = "independent_whole_cell_negative_disk_cover"
        for channel in sorted(missing):
            points = negative_points[channel]
            result = independent_cover(points, cells)
            reported = certificate.get("channel_coverage", {}).get(str(channel), {})
            result["reported_points_equal_accepted_feedback"] = (
                reported.get("negative_points") == [list(p) for p in points])
            result["reported_cell_counts_match"] = (
                reported.get("total_cells") == len(cells) and
                reported.get("uncovered_cells") == result["center_plus_half_diagonal_uncovered_cells"] and
                reported.get("excluded_cells") == len(cells) - result["center_plus_half_diagonal_uncovered_cells"])
            coverage[str(channel)] = result
            check(result["continuous_absence_proved"], f"unproved_absence_channel_{channel}")
            check(result["reported_points_equal_accepted_feedback"] and result["reported_cell_counts_match"], f"coverage_summary_mismatch_channel_{channel}")
        termination_valid = all(v["continuous_absence_proved"] for v in coverage.values())
        check(certificate.get("basis") == "adaptive_coverage", "termination_basis_mismatch")
        check(set(certificate.get("absent_channels", [])) == missing, "absent_channel_mismatch")
        check(certificate.get("absent_by_count_channels") == [], "invalid_count_shortcut")

    real_elapsed = (exits[0]["response"]["real_timestamp_ms"] - enters[0]["response"]["real_timestamp_ms"]) / 1000
    available = enters[0]["response"].get("remaining_real_duration_s")
    check(0 <= real_elapsed < available, "real_time_budget_exceeded")
    check(observed_time < enters[0]["response"]["max_virtual_duration_s"], "virtual_time_budget_exceeded")
    check(summary.get("official_program_runtime_s") is None, "unexpected_official_runtime_source")
    return {
        "case": path.stem.removeprefix("q3-practice-final-"),
        "input_log_sha256": sha256(path),
        "input_belief_trace_sha256": sha256(Path(str(path) + ".beliefs.json")),
        "user_reported_sources": expected_total,
        "user_reported_omnidirectional_sources": expected_total,
        "user_reported_directional_sources": 0,
        "successful_unique_clears": len(cleared), "clear_fraction": len(cleared) / expected_total,
        "source_hashes_match_current_and_frozen": summary.get("source_sha256") == current_hashes == frozen_hashes,
        "policy": summary.get("policy"), "schedule": summary.get("schedule"),
        "accepted_unique_action_count": action_count,
        "outcomes": dict(sorted(outcomes.items())),
        "distance_m": distance_m,
        "virtual_total_s": observed_time,
        "virtual_s_per_cleared": observed_time / len(cleared),
        "independent_unrounded_total_s": sum(fees[k] for k in ("movement_s", "measurement_s", "switch_s", "successful_clear_s", "failed_clear_s")),
        "independent_microsecond_total_s": expected_microseconds / 1_000_000,
        "max_per_action_unrounded_fee_error_s": max_fee_error,
        "max_cumulative_microsecond_fee_error_us": max_microsecond_error,
        "time_components": dict(sorted(fees.items())),
        "last_clear_virtual_s": last_clear_time,
        "virtual_s_after_last_clear": observed_time - last_clear_time if last_clear_time is not None else None,
        "action_count_after_last_clear": action_count - last_clear_action if last_clear_action is not None else None,
        "repeated_same_position_same_channel_measure_count": repeat_measure,
        "repeated_same_position_same_channel_direction_count": repeat_direction,
        "actions_on_already_cleared_channel_count": after_clear,
        "duplicate_request_id_count": duplicate_ids, "conflicting_duplicate_count": conflicts,
        "rejected_responses": rejected, "client_error_records": len(errors),
        "transport_error_records": len(transport_errors),
        "error_records_on_retry_attempts": len(later_attempt_errors),
        "retry_marker_records": len(retry_records), "timeout_error_records": len(timeout_records),
        "fallback_count_reported": summary.get("fallback_count"),
        "accepted_enter_count": len(enters), "accepted_exit_count": len(exits),
        "exit_reason": exits[0]["response"].get("exit_reason"),
        "termination_basis_in_log": certificate.get("basis"),
        "independent_termination_basis": independent_basis,
        "independent_termination_valid": termination_valid,
        "absent_channel_count": len(missing), "independent_absence_coverage": coverage,
        "response_timestamp_elapsed_s": real_elapsed,
        "local_solver_wall_time_s_reported": summary.get("solver_wall_time_s"),
        "local_session_wall_time_s_reported": summary.get("local_session_wall_time_s"),
        "remaining_real_duration_s_at_enter": available,
        "official_program_runtime_s": None,
        "official_program_runtime_note": "Official final summary runtime was not supplied; response timestamps and client wall time are separate diagnostics.",
        "all_checks_passed": not failures, "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "official_runs/received-20260912-q3-final-batch10")
    parser.add_argument("--output", type=Path, default=GROUP / "results/official_practice_20260912_batch10_independent_audit.json")
    args = parser.parse_args()
    manifest = read_json(args.input_dir / "input_manifest.json")
    for member in manifest["members"]:
        file = args.input_dir / member["name"]
        if file.stat().st_size != member["bytes"] or sha256(file) != member["sha256"]:
            raise ValueError(f"Input member integrity failure: {file.name}")
    freeze_path = MODEL / "00_主方案_lean/results/main_solution_freeze.json"
    frozen_all = read_json(freeze_path)["file_sha256"]
    frozen = {name: frozen_all[name] for name in RUNTIME_FILES}
    current = {name: sha256(MODEL / name) for name in RUNTIME_FILES}
    if current != frozen:
        raise ValueError("Current runtime source differs from the original frozen source")
    if manifest.get("user_reported_all_omnidirectional") is not True:
        raise ValueError("Q3 all-omnidirectional user confirmation is required")
    cells = independent_cells()
    cases = [audit_case(args.input_dir / f"q3-practice-final-{case}.jsonl", count, current, frozen, cells)
             for case, count in sorted(manifest["user_reported_totals"].items())]
    source_sum = sum(case["successful_unique_clears"] for case in cases)
    seconds_sum = sum(case["virtual_total_s"] for case in cases)
    output = {
        "kind": "independent_offline_q3_official_practice_audit",
        "batch": "20260912_batch10", "official_simulator_contacted": False,
        "formal_test_results": False, "solver_or_parameters_changed": False,
        "audit_script_sha256": sha256(Path(__file__)),
        "source_freeze_path": freeze_path.relative_to(ROOT).as_posix(),
        "source_freeze_sha256": sha256(freeze_path),
        "source_sha256": frozen,
        "input_manifest_sha256": sha256(args.input_dir / "input_manifest.json"),
        "input_manifest": manifest,
        "evidence_limits": [
            "The five source totals are the user's transcription of official practice end screens, not fields returned by the API.",
            "Accepted success responses establish clear outcomes; hidden source coordinates are unavailable for an independent 20 m distance check.",
            "Absence is recomputed from actual accepted no_signal points, on every closed 20 m square intersecting the target disk. No model imports or solver reruns are used.",
            "At 16 distinct successful channels the public source-count upper bound establishes completion; this is a valid problem assumption, not an unproved geometric certificate.",
            "The official final program-runtime field remains null; elapsed response timestamps and client wall-clock measurements are separately labeled.",
            "Movement-to-clear subcomponents overlap movement_s and must not be added twice. Actions include both measures and clear attempts; enter and exit are excluded.",
            "Raw logs contain private identifiers and remain local. This output retains only de-identified aggregates and integrity hashes.",
        ],
        "cases": cases,
        "aggregate": {
            "case_count": len(cases), "user_reported_source_sum": sum(manifest["user_reported_totals"].values()),
            "successful_unique_clear_sum": source_sum,
            "clear_fraction": source_sum / sum(manifest["user_reported_totals"].values()),
            "virtual_total_s_sum": seconds_sum, "mean_virtual_total_s": seconds_sum / len(cases),
            "pooled_virtual_s_per_cleared": seconds_sum / source_sum,
            "unweighted_mean_case_virtual_s_per_cleared": statistics.mean(case["virtual_s_per_cleared"] for case in cases),
            "sample_sd_case_virtual_total_s": statistics.stdev(case["virtual_total_s"] for case in cases),
            "all_checks_passed": all(case["all_checks_passed"] for case in cases),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(output["aggregate"], ensure_ascii=False))
    if not output["aggregate"]["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
