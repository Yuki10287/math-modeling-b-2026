"""Audit completed holdout artifacts by replaying saved public actions only.

This script never calls solve_multi, regenerates a test population, tunes a
parameter, contacts a network, or starts an official simulator. LocalArena
receives the already saved sources and executes the already saved actions.
Independent auditors then verify the saved trace and completion certificate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

from environment import LocalArena
from validate_model import (
    certificate_audit,
    independent_cells,
    independent_negative_cover,
    independent_time_audit,
    observation_audit,
    source_hashes,
    trace_truth_audit,
)


BASE = Path(__file__).resolve().parent
FIELDS = ("smooth", "hash", "extreme")
SEEDS = range(4000, 4020)


def implementation_hashes():
    hashes = source_hashes()
    hashes[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashes


def all_channel_evidence_audit(arena, outcome, centers):
    """Check every channel against its own pre-removal public observations."""
    negatives = {c: set() for c in range(1, 21)}
    removed = set()
    positives = set()
    for event in arena.events:
        channel = event["channel"]
        if event["action"] == "clear":
            if event["clear_result"] == "success":
                removed.add(channel)
        elif channel not in removed:
            if event["measure_result"] == "no_signal":
                negatives[channel].add(tuple(event["position"]))
            else:
                positives.add(channel)
    certificate = outcome["completion_certificate"]
    declared = certificate["channel_coverage"]
    assert set(declared) == {str(c) for c in range(1, 21)}
    rows = {}
    for channel in range(1, 21):
        item = declared[str(channel)]
        assert item["channel"] == channel
        assert item["basis"] == "whole_cell_negative_disk_cover"
        assert item["target_radius_m"] == 1800.0
        assert item["minimum_reception_radius_m"] == 1000.0
        assert item["cell_size_m"] == 20.0
        assert item["total_cells"] == len(centers)
        claimed_points = {tuple(q) for q in item["negative_points"]}
        assert claimed_points == negatives[channel], (channel, "negative evidence not equal to public history")
        assert len(claimed_points) == len(item["negative_points"]), (channel, "duplicate certificate evidence")
        covered = independent_negative_cover(sorted(negatives[channel]), centers)
        # The production tracker uses the square circumradius, which can be
        # more conservative than this independent exact four-corner test.
        assert 0 <= item["excluded_cells"] <= int(covered.sum())
        assert item["uncovered_cells"] == len(centers) - item["excluded_cells"]
        assert item["complete"] == (item["excluded_cells"] == len(centers))
        if item["complete"]:
            assert covered.all()
        rows[str(channel)] = dict(
            positive_observed=channel in positives,
            success_observed=channel in removed,
            negative_positions=len(negatives[channel]),
            certified_excluded_cells=item["excluded_cells"],
            independent_whole_square_excluded_cells=int(covered.sum()),
            complete=item["complete"],
        )
    assert set(certificate["cleared_channels"]) == removed
    absent_by_count = set(certificate.get("absent_by_count_channels", []))
    if absent_by_count:
        assert len(removed) == 16
        assert absent_by_count == set(range(1, 21)) - removed
    return dict(channels_checked=20, positive_channels=sorted(positives),
                all_negative_points_match_pre_removal_actions=True, per_channel=rows)


def audit_case(path, seed, field, centers):
    raw = path.read_bytes()
    saved = json.loads(raw)
    metric, outcome = saved["result"], saved["outcome"]
    assert metric["seed"] == seed and metric["field"] == field
    assert metric["schedule"] == "lean"
    assert outcome["model_configuration"]["schedule"] == "lean"
    arena = LocalArena(saved["sources"], seed=seed, field=field)
    for index, event in enumerate(saved["events"], 1):
        action = event["action"]
        assert action in ("measure", "clear"), (index, action)
        getattr(arena, action)(event["position"], event["channel"])
        actual = arena.events[-1]
        assert actual == event, dict(event_index=index, saved=event, replayed=actual)
    assert arena.events == saved["events"], "Full replay event list differs"
    assert len(arena._sources) == metric["source_count"]
    assert len(arena._removed) == metric["cleared"] == outcome["cleared"]
    assert len(arena._removed) == len(arena._sources)
    assert metric["all_cleared"] is True and outcome["complete"] is True
    assert metric["certificate_valid"] is True
    assert arena.time_s == metric["total_s"]
    assert arena.distance_m == metric["distance_m"]

    timing = independent_time_audit(arena)
    assert timing["counts"] == metric["counts"]
    assert abs(timing["average_s_per_cleared"] - metric["average_s"]) < 1e-8
    observations = observation_audit(arena)
    certificate = certificate_audit(arena, outcome, observations, centers)
    channel_evidence = all_channel_evidence_audit(arena, outcome, centers)
    geometry = trace_truth_audit(arena, saved["trace"])
    fallback_count = sum(row.get("phase") == "fallback_start" for row in saved["trace"])
    assert fallback_count == metric["fallback_count"] == outcome["fallback_count"]
    return dict(
        file=str(path.relative_to(BASE)).replace("\\", "/"),
        input_sha256=hashlib.sha256(raw).hexdigest(),
        seed=seed, field=field, schedule="lean", passed=True,
        source_count=len(arena._sources), cleared=len(arena._removed),
        actions_replayed=len(saved["events"]), event_values_exactly_equal=True,
        independent_timing=timing, independent_certificate=certificate,
        channel_evidence=channel_evidence, geometry=geometry,
        max_observed_absolute_error_deg=observations["max_observed_absolute_error_deg"],
        repeated_place_checks=observations["repeated_place_checks"],
        trace_rows=len(saved["trace"]), fallback_count=fallback_count,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "results" / "holdout_replay_validation.json")
    args = parser.parse_args()
    started = time.perf_counter()
    initial_hashes = implementation_hashes()
    centers = independent_cells()
    report = dict(
        audit_kind="replay_of_completed_holdout_artifacts", local_only=True,
        strategy_rerun=False, source_population_regenerated=False,
        network_used=False, official_simulator_used=False,
        seeds=[4000, 4019], fields=list(FIELDS), requested_cases=60,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256_before=initial_hashes, cases=[],
    )
    for field in FIELDS:
        for seed in SEEDS:
            path = BASE / "results" / f"holdout-{field}" / "cases" / f"{seed}-{field}-lean.json"
            try:
                row = audit_case(path, seed, field, centers)
            except Exception as exc:
                row = dict(file=str(path.relative_to(BASE)).replace("\\", "/"), seed=seed,
                           field=field, passed=False, error_type=type(exc).__name__,
                           error=str(exc), traceback=traceback.format_exc())
            report["cases"].append(row)
            if not row["passed"] or len(report["cases"]) % 10 == 0:
                print(json.dumps(dict(completed=len(report["cases"]), requested=60,
                                      latest=f"{seed}-{field}", passed=row["passed"],
                                      error=row.get("error")), ensure_ascii=False), flush=True)
    final_hashes = implementation_hashes()
    good = [row for row in report["cases"] if row["passed"]]
    report["source_sha256_after"] = final_hashes
    report["source_snapshot_stable"] = initial_hashes == final_hashes
    report["passed"] = len(good) == 60 and report["source_snapshot_stable"]
    report["summary"] = dict(
        passed_cases=len(good), total_cases=len(report["cases"]),
        total_sources=sum(row["source_count"] for row in good),
        total_cleared=sum(row["cleared"] for row in good),
        exact_event_replays=sum(row["actions_replayed"] for row in good),
        polygons_checked=sum(row["geometry"]["polygons_checked"] for row in good),
        full_optical_paths_checked=sum(row["geometry"]["full_optical_paths_checked"] for row in good),
        channel_certificates_checked=sum(row["channel_evidence"]["channels_checked"] for row in good),
        clear_failures_included_in_timing=sum(row["independent_timing"]["counts"]["clear_fail"] for row in good),
        fallback_count=sum(row["fallback_count"] for row in good),
        max_independent_step_time_error_s=max((row["independent_timing"]["max_step_error_s"] for row in good), default=0.),
        audit_wall_s=time.perf_counter() - started,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(passed=report["passed"], source_snapshot_stable=report["source_snapshot_stable"],
                          **report["summary"]), ensure_ascii=False), flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
