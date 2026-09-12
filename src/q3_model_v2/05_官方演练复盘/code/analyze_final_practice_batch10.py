"""Offline diagnostics for the five Q3 practices in 测试结果(10).zip.

Reads saved feedback and belief traces only. Does not generate hidden truth,
contact a simulator, rerun the policy, or tune model parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys


def deny_network(event, args):
    if event in ("socket.connect", "socket.sendto", "socket.getaddrinfo"):
        raise RuntimeError("This analysis only reads completed local logs")


sys.addaudithook(deny_network)
HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
MODEL = STUDY.parent
ROOT = MODEL.parents[1]
VALIDATION = MODEL / "00_主方案_lean" / "code"
sys.path[:0] = [str(MODEL), str(VALIDATION)]

from analyze_official_results import read_case, visited_stop_diagnostic
from validate_model import independent_cells


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(cases):
    total = sum(c["total_s"] for c in cases)
    cleared = sum(c["cleared"] for c in cases)
    costs = {k: sum(c["costs_s"][k] for c in cases) for k in cases[0]["costs_s"]}
    return dict(
        cases=len(cases), cleared=cleared,
        user_reported_total_sources=sum(c["user_reported_source_count"] for c in cases),
        clearance_ratio=cleared / sum(c["user_reported_source_count"] for c in cases),
        total_s=total, mean_total_s=total / len(cases),
        pooled_average_s=total / cleared,
        mean_case_average_s=statistics.mean(c["average_s"] for c in cases),
        min_total_s=min(c["total_s"] for c in cases), max_total_s=max(c["total_s"] for c in cases),
        sum_costs_s=costs, mean_costs_s={k: v / len(cases) for k, v in costs.items()},
        cost_fraction={k: v / total for k, v in costs.items()},
        mean_scan_tail_s=statistics.mean(c["scan_tail_s"] for c in cases),
        scan_tail_fraction=sum(c["scan_tail_s"] for c in cases) / total,
        accepted_actions=sum(c["accepted_actions"] for c in cases),
        max_timing_error_s=max(c["max_cumulative_timing_error_s"] for c in cases),
        runtime_source_checks=sum(c["source_files_checked"] for c in cases),
        max_local_session_wall_s=max(c["local_session_wall_s"] for c in cases),
        min_local_session_wall_s=min(c["local_session_wall_s"] for c in cases),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output file to preserve prior evidence")
    inputs = json.loads((args.input_dir / "input_manifest.json").read_text(encoding="utf-8-sig"))
    expected = {f"q3-practice-final-{i:02}.jsonl": n for i, n in enumerate([16, 10, 11, 16, 10], 1)}
    paths = sorted(args.input_dir.glob("*.jsonl"))
    assert {p.name for p in paths} == set(expected)
    frozen = json.loads((MODEL / "00_主方案_lean/results/main_solution_freeze.json").read_text(encoding="utf-8"))
    runtime = ("official_client.py", "geometry.py", "solver.py", "belief_model.py", "local_policy.py",
               "coverage_model.py", "scan_planning.py", "recovery.py", "v1_solver.py", "baseline_solver.py")
    current = {name: digest(MODEL / name) for name in runtime}
    assert current == {name: frozen["file_sha256"][name] for name in runtime}
    members = {item["name"]: item["sha256"] for item in inputs["members"]}
    centers = independent_cells()
    cases = []
    for path in paths:
        trace_path = Path(str(path) + ".beliefs.json")
        assert digest(path) == members[path.name] and digest(trace_path) == members[trace_path.name]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        summary = next(row for row in rows if row.get("type") == "session_summary")
        assert summary["source_sha256"] == current
        assert (summary["policy"], summary["schedule"]) == ("time", "lean")
        case = read_case(path, centers, source_dir=MODEL)
        case.update(user_reported_source_count=expected[path.name],
                    user_reported_omnidirectional_count=expected[path.name],
                    user_reported_directional_count=0,
                    source_count_evidence="User transcribed simulator end-screen counts in this batch request",
                    trace_sha256=digest(trace_path),
                    completion_basis=summary["completion_certificate"]["basis"],
                    absent_by_count=len(summary["completion_certificate"].get("absent_by_count_channels", [])),
                    model_configuration=summary["model_configuration"])
        assert case["cleared"] == case["user_reported_source_count"]
        case["clearance_ratio"] = case["cleared"] / case["user_reported_source_count"]
        case["hindsight_pre_last_clear_visited_stop_cover"] = visited_stop_diagnostic(case, centers)
        cases.append(case)
    old_path = STUDY / "results/official_practice_20260911_analysis.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    assert len(old["cases"]) == 5
    old_time = sum(c["total_s"] for c in old["cases"])
    old_cleared = sum(c["cleared"] for c in old["cases"])
    new = aggregate(cases)
    report = dict(
        analysis_kind="offline_official_practice_feedback_and_geometry_audit",
        batch="Q3 final-practice 01-05, archive 测试结果(10).zip", analysis_date="2026-09-12",
        official_simulator_contacted=False, solver_rerun=False, hidden_truth_available=False,
        new_model_performance_improvement_claim=False,
        scope="Five supplied practice records; no formal-test records or official runtime summaries supplied",
        input_manifest=inputs, frozen_runtime_sha256=current,
        audit_source_sha256={str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in
                            (Path(__file__), HERE / "analyze_official_results.py", VALIDATION / "validate_model.py",
                             VALIDATION / "audit_holdout_replays.py")},
        aggregate=new, cases=cases,
        earlier_batch=dict(path=str(old_path.relative_to(ROOT)).replace('\\', '/'), sha256=digest(old_path),
                           cases=5, cleared=old_cleared, total_s=old_time,
                           mean_total_s=old_time / 5, pooled_average_s=old_time / old_cleared),
        combined_description=dict(cases=10, cleared=old_cleared + new["cleared"],
            mean_total_s=(old_time + new["total_s"]) / 10,
            pooled_average_s=(old_time + new["total_s"]) / (old_cleared + new["cleared"]),
            note="Same frozen solver; different layouts, descriptive aggregation only. Earlier batch lacks user-reported source totals."),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"aggregate": new, "combined": report["combined_description"]}, ensure_ascii=False))
    for c in cases:
        print(json.dumps({k: c[k] for k in ("case", "cleared", "total_s", "average_s", "counts",
            "scan_tail_s", "completion_basis", "geometry_proofs", "costs_s", "local_session_wall_s")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
