"""Offline geometry of the actually travelled pre-last-clear polyline.

Only de-identified completed-log analysis is read. This script neither calls a
policy nor reconstructs hidden sources / unobserved feedback. A continuum of
possible stopping points is an optimistic capacity relaxation, not evidence
that any channel was measured there.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


TARGET_R = 1800.0
SIGNAL_R = 1000.0
CELL = 20.0
SPEED = 5.0
HALF_DIAGONAL = CELL / np.sqrt(2.0)
STUDY = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = STUDY.parent / "05_官方演练复盘/results/official_practice_20260911_analysis.json"


def cells():
    """All closed squares intersecting the target disk, including overhang."""
    n = int(np.ceil(TARGET_R / CELL))
    axis = (np.arange(-n, n) + 0.5) * CELL
    x, y = np.meshgrid(axis, axis)
    centers = np.column_stack((x.ravel(), y.ravel()))
    nearest = np.maximum(np.abs(centers) - CELL / 2, 0)
    return centers[np.sum(nearest ** 2, axis=1) <= (TARGET_R + 1e-9) ** 2]


def segment_distance(points, a, b):
    """Analytic point-to-closed-segment distance, including zero-length legs."""
    points, a, b = np.asarray(points, float), np.asarray(a, float), np.asarray(b, float)
    delta = b - a
    length2 = float(delta @ delta)
    if length2 < 1e-20:
        return np.linalg.norm(points - a, axis=1)
    fraction = np.clip((points - a) @ delta / length2, 0, 1)
    closest = a + fraction[:, None] * delta
    return np.linalg.norm(points - closest, axis=1)


def polyline_distance(points, path):
    path = np.asarray(path, float)
    if len(path) == 0:
        raise ValueError("A nonempty path is required")
    result = np.linalg.norm(points - path[0], axis=1)
    for a, b in zip(path[:-1], path[1:]):
        result = np.minimum(result, segment_distance(points, a, b))
    return result


def self_checks():
    """Independent analytic cases, especially interior / endpoint distinctions."""
    np.testing.assert_allclose(segment_distance([[1, 3], [-1, 0], [4, 0]], [0, 0], [2, 0]), [3, 1, 2])
    np.testing.assert_allclose(segment_distance([[3, 4]], [0, 0], [0, 0]), [5])
    path = [[0, 0], [10, 0], [10, 10]]
    query = np.array([[5, 1], [11, 5], [-2, -2]], float)
    np.testing.assert_allclose(polyline_distance(query, path), [1, 1, np.sqrt(8)])
    np.testing.assert_allclose(polyline_distance(query, path), polyline_distance(query, path[::-1]))
    np.testing.assert_allclose(polyline_distance(query, path), polyline_distance(query, [[0, 0], [5, 0], [10, 0], [10, 5], [10, 10]]))
    assert float(polyline_distance(np.array([[5.0, 0.0]]), path)[0]) < min(np.linalg.norm(np.array([5.0, 0.0]) - p) for p in path)
    return {"passed": True, "analytic_checks": 6}


def audit_case(case, centers):
    prior = [a for a in case["actions"] if a["time_s"] <= case["last_clear_s"] + 1e-7]
    if not prior or prior[-1].get("clear_result") != "success":
        raise ValueError("Last-clear checkpoint is not an accepted successful clear")
    path = np.array([[0.0, 0.0]] + [a["position"] for a in prior])
    stop_distance = np.full(len(centers), np.inf)
    for q in path:
        stop_distance = np.minimum(stop_distance, np.linalg.norm(centers - q, axis=1))
    distance = polyline_distance(centers, path)
    assert np.all(distance <= stop_distance + 1e-8)
    whole_stop = stop_distance + HALF_DIAGONAL <= SIGNAL_R - 1e-9
    whole_path = distance + HALF_DIAGONAL <= SIGNAL_R - 1e-9
    inside = np.linalg.norm(centers, axis=1) <= TARGET_R
    holes = inside & (distance > SIGNAL_R + 1e-8)
    maximum = int(np.argmax(np.where(inside, distance, -1.0)))
    physical_proof = bool(holes.any())
    witness = dict(position=centers[maximum].tolist(),
                   norm_from_origin_m=float(np.linalg.norm(centers[maximum])),
                   nearest_pre_last_clear_path_distance_m=float(distance[maximum]),
                   excess_over_minimum_reception_radius_m=float(max(0.0, distance[maximum] - SIGNAL_R)))
    tail = [a for a in case["actions"] if a["time_s"] > case["last_clear_s"] + 1e-7]
    bound = None
    if physical_proof:
        # Every such point is undetectable anywhere on the preserved prefix.
        # To exclude even one of them after that prefix, a route from its final
        # position must enter a radius-1000 disk about that point. The reverse
        # triangle inequality gives a necessary length, not an attainable tour.
        final_distances = np.linalg.norm(centers - path[-1], axis=1)
        j = int(np.argmax(np.where(holes, final_distances, -1.0)))
        necessary_m = max(0.0, float(final_distances[j]) - SIGNAL_R)
        actual_m = sum(float(a["movement_m"]) for a in tail)
        assert necessary_m <= actual_m + 1e-4
        bound = dict(witness_position=centers[j].tolist(),
                     witness_nearest_prefix_distance_m=float(distance[j]),
                     checkpoint_position=path[-1].tolist(),
                     necessary_tail_movement_at_least_m=necessary_m,
                     necessary_tail_movement_at_least_s=necessary_m / SPEED,
                     actual_tail_movement_s=actual_m / SPEED,
                     tail_movement_saving_at_most_s=max(0.0, (actual_m - necessary_m) / SPEED),
                     scope="Preserve the full original prefix through the last clear; allow arbitrary extra on-prefix measurements; then change only the suffix. This is a movement-only optimistic bound and excludes added measurement costs. Changing the earlier route or clear order falls outside it.")
    finite = None
    if whole_path.all():
        # Both actual zero-tail records already have complete absence evidence.
        # They need no hypothetical inserted stop to finish by this checkpoint.
        if case["scan_tail_s"] <= 1e-7 and case["passed"]:
            finite = dict(additional_stops_needed_for_actual_completion=0,
                          additional_measurement_fee_s=0.0,
                          basis="The recorded run already completed at this checkpoint; no new feedback is assumed.")
        else:
            finite = dict(status="continuum_capacity_only_no_finite_measurement_plan_constructed")
    return dict(case=case["case"], cleared=case["cleared"], absent=case["absent"],
                actual_total_s=case["total_s"], actual_tail_s=case["scan_tail_s"],
                path_distinct_stop_count=len(np.unique(path, axis=0)),
                path_positive_length_segments=int(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-8)),
                square_cell_count=len(centers),
                stopped_point_whole_cell_capacity=int(whole_stop.sum()),
                continuous_path_whole_cell_capacity=int(whole_path.sum()),
                additional_cells_reachable_between_stops=int(np.sum(whole_path & ~whole_stop)),
                continuous_path_whole_cell_cover_complete=bool(whole_path.all()),
                interior_sample_points_proving_physical_holes=int(holes.sum()),
                physical_hole_proved=physical_proof,
                physical_hole_witness=witness if physical_proof else None,
                farthest_examined_in_domain_point=witness,
                finite_insertion_completion=finite,
                retained_prefix_tail_lower_bound=bound)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve historical diagnostics; use a new output filename")
    raw = args.input.read_bytes()
    original = json.loads(raw.decode("utf-8-sig"))
    checks = self_checks()
    results = [audit_case(case, cells()) for case in original["cases"]]
    report = dict(analysis_kind="offline_continuous_prefix_corridor_capacity",
                  official_simulator_contacted=False, policy_executed=False,
                  hidden_truth_used=False, unobserved_feedback_generated=False,
                  source_input=str(args.input.resolve()), input_sha256=hashlib.sha256(raw).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  constants=dict(target_radius_m=TARGET_R, minimum_signal_radius_m=SIGNAL_R,
                                 cell_size_m=CELL, cell_half_diagonal_m=HALF_DIAGONAL, speed_m_per_s=SPEED),
                  geometry_self_checks=checks,
                  interpretation=[
                      "A hole witness is strictly in the target disk and more than 1000 m from every point of the recorded prefix. It proves that unlimited on-path readings alone cannot certify its absence for minimum-range sources.",
                      "For each whole-cell capacity claim, at least one point on the path is within 1000 minus the cell half diagonal of its center. That single disk contains the entire square. This is prospective capacity, never accepted evidence.",
                      "A sampled count is not exact uncovered area. No sampled hole is not by itself a proof of continuous coverage; the separate whole-cell test supplies a sufficient coverage certificate.",
                      "No alternative official score is calculated. Changed policies may change discoveries, routes, source order, and channel costs.",
                  ], cases=results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in results:
        print(json.dumps({key: row[key] for key in ("case", "physical_hole_proved", "continuous_path_whole_cell_capacity", "additional_cells_reachable_between_stops", "interior_sample_points_proving_physical_holes", "retained_prefix_tail_lower_bound")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
