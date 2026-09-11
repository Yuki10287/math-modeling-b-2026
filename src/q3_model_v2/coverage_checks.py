"""Standalone correctness checks for continuous negative-coverage certificates.

These are local geometric checks and do not call any simulator or network.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from coverage_model import CoverageTracker


def run_checks() -> dict:
    checks = {}
    coverage = CoverageTracker()
    assert coverage.total_cells > math.pi * 90 ** 2
    assert not coverage.complete(1)
    assert coverage.uncovered_count(1) == coverage.total_cells
    checks["full_disk_cells_include_boundary_overhang"] = True

    # Every real domain point belongs to a retained cell, including the rim.
    rng = np.random.default_rng(260911)
    theta = np.linspace(0, 2 * np.pi, 10001)
    interior_theta = rng.uniform(0, 2 * np.pi, 20000)
    radii = 1800 * np.sqrt(rng.uniform(0, 1, 20000))
    points = np.vstack((np.column_stack((1800 * np.cos(theta), 1800 * np.sin(theta))),
                        np.column_stack((radii * np.cos(interior_theta), radii * np.sin(interior_theta)))))
    cell_keys = {tuple(np.rint(c / 20 * 2).astype(int)) for c in coverage.centers}
    centers = (np.floor(np.clip(points, -1800, 1800 - 1e-9) / 20) + .5) * 20
    assert all(tuple(np.rint(c / 20 * 2).astype(int)) in cell_keys for c in centers)
    checks["boundary_and_interior_points_in_retained_cells"] = len(points)

    initial_gain = coverage.channel_gain([0, 0], 1)
    assert initial_gain > 0
    assert coverage.gain([0, 0], [1, 2]) == 2 * initial_gain
    assert coverage.observe(1, [0, 0]) == initial_gain
    assert coverage.gain([0, 0], [1, 2]) == initial_gain
    assert coverage.observe(1, [0, 0]) == 0
    assert coverage.uncovered_count(2) == coverage.total_cells
    assert len(coverage.certificate(1)["negative_points"]) == 1
    checks["gain_and_channel_isolation_and_deduplication"] = True

    stations = np.vstack(([0, 0], [[1200 * math.cos(k * math.pi / 3),
                                   1200 * math.sin(k * math.pi / 3)] for k in range(6)]))
    for q in stations:
        coverage.observe(1, q)
    assert coverage.complete(1)
    assert coverage.uncovered_count(1) == 0
    assert not coverage.complete(2)
    checks["original_seven_stations_complete"] = True

    # Verify each excluded cell has an entire-square witness. Convexity of a
    # disk means all four corners inside also establishes square containment.
    negative = np.array(coverage.certificate(1)["negative_points"])
    offsets = np.array([[-10, -10], [-10, 10], [10, -10], [10, 10]])
    corners = coverage.centers[:, None, :] + offsets
    witnesses = np.zeros(coverage.total_cells, dtype=bool)
    for q in negative:
        witnesses |= np.max(np.linalg.norm(corners - q, axis=2), axis=1) <= 1000 + 1e-8
    assert np.all(witnesses)
    checks["every_certified_square_has_continuous_witness"] = int(witnesses.sum())

    # A negative observation outside the original seven stations is useful.
    arbitrary = CoverageTracker(channels=[7])
    predicted = arbitrary.channel_gain([850.25, -400.75], 7)
    assert predicted > 0
    assert arbitrary.observe(7, [850.25, -400.75]) == predicted
    checks["arbitrary_feedback_locations_counted"] = True

    # Never turn a union of sampled-point inclusions into a continuous claim.
    # A disk matching the target disk does geometrically cover it, but our
    # full-square overhang leaves boundary cells unproved: conservative false
    # negatives are allowed. Likewise a shifted disk must not certify.
    exact = CoverageTracker(channels=[1], target_r=1000)
    exact.observe(1, [0, 0])
    assert not exact.complete(1)
    shifted = CoverageTracker(channels=[1], target_r=1000)
    shifted.observe(1, [1, 0])
    assert not shifted.complete(1)
    assert shifted.uncovered_count(1) > 0
    checks["boundary_overhang_and_shifted_circle_not_falsely_certified"] = True

    # Leaving out the far-east station must leave a real uncovered target.
    missing = CoverageTracker(channels=[1])
    for q in stations[[0, 2, 3, 4, 5, 6]]:
        missing.observe(1, q)
    assert not missing.complete(1)
    assert min(np.linalg.norm(np.array([1800., 0.]) - q) for q in stations[[0, 2, 3, 4, 5, 6]]) > 1000
    checks["real_gap_is_not_certified"] = True

    # Configuration too coarse for a single-circle witness yields zero gain.
    coarse = CoverageTracker(channels=[1], target_r=100, signal_r=1, cell_size=20)
    assert coarse.observe(1, [0, 0]) == 0
    assert not coarse.complete(1)
    checks["no_full_cell_witness_when_signal_radius_too_small"] = True

    for invalid in ([float("nan"), 0], [0], [float("inf"), 0]):
        try:
            coverage.observe(1, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid point accepted")
    try:
        coverage.complete(999)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown channel accepted")
    checks["invalid_feedback_rejected"] = True

    return dict(passed=True, checks=checks, total_cells=coverage.total_cells,
                proof="No source truth is used. Every disk-intersecting square is excluded only with an entire-square no-signal witness.")


if __name__ == "__main__":
    result = run_checks()
    output = Path(__file__).with_name("coverage-check-results.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
