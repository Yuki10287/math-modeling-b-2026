"""Small, reproducible Q3 assumption probes; never contact any simulator.

These probes supplement the mathematical argument. They are neither a proof
over all floating-point inputs nor an additional performance/official sample.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

MODEL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL))
import geometry as core
from belief_model import FeedbackModel, exclude_disk_hull
from coverage_model import CoverageTracker
from recovery import optical_grid

SEED = 912003


def deny_network(event, args):
    if event.startswith("socket."):
        raise RuntimeError("Network is forbidden in this mathematical audit")


def outside_distance(P, point):
    """Positive normalized edge violation for a counterclockwise convex hull."""
    if not len(P):
        return float("inf")
    if len(P) == 1:
        return float(np.linalg.norm(P[0] - point))
    if len(P) == 2:
        edge = P[1] - P[0]
        t = np.clip(np.dot(point - P[0], edge) / max(np.dot(edge, edge), 1e-30), 0, 1)
        return float(np.linalg.norm(point - P[0] - t * edge))
    edges = np.roll(P, -1, axis=0) - P
    offsets = point - P
    signed = (edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0])
    signed /= np.maximum(np.linalg.norm(edges, axis=1), 1e-15)
    return max(0., float(-signed.min()))


class PublicAPI:
    """Truth belongs to this test fixture, never to the tested FeedbackModel."""
    def __init__(self, truth, radius):
        self.truth, self.radius = truth, radius
        self.position, self.channel, self.index = np.zeros(2), 1, 0

    def measure(self, q, channel):
        self.position, self.channel = np.array(q), channel
        self.index += 1
        distance = float(np.linalg.norm(self.truth - q))
        if distance > self.radius:
            return {"measure_result": "no_signal", "accepted": True}
        if distance <= 5:
            return {"measure_result": "near", "accepted": True}
        angle = math.degrees(math.atan2(*(self.truth - q)[::-1])) % 360
        error = (-1., 1., 0.)[self.index % 3]
        # Explicit nearest-rounding interpretation; not an official generator.
        return {"measure_result": "direction", "accepted": True,
                "svd_deg": round((angle + error) % 360, 2) % 360}

    def clear(self, q, channel):
        self.position = np.array(q)
        success = np.linalg.norm(self.truth - q) <= 20
        return {"clear_result": "success" if success else "no_target_in_range",
                "accepted": True}


def audit():
    rng = np.random.default_rng(SEED)
    max_violation, conflicts, updates = 0., 0, 0
    for case in range(200):
        theta = rng.uniform(0, 2 * math.pi)
        radial = 1800. if case % 5 == 0 else 1800. * math.sqrt(rng.random())
        truth = radial * np.array([math.cos(theta), math.sin(theta)])
        radius = (1000., 1500.)[case % 2]
        model = FeedbackModel(PublicAPI(truth, radius), [], range_cuts=True)
        for distance in (radius + 1e-4, radius - 1e-4, .7 * radius,
                         radius + 250., 6., radius + 1.):
            angle = rng.uniform(0, 2 * math.pi)
            q = truth + distance * np.array([math.cos(angle), math.sin(angle)])
            model.measure(q, 1)
            if 1 in model.beliefs:
                item = model.beliefs[1]
                conflicts += bool(item["conflict"])
                updates += 1
                max_violation = max(max_violation, outside_distance(item["P"], truth))
    assert conflicts == 0 and max_violation <= 1e-6
    tracker = CoverageTracker(channels=[1])
    for q in core.coverage_points():
        tracker.observe(1, q)
    assert tracker.complete(1)
    square = np.array([[-50., -50.], [50., -50.], [50., 50.], [-50., 50.]])
    relaxed = exclude_disk_hull(square, np.zeros(2), 20.)
    refilled = len(relaxed) == 4 and np.allclose(relaxed, square)
    assert refilled

    # An intentionally conditional calculation, NOT an admissible official
    # counterexample: its final returned error exceeds the literal +/-1 bound.
    true_angle = 45.0099
    reply = math.floor((true_angle - 1) * 100) / 100
    truth = 1500 * np.array([math.cos(math.radians(true_angle)),
                             math.sin(math.radians(true_angle))])
    polygon = core.initial_belief(np.zeros(2), reply)
    path = optical_grid(np.zeros(2), reply)
    source_files = ("geometry.py", "belief_model.py", "coverage_model.py",
                    "recovery.py", "solver.py", "local_policy.py", "scan_planning.py")
    return {
        "schema_version": 1,
        "audit": "independent small mathematical probes, no simulator connection",
        "seed": SEED,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "source_sha256": {name: hashlib.sha256((MODEL / name).read_bytes()).hexdigest()
                          for name in source_files},
        "scope": "Finite boundary/random probes; not a universal numerical proof or performance sample",
        "observation_generation": "Nearest rounding to two decimals after errors from {-1,0,+1} degrees",
        "generated_truth_cases": 200,
        "belief_updates_checked": updates,
        "max_truth_outside_distance_m": max_violation,
        "conflict_count": conflicts,
        "fixed_seven_stations_complete": tracker.complete(1),
        "coverage_cells": tracker.total_cells,
        "convex_hull_negative_hole_is_refilled": bool(refilled),
        "hole_probe_interpretation": "Conservative information loss, not exclusion of a feasible source",
        "conditional_pre_quantization_truncation_only": {
            "true_angle_degrees": true_angle,
            "noise_degrees": -1.,
            "returned_angle_degrees": reply,
            "total_error_degrees": true_angle - reply,
            "truth_outside_belief_m": outside_distance(polygon, truth),
            "fallback_nearest_distance_m": float(np.min(np.linalg.norm(path - truth, axis=1))),
            "admissible_under_literal_returned_error_bound": False,
            "interpretation": "Only relevant if an additional official rule permits pre-quantization noise plus truncation and final errors over one degree; not an established official model counterexample",
        },
    }


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    sys.addaudithook(deny_network)
    result = audit()
    destination = Path(__file__).resolve().parents[1] / "results" / "model_assumptions_audit_v1.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
