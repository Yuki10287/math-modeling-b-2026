"""Offline lower bound from successful Q3 clear positions, without hidden truth.

Run from the repository root. The default input is the private batch-10 log
directory; the public result contains no robot IDs, request IDs or timestamps.
Only the Python standard library is used; no solver is imported or HTTP sent.

For each source i, its true position g_i is within 20 m of the logged successful
position p_i. Any other successful clear position q_i is also within 20 m of
g_i, so ||q_i-p_i|| <= 40 m. Starting at the origin, a successful route with
order pi therefore moves at least
  max(||p_pi[0]|| - 40, 0)
  + sum(max(||p_pi[k]-p_pi[k-1]|| - 80, 0)).
An exact Held-Karp DP minimizes these relaxed edge costs over ALL orders, with
no return to the origin. Dividing by 5 m/s and adding the unavoidable 5 s per
successful clear gives a lower bound, even before other necessary costs.

The graph relaxation can be unattainable: separately shortened edges need not
share realizable clear positions. The computation also omits all information
acquisition and absence certification. It is NOT an executable online policy,
an achievable saving, or an estimate of the true optimum. U-L is an UPPER
bound on possible savings, never evidence that this saving is available.
"""
from __future__ import annotations

import argparse
from array import array
from decimal import Decimal, ROUND_FLOOR, localcontext
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import random
import time


ROOT = Path(__file__).resolve().parents[4]
GROUP = ROOT / "src/q3_model_v2/05_官方演练复盘"
INF = 10**18


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edge_mm(a, b, deduction_m):
    """Conservative integer millimetres; Decimal avoids binary sqrt rounding.

    A 1e-30 m guard dominates the 60-digit Decimal sqrt rounding for all valid
    input coordinates (absolute components <= 2,000,000 m). Flooring then
    loses less than 1 mm per edge, apart from this negligible guard.
    """
    with localcontext() as ctx:
        ctx.prec = 60
        dx, dy = a[0] - b[0], a[1] - b[1]
        relaxed = (dx * dx + dy * dy).sqrt() - Decimal(deduction_m)
        guarded = max(Decimal(0), relaxed - Decimal("1e-30"))
        return int((guarded * 1000).to_integral_value(rounding=ROUND_FLOOR))


def relaxed_weights(points):
    origin = (Decimal(0), Decimal(0))
    starts = [edge_mm(origin, p, 40) for p in points]
    edges = [[edge_mm(a, b, 80) for b in points] for a in points]
    return starts, edges


def held_karp_path(starts, edges):
    """Exact open Hamilton-path optimum for nonnegative INTEGER edge weights."""
    n = len(starts)
    if n == 0:
        return 0, []
    full = (1 << n) - 1
    dp = array("q", [INF]) * ((full + 1) * n)
    for j, value in enumerate(starts):
        dp[(1 << j) * n + j] = value
    for mask in range(1, full + 1):
        ends = mask
        while ends:
            bit = ends & -ends
            ends -= bit
            j = bit.bit_length() - 1
            previous = mask ^ bit
            if not previous:
                continue
            best = INF
            choices = previous
            base = previous * n
            while choices:
                k_bit = choices & -choices
                choices -= k_bit
                k = k_bit.bit_length() - 1
                candidate = dp[base + k] + edges[k][j]
                if candidate < best:
                    best = candidate
            dp[mask * n + j] = best
    last = min(range(n), key=lambda j: dp[full * n + j])
    optimum = dp[full * n + last]
    reverse_order = [last]
    mask = full
    while mask != 1 << last:
        previous = mask ^ (1 << last)
        previous_end = min(
            (k for k in range(n) if previous & (1 << k)),
            key=lambda k: dp[previous * n + k] + edges[k][last],
        )
        assert dp[mask * n + last] == dp[previous * n + previous_end] + edges[previous_end][last]
        mask, last = previous, previous_end
        reverse_order.append(last)
    order = list(reversed(reverse_order))
    reconstructed = starts[order[0]] + sum(edges[a][b] for a, b in zip(order, order[1:]))
    assert reconstructed == optimum and sorted(order) == list(range(n))
    return optimum, order


def validate_dp():
    """Independent enumeration on small complete graphs and geometric cases."""
    rng = random.Random(20260912)
    checked = 0
    for n in range(1, 8):
        for repeat in range(3):
            if repeat == 0:
                starts = [rng.randrange(100) for _ in range(n)]
                edges = [[rng.randrange(100) for _ in range(n)] for _ in range(n)]
            else:
                points = [(Decimal(rng.randrange(-200, 201)), Decimal(rng.randrange(-200, 201)))
                          for _ in range(n)]
                starts, edges = relaxed_weights(points)
            expected = min(starts[order[0]] + sum(edges[a][b] for a, b in zip(order, order[1:]))
                           for order in permutations(range(n)))
            actual, _ = held_karp_path(starts, edges)
            assert actual == expected, (n, repeat, actual, expected)
            checked += 1
    assert held_karp_path([], []) == (0, [])
    assert edge_mm((Decimal(0), Decimal(0)), (Decimal(30), Decimal(0)), 40) == 0
    assert edge_mm((Decimal(0), Decimal(0)), (Decimal(140), Decimal(0)), 40) == 99999
    return {"passed": True, "exhaustive_permutation_cases": checked,
            "largest_enumerated_n": 7, "empty_case_checked": True,
            "integer_edge_reconstruction_checked": True}


def analyze_case(path, expected_count, expected_sha256):
    assert sha256(path) == expected_sha256, "input differs from saved manifest"
    records = [json.loads(line, parse_float=Decimal)
               for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    points, channels, request_ids = [], [], set()
    accepted_actions = []
    for row in records:
        if "path" not in row or row.get("response", {}).get("accepted") is not True:
            continue
        request_id = row["request"]["request_id"]
        if request_id in request_ids:
            raise ValueError("duplicate request ID requires a separate deduplication audit")
        request_ids.add(request_id)
        if row["path"] in ("/measure", "/clear"):
            accepted_actions.append(row)
        if row["path"] == "/clear" and row["response"].get("clear_result") == "success":
            position = row["request"]["position"]
            point = (Decimal(position["x"]), Decimal(position["y"]))
            assert all(x.is_finite() and abs(x) <= 2_000_000 for x in point)
            points.append(point)
            channels.append(row["request"]["channel"])
    assert len(points) == len(set(channels)) == expected_count
    summaries = [r for r in records if r.get("type") == "session_summary"]
    assert len(summaries) == 1 and summaries[0]["complete"] is True
    summary = summaries[0]
    observed_s = Decimal(summary["total_virtual_s"])
    exits = [r for r in records if r.get("path") == "/exit" and r["response"].get("accepted") is True]
    assert len(exits) == 1 and Decimal(exits[0]["response"]["virtual_time_s"]) == observed_s
    starts, edges = relaxed_weights(points)
    started = time.perf_counter()
    distance_mm, order = held_karp_path(starts, edges)
    elapsed = time.perf_counter() - started
    mathematical_lower_s = Decimal(distance_mm) / 5000 + 5 * expected_count
    # The mathematical model charges exact physical distance/5. The interface
    # accumulates microseconds. Reserve a FULL microsecond per possible charged
    # action of a strategy that is faster than U; every such action costs at
    # least 3 s before rounding. Strategies >= U already exceed this bound.
    max_competing_actions = math.ceil(observed_s / (Decimal(3) - Decimal("0.000001")))
    rounding_allowance_s = Decimal(max_competing_actions) / 1_000_000
    conservative_lower_s = mathematical_lower_s - rounding_allowance_s
    assert 0 < conservative_lower_s <= observed_s
    observed_distance_m = Decimal(0)
    previous = (Decimal(0), Decimal(0))
    with localcontext() as ctx:
        ctx.prec = 60
        for row in accepted_actions:
            position = row["request"]["position"]
            point = (Decimal(position["x"]), Decimal(position["y"]))
            observed_distance_m += ((point[0] - previous[0])**2 + (point[1] - previous[1])**2).sqrt()
            previous = point
    observed_movement_s = observed_distance_m / 5
    return {
        "case": path.stem.rsplit("-", 1)[1], "private_log_sha256": expected_sha256,
        "sources_cleared_and_user_reported_total": expected_count,
        "observed_feasible_virtual_time_s_U": float(observed_s),
        "observed_movement_s": float(observed_movement_s),
        "observed_nonmovement_s_including_clock_rounding": float(observed_s - observed_movement_s),
        "relaxed_path_minimum_distance_mm_exact_integer": distance_mm,
        "relaxed_path_minimum_distance_m": distance_mm / 1000,
        "unavoidable_successful_clear_s": 5 * expected_count,
        "physical_model_lower_bound_s": float(mathematical_lower_s),
        "clock_rounding_allowance_s": float(rounding_allowance_s),
        "conservative_time_lower_bound_s_L": float(conservative_lower_s),
        "lower_bound_as_fraction_of_observed": float(conservative_lower_s / observed_s),
        "observed_over_lower_bound_U_div_L": float(observed_s / conservative_lower_s),
        "upper_bound_on_possible_savings_s_U_minus_L": float(observed_s - conservative_lower_s),
        "upper_bound_on_possible_savings_fraction": float(1 - conservative_lower_s / observed_s),
        "achievable_savings_or_improvement_estimate": None,
        "relaxed_optimum_channel_order_not_a_strategy": [channels[j] for j in order],
        "dp_wall_time_s": elapsed,
        "runtime_source_sha256_from_log": summary["source_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=ROOT / "official_runs/received-20260912-q3-final-batch10")
    parser.add_argument("--output", type=Path,
                        default=GROUP / "results/q3_information_lower_bound_v1.json")
    args = parser.parse_args()
    checks = validate_dp()
    manifest_path = args.input_dir / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    members = {m["name"]: m["sha256"] for m in manifest["members"]}
    cases = []
    for case, count in sorted(manifest["user_reported_totals"].items()):
        path = args.input_dir / f"q3-practice-final-{case}.jsonl"
        result = analyze_case(path, count, members[path.name])
        cases.append(result)
        print(f"case {case}: U={result['observed_feasible_virtual_time_s_U']:.6f} s; "
              f"L={result['conservative_time_lower_bound_s_L']:.6f} s; "
              f"U/L={result['observed_over_lower_bound_U_div_L']:.3f}; "
              f"DP={result['dp_wall_time_s']:.3f} s")
    sum_u = sum(c["observed_feasible_virtual_time_s_U"] for c in cases)
    sum_l = sum(c["conservative_time_lower_bound_s_L"] for c in cases)
    output = {
        "kind": "offline_success_position_relaxation_lower_bound",
        "batch": "20260912_q3_practice_batch10",
        "formal_test_results": False, "official_simulator_contacted": False,
        "solver_or_parameters_changed": False,
        "hidden_source_positions_read_or_reconstructed": False,
        "source_script_sha256": sha256(Path(__file__)),
        "input_manifest_sha256": sha256(manifest_path),
        "proof": {
            "successful_clear_radius_m": 20, "alternative_success_point_displacement_bound_m": 40,
            "start_edge_m": "max(norm(p_i) - 40, 0)",
            "between_success_edges_m": "max(norm(p_j - p_i) - 80, 0)",
            "path": "exact minimum over permutations; start at origin; no return required",
            "edge_quantization": "Decimal 60-digit square root, subtract 1e-30 m guard, floor to integer mm",
            "quantization_weakening_at_most_s": "N / 5000 plus negligible square-root guard",
            "physical_time_bound": "minimum_relaxed_distance_m / 5 + 5*N",
            "clock_allowance": "1 microsecond per possible action of a strategy no slower than U; minimum action fee 3 seconds",
            "required_assumptions": [
                "Each source is stationary and occupies a unique channel, as the problem states.",
                "Logged accepted clear success means actual source distance <= 20 m.",
                "The supplied source counts are the full counts; these logs cleared each source once.",
                "Timing uses the stated movement/clear costs; microsecond quantization error <= 1 microsecond per action.",
            ],
        },
        "interpretation": [
            "L <= the true case optimum <= U, where U is this observed successful complete execution.",
            "U-L bounds possible savings from above; it does not prove ANY saving is achievable.",
            "U/L is an upper bound on the observed/optimum ratio, NOT a measured optimality gap.",
            "The relaxation omits detection, channel switching, failure, and proving absence.",
            "Its edge minima can be mutually geometrically inconsistent; the DP order is not a realizable policy.",
            "Past success positions are used only for retrospective diagnosis and never supplied to the online solver.",
            "Results cannot establish official ranking or comparison with another team's unrelated cases.",
        ],
        "self_check": checks, "cases": cases,
        "aggregate_descriptive_only": {
            "case_count": len(cases), "source_count": sum(c["sources_cleared_and_user_reported_total"] for c in cases),
            "sum_observed_time_s": sum_u, "sum_lower_bound_s": sum_l,
            "mean_observed_time_s": sum_u / len(cases), "mean_lower_bound_s": sum_l / len(cases),
            "sum_observed_over_sum_lower_bound": sum_u / sum_l,
            "upper_bound_on_total_possible_savings_s": sum_u - sum_l,
            "upper_bound_on_total_possible_savings_fraction": 1 - sum_l / sum_u,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved {args.output.relative_to(ROOT) if args.output.is_relative_to(ROOT) else args.output.name}")


if __name__ == "__main__":
    main()
