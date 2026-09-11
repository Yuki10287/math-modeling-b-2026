"""Supplemental offline audit; preserve the original frozen audit and all results.

Use general polygon membership, not convex halfplanes or a convex hull. This
handles roundoff-sized concavities without accepting points in real concave gaps.
Matplotlib's independent path implementation cross-checks every polygon.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from matplotlib.path import Path as PlotPath

from benchmark_forecast import hashes
from validate_model import (BoundedArena, _contains_convex, certificate_audit,
    independent_cells, independent_disk_cover, independent_time_audit, observation_audit)

ROOT = Path(__file__).resolve().parent
TOLERANCE_M = 1e-5  # The original audit tolerance; not enlarged for this case.


def boundary_distance(polygon, point):
    P, g = np.asarray(polygon, float), np.asarray(point, float)
    edges = np.roll(P, -1, axis=0) - P
    length2 = np.sum(edges * edges, axis=1)
    t = np.clip(np.sum((g - P) * edges, axis=1) / np.maximum(length2, 1e-30), 0, 1)
    return float(np.linalg.norm(g - P - t[:, None] * edges, axis=1).min())


def contains_polygon(polygon, point, tolerance=TOLERANCE_M):
    P, g = np.asarray(polygon, float), np.asarray(point, float)
    assert P.ndim == 2 and P.shape[1] == 2 and len(P) and np.isfinite(P).all()
    if boundary_distance(P, g) <= tolerance:
        return True
    if len(P) < 3:
        return False
    inside = False
    for a, b in zip(P, np.roll(P, -1, axis=0)):
        if (a[1] > g[1]) != (b[1] > g[1]):
            x_crossing = a[0] + (g[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x_crossing > g[0]:
                inside = not inside
    return inside


def regression_checks():
    square = [[0, 0], [2, 0], [2, 2], [0, 2]]
    concave = [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]]
    checks = [(square, [1, 1], True), (square[::-1], [1, 1], True),
              (square, [2, 1], True), (square, [2.001, 1], False),
              (concave, [1.5, 1.5], False), (concave, [.5, 1.5], True),
              ([[0, 0]], [0, 0], True), ([[0, 0]], [.001, 0], False),
              ([[0, 0], [2, 0]], [1, 0], True),
              ([[0, 0], [2, 0]], [1, .001], False)]
    micro_dent = [[-1485.7744640310023, -83.92557238910767],
                  [-1491.5134372635835, -78.18659915652651],
                  [-1577.5501038696184, -278.2979022559528],
                  [-1577.5501038314858, -278.29790214548973]]
    truth = np.array([-1562.3300404485096, -245.32353442855123])
    checks.append((micro_dent, truth, True))
    assert not _contains_convex(micro_dent, truth)
    for polygon, point, expected in checks:
        assert contains_polygon(polygon, point) == expected, (polygon, point, expected)
    return len(checks)


def main():
    tested = regression_checks()
    selection = json.loads((ROOT / 'results/cost_selection.json').read_text())
    assert hashes() == selection['source_sha256'], 'Frozen model or old auditor changed'
    centers, rows, disagreements = independent_cells(), [], []
    for folder in ('cost-holdout-uniform', 'cost-holdout-sparse', 'cost-holdout-boundary', 'cost-stress'):
        for file in sorted((ROOT / 'results' / folder / 'cases').glob('*.json')):
            data = json.loads(file.read_text())
            result = data['result']
            arena = BoundedArena(data['sources'], result['seed'], result['field'], max_actions=3000)
            for index, event in enumerate(data['events']):
                action = getattr(arena, event['action'])
                action(event['position'], event['channel'])
                assert arena.events[-1] == event, (str(file), index, 'feedback replay mismatch')
            assert arena.evaluation()['all_cleared'] and result['all_cleared']
            assert abs(arena.time_s - result['total_s']) < 1e-8
            timing = independent_time_audit(arena)
            observations = observation_audit(arena)
            certificate = certificate_audit(arena, data['outcome'], observations, centers)
            polygons, optical = 0, 0
            assert not any(r.get('phase') == 'belief_conflict' for r in data['trace'])
            for index, row in enumerate(data['trace']):
                if 'polygon' not in row or 'channel' not in row:
                    continue
                P = np.asarray(row['polygon'], float)
                g = np.asarray(arena._sources[row['channel']]['position'], float)
                distance = boundary_distance(P, g)
                general = contains_polygon(P, g)
                independent = bool((len(P) >= 3 and PlotPath(P).contains_point(g)) or distance <= TOLERANCE_M)
                assert general == independent, (str(file), index, 'membership cross-check mismatch')
                assert general, (str(file), index, 'true source excluded')
                if _contains_convex(P, g) != general:
                    disagreements.append(dict(file=str(file.relative_to(ROOT)), trace_index=index,
                        phase=row.get('phase'), channel=row['channel'], polygon=P.tolist(), truth=g.tolist(),
                        minimum_edge_m=float(np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1).min()),
                        truth_to_boundary_m=distance, convex_halfplanes=False,
                        ray_crossing=general, matplotlib_path=independent))
                polygons += 1
                if row.get('optical_path') is not None:
                    assert independent_disk_cover(P, row['optical_path'])['complete'], (str(file), index)
                    optical += 1
            rows.append(dict(name=result['name'], schedule=result['schedule'],
                original_audit_passed=result['passed'], supplemental_audit_passed=True,
                actions=timing['actions'], polygons_checked=polygons, optical_paths_checked=optical,
                certificate_basis=certificate['independent_basis'],
                saved_case_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        print(json.dumps(dict(folder=folder, audited_runs=len(rows), disagreements=len(disagreements))), flush=True)
    frozen = hashes() == selection['source_sha256']
    assert len(rows) == 78 and frozen
    output = dict(local_offline_replay_only=True, official_simulator_used=False,
        original_results_modified=False, convex_hull_substitution=False,
        boundary_tolerance_m=TOLERANCE_M, regression_cases_passed=tested,
        source_snapshot_stable=frozen, runs=len(rows),
        original_audit_passed=sum(r['original_audit_passed'] for r in rows),
        supplemental_audit_passed=sum(r['supplemental_audit_passed'] for r in rows),
        polygons_checked=sum(r['polygons_checked'] for r in rows),
        optical_paths_checked=sum(r['optical_paths_checked'] for r in rows),
        original_false_negative_disagreements=disagreements, rows=rows,
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        old_auditor_sha256=hashes()['validate_model.py'])
    (ROOT / 'results/cost_polygon_audit.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in output.items() if k not in ('rows', 'original_false_negative_disagreements')}))


if __name__ == '__main__':
    main()
