"""Geometry-only audit of observed service stops; never a replay of new actions.

The retrospective absent-channel labels only check the supplied archive. Every
additional negative measurement below is explicitly hypothetical and free.
No simulator client is imported or contacted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from scipy.spatial import ConvexHull, QhullError

Q4 = next(p for p in Path(__file__).resolve().parents if (p/'polar_cover.py').is_file())
sys.path.insert(0, str(Q4))
from polar_cover import PolarCover


def subdivide(cells):
    a, b, c = cells[:, 0], cells[:, 1], cells[:, 2]
    ab, bc, ca = (a+b)/2, (b+c)/2, (c+a)/2
    return np.stack((np.stack((a, ab, ca), 1), np.stack((ab, b, bc), 1),
                     np.stack((ca, bc, c), 1), np.stack((ab, bc, ca), 1)), 1).reshape(-1, 3, 2)


def covered(cells, points):
    """Independent normalized half-space test of the continuous certificate."""
    eligible = np.linalg.norm(cells[:, :, None, :]-points, axis=3).max(axis=1) <= 1000-1e-6
    for cell_id, cell in enumerate(cells):
        witnesses = np.unique(points[eligible[cell_id]], axis=0)
        if len(witnesses) < 3:
            return False
        try:
            hull = ConvexHull(witnesses)
        except QhullError:
            return False
        if np.max(cell @ hull.equations[:, :2].T+hull.equations[:, 2]) > 1e-8:
            return False
    return True


def audit_case(trace, summary, declared_count):
    actions = [r for r in trace if r.get('phase') == 'actual_action']
    assert summary['variant'] == 'share25' and summary['complete'] is True
    assert summary['successful_clear_count'] == declared_count
    absent = summary['certificate']['absent']
    actual_absent_measurements = [r for r in actions if r['action'] == 'measure' and r['channel'] in absent]
    assert all(r['result'] == 'no_signal' for r in actual_absent_measurements)
    assert set(absent) | set(summary['cleared']) == set(range(1, 21))
    service = [r for r in actions if r['reason'] == 'source_measure' or
               (r['action'] == 'clear' and r['result'] == 'success'
                and r['reason'] in ('near_clear', 'certified_clear', 'optical_cover'))]
    stops = []
    seen = set()
    for row in service:
        key = tuple(row['position'])
        if key not in seen:
            stops.append(dict(position=row['position'], event=row['event'], reason=row['reason']))
            seen.add(key)
    extra = np.asarray([r['position'] for r in stops])
    cover = PolarCover()
    station_first_scan = {}
    for row in actions:
        if row['reason'] == 'scan':
            distances = np.linalg.norm(cover.stations-row['position'], axis=1)
            k = int(np.argmin(distances))
            assert distances[k] < 1e-7
            station_first_scan.setdefault(k, row['event'])
    levels = []
    for level in range(3):
        cells = cover.triangles.copy()
        for _ in range(level):
            cells = subdivide(cells)
        assert covered(cells, cover.stations)
        parents = np.repeat(np.arange(36), 4**level)

        def valid_removed(removed, additions):
            # Unaffected parent triangles still have all three original vertex
            # witnesses; only cells in parents incident to a removed station
            # need another independent hull test.
            touched = np.any(np.isin(cover.indices, list(removed)), axis=1)
            affected = cells[touched[parents]]
            points = np.vstack((cover.stations[[k for k in range(25) if k not in removed]], additions))
            return covered(affected, points)

        removable = [k for k in range(25) if valid_removed({k}, extra)]
        greedy = set()
        for k in [k for k in removable if k != 0]:
            if valid_removed(greedy | {k}, extra):
                greedy.add(k)
        remaining = [k for k in range(25) if k not in greedy]
        assert covered(cells, np.vstack((cover.stations[remaining], extra)))
        temporal = []
        if level == 2:
            for i, stop in enumerate(stops):
                future = [k for k in range(1, 13) if station_first_scan.get(k, float('inf')) > stop['event']]
                current = [k for k in future if valid_removed({k}, extra[i:i+1])]
                prefix = [k for k in future if valid_removed({k}, extra[:i+1])]
                if current or prefix:
                    temporal.append(dict(event=stop['event'], position=stop['position'],
                        current_stop_only_removable=current, free_past_stops_removable=prefix))
        levels.append(dict(level=level, cells=len(cells),
            individually_removable_station_ids=removable,
            individually_removable_future_inner_count=sum(1 <= k <= 12 for k in removable),
            jointly_feasible_greedy_inner_station_ids=sorted(greedy),
            greedy_is_not_a_maximum_cardinality_claim=True,
            joint_plan_independently_validated=True, temporal_upper_bound=temporal,
            temporal_current_stop_any=any(r['current_stop_only_removable'] for r in temporal)))
    radii = np.linalg.norm(extra, axis=1)
    outer_gaps = []
    for k in range(13, 25):
        normal = cover.stations[k]/cover.outer_radius
        points = np.vstack((np.delete(cover.stations, k, axis=0), extra))
        outer_gaps.append(dict(station=k, support_gap_m=float(cover.outer_radius-np.max(points @ normal))))
    assert all(r['support_gap_m'] > 1e-6 for r in outer_gaps)
    return dict(declared_sources=declared_count, cleared=summary['successful_clear_count'],
        official_total_virtual_s=summary['total_virtual_s'], retrospective_absent_channels=absent,
        recorded_absent_measurements_all_negative=True,
        service_stops=stops, service_stop_count=len(stops),
        radius_min_m=float(radii.min()), radius_max_m=float(radii.max()),
        service_points_in_960_to_1000_annulus=int(((radii >= 960)&(radii <= 1000)).sum()),
        outer_station_separating_support_gaps=outer_gaps, levels=levels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--declared-counts', default='14,13,14,10')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    counts = list(map(int, args.declared_counts.split(',')))
    rows = []
    with zipfile.ZipFile(args.archive) as archive:
        names = sorted(n for n in archive.namelist() if n.endswith('.jsonl.beliefs.json'))
        assert len(names) == len(counts)
        for name, count in zip(names, counts):
            trace_bytes = archive.read(name)
            records = [json.loads(line) for line in archive.read(name[:-len('.beliefs.json')]).splitlines() if line]
            summaries = [r for r in records if r.get('type') == 'session_summary']
            assert len(summaries) == 1
            row = audit_case(json.loads(trace_bytes), summaries[0], count)
            row.update(case=Path(name).name, trace_sha256=hashlib.sha256(trace_bytes).hexdigest())
            rows.append(row)
            print(row['case'], [x['individually_removable_future_inner_count'] for x in row['levels']], flush=True)
    report = dict(local_only=True, official_simulator_contacted=False,
        archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_cover_source_sha256={n:hashlib.sha256((Q4/n).read_bytes()).hexdigest()
            for n in ('polar_cover.py', 'directional_cover.py')},
        purpose='Certificate precision and service-stop geometric potential only; not counterfactual performance.',
        relaxation='Observed service endpoints are hypothetically available for free and negative; final absent labels are used only retrospectively in this auditor.',
        exclusions='No invented simulator feedback, no new official score, no claimed joint maximum, no source count supplied to any solver.',
        certificate='Each cell is contained in the convex hull of witnesses within 1000-1e-6 m of all cell vertices; midpoint refinement preserves the exact geometric domain.',
        rows=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
