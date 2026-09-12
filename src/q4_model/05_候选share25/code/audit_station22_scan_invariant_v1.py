"""Read-only trajectory audit of the fixed-station unknown-channel invariant.

Unlike the earlier refined service-scan experiment, share25 only measures still
unknown channels at the initial origin and canonical fixed scan destinations.
Consequently a close-but-distinct service point cannot enter their ledger.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def audit_case(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    stations = np.asarray(data['result']['certificate']['stations'], float)
    station_keys = {tuple(q) for q in stations}
    assert (0., 0.) in station_keys
    distances = np.linalg.norm(stations[:, None, :]-stations[None, :, :], axis=2)
    distances += np.eye(len(stations))*1e20
    separation = float(np.min(distances))
    assert separation > 1e-7
    negative, discovered = {c: [] for c in range(1, 21)}, set()
    action_count = checks = task_checks = 0
    for row in data['trace']:
        if row['phase'] == 'task_choice' and row['task'] == 'scan':
            q = stations[row['index']]
            for c in range(1, 21):
                if c not in discovered:
                    near = any(np.linalg.norm(q-p) < 1e-7 for p in negative[c])
                    equal = any(np.array_equal(q, p) for p in negative[c])
                    assert near == equal
                    task_checks += 1
        if row['phase'] != 'actual_action':
            continue
        event = data['events'][action_count]
        action_count += 1
        assert event['action'] == row['action'] and event['channel'] == row['channel']
        assert tuple(event['position']) == tuple(row['position'])
        if event['action'] != 'measure':
            continue
        c, q = event['channel'], np.asarray(event['position'])
        if c not in discovered:
            assert row['reason'] == 'scan'
            assert tuple(q) in station_keys
            assert all(tuple(p) in station_keys for p in negative[c])
            near = any(np.linalg.norm(q-p) < 1e-7 for p in negative[c])
            equal = any(np.array_equal(q, p) for p in negative[c])
            assert near == equal
            checks += 1
        if event['measure_result'] == 'no_signal':
            negative[c].append(q)
        else:
            discovered.add(c)
    assert action_count == len(data['events'])
    return dict(passed=True, path=str(path), input_sha256=hashlib.sha256(raw).hexdigest(),
        variant=data['summary']['variant'], seed=data['summary']['seed'],
        unknown_measurements_checked=checks, planned_scan_channel_checks=task_checks,
        minimum_distinct_station_separation_m=separation,
        unknown_negative_points_are_canonical_stations=True,
        near_and_exact_scan_dedup_identical_on_actual_path=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directories', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for directory in args.directories:
        for path in sorted(directory.glob('*.json')):
            if path.name[0].isdigit():
                rows.append(audit_case(path))
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(dict(passed=all(row['passed'] for row in rows), runs=len(rows), rows=rows,
            auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope='recorded paths and the stated origin-start fixed-unknown-scan invariant; no arbitrary-start claim'), stream, indent=2)
    print('PASSED', len(rows), 'paths; unknown measurements', sum(row['unknown_measurements_checked'] for row in rows))


if __name__ == '__main__':
    main()
