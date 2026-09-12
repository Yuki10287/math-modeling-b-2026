"""Additional local geometry rejection and exact candidate certification.

Reads the preserved coarse screening; writes only a new chosen output folder.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from screen_station_geometry_v1 import stations, assess, grid, route, exact_certificate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    def save(name, obj):
        with (args.out/name).open('x', encoding='utf-8') as stream:
            json.dump(obj, stream, indent=2)
    rows = json.loads(args.input.read_text())['passing_rows']
    for m, n in ((17, 6), (17, 7), (18, 6)):
        for radius in range(900, 1001, 10):
            for fraction in (0., .25, .5):
                phase = fraction*2*np.pi/np.lcm(m, n)
                q = stations(m, n, radius, phase)
                rows.append(dict(m=m, n=n, inner_radius=radius, phase=phase, count=len(q), **route(q)))
    dense = grid(360, 181)
    passed, rejected = [], []
    for index, row in enumerate(rows):
        q = stations(row['m'], row['n'], row['inner_radius'], row['phase'])
        first_bad = None
        for begin in range(0, len(dense), 12000):
            check = assess(q, dense[begin:begin+12000])
            if check['bad']:
                first_bad = check
                break
        if first_bad:
            rejected.append(dict(**row, first_rejecting_block=first_bad))
        else:
            passed.append(row)
        if index % 100 == 0:
            print('DENSE', index, 'passed', len(passed), 'wall_s', round(time.perf_counter()-start, 1), flush=True)
    passed.sort(key=lambda row:row['route_m'])
    save('dense_screening.json', dict(checked=len(rows), dense_points=len(dense),
        passed=passed, rejected=rejected, input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest()))
    chosen, certified = set(), []
    for row in passed:
        if row['count'] in chosen:
            continue
        q = stations(row['m'], row['n'], row['inner_radius'], row['phase'])
        cert = exact_certificate(q, max_depth=18)
        name = 'candidate_'+str(len(certified))
        save(name+'_certificate.json', cert)
        result = dict(**row, name=name, certificate_passed=cert['passed'],
            cells=cert.get('cell_count'), failures=cert.get('failed_count'))
        certified.append(result)
        print('EXACT', result, flush=True)
        if cert['passed']:
            chosen.add(row['count'])
        if len(chosen) >= 3 or len(certified) >= 10:
            break
    save('summary.json', dict(local_geometry_only=True, official_contacted=False,
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__), Path(__file__).with_name('screen_station_geometry_v1.py'))},
        candidates=certified, elapsed_s=time.perf_counter()-start,
        note='Dense grid only rejects. Exact cell proof accepts. Feasible heuristic routes are not optimality or full-simulation claims.'))


if __name__ == '__main__':
    main()
