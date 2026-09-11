"""Predeclared correctness stress cases; no HTTP or official simulator access."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
from benchmark import LocalArena, public_api
from benchmark_joint import hashes
from joint_solver import solve_multi
from polar_cover import PolarCover
from validation import validate_run


class EndpointArena(LocalArena):
    """Fixed spatial fields that attain the exact allowed one-degree endpoints."""
    def _error(self, q, channel):
        if self._field == 'plus_one':
            return 1.0
        if self._field == 'minus_one':
            return -1.0
        if self._field == 'checker_one':
            return 1.0 if (math.floor(q[0]/37)+math.floor(q[1]/41)+channel)%2 else -1.0
        return super()._error(q, channel)


def stress_cases():
    cases = []
    specs = [
        ('outward_min_radius', 10, 'outer ring, outward headings, R=1000'),
        ('tangent_min_radius', 12, 'outer ring, alternating tangent headings, R=1000'),
        ('mesh_vertices', 16, 'sources at inner scan vertices and triangle seams'),
        ('near_clear_thresholds', 10, 'origin distances 0, 5 +/- epsilon and 20 +/- epsilon'),
        ('almost_collinear', 12, 'almost collinear sources across the full target disk'),
        ('almost_coincident', 16, 'sixteen distinct channels within a small off-center cluster'),
        ('one_directional', 10, 'one outward directional source and nine omnidirectional sources'),
        ('one_omni', 10, 'nine outward directional sources and one omnidirectional source'),
        ('max_radius_wrap', 12, 'R=1500, mixed types, headings near the zero-angle seam'),
        ('mixed_radius_endpoints', 16, 'R alternates between 1000 and 1500, mixed radii and headings'),
    ]
    for index, (name, count, purpose) in enumerate(specs):
        sources = []
        for k in range(count):
            a = 2*math.pi*k/count + (1e-7 if index%2 else 0)
            r = 1800.0
            radius = 1000.0
            heading = a
            directional = k != 0
            if name == 'tangent_min_radius':
                heading = a + (math.pi/2 if k%2 else -math.pi/2)
            elif name == 'mesh_vertices':
                a = 2*math.pi*(k%12)/12 + math.pi/12
                r = 960.0 if k < 12 else 480.0
                heading = a + math.pi/2
            elif name == 'near_clear_thresholds':
                r = [0, 5-1e-6, 5, 5+1e-6, 20-1e-6, 20, 20+1e-6, 25, 1000, 1500][k]
                a = 0.0
                heading = math.pi if k%2 else 0.0
                directional = k%2 == 1
            elif name == 'almost_collinear':
                x = -1799.0 + 3598.0*k/(count-1)
                y = 1e-6*(-1)**k
                r, a = math.hypot(x, y), math.atan2(y, x)
                heading = math.pi/2 if k%2 else -math.pi/2
                directional = k%3 != 0
            elif name == 'almost_coincident':
                x, y = 1230+1e-4*math.cos(a), -730+1e-4*math.sin(a)
                r, a = math.hypot(x, y), math.atan2(y, x)
                heading = 2*math.pi*k/count
            elif name == 'one_directional':
                directional = k == count-1
            elif name == 'max_radius_wrap':
                r = 750 if k%2 else 1500
                radius = 1500.0
                heading = 1e-10 if k%2 else 2*math.pi-1e-10
                directional = k%2 == 0
            elif name == 'mixed_radius_endpoints':
                r = [0, 960, 1799.999, 1800][k%4]
                radius = 1000.0 if k%2 else 1500.0
                heading = a + k*math.pi/7
                directional = k%4 != 0
            sources.append(dict(channel=(7*k+3)%20+1,
                position=[r*math.cos(a), r*math.sin(a)], radius=radius,
                orientation=heading if directional else None))
        assert 10 <= len(sources) <= 16
        assert len({s['channel'] for s in sources}) == count
        assert any(s['orientation'] is None for s in sources)
        assert any(s['orientation'] is not None for s in sources)
        assert all(np.linalg.norm(s['position']) <= 1800+1e-9 for s in sources)
        cases.append(dict(name=name, seed=19000+index, purpose=purpose, sources=sources))
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cases = stress_cases()
    fields = ['plus_one', 'minus_one', 'checker_one']
    snapshot = hashes()
    manifest = dict(local_only=True, official_simulator_used=False, variant='shared',
        purpose='correctness stress; not an unbiased performance benchmark',
        fields=fields, cases=cases, source_sha256=snapshot,
        test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rounding='nearest 0.01 degree, after fixed error in [-1,1]')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    rows = []
    mesh = PolarCover()
    for case in cases:
        for field in fields:
            arena = EndpointArena(case['sources'], case['seed'], field)
            trace = []
            started = time.perf_counter()
            result = None
            try:
                result = solve_multi(public_api(arena), variant='shared', trace=trace)
                runtime = time.perf_counter()-started
                audit = validate_run(case['sources'], arena.events, trace, result, mesh.stations, mesh.indices)
            except Exception as exc:
                runtime = time.perf_counter()-started
                audit = dict(passed=False, error_type=type(exc).__name__, message=str(exc))
            row = dict(case=case['name'], field=field, runtime_s=runtime, audit=audit, **arena.evaluation())
            rows.append(row)
            record = dict(summary=row, sources=case['sources'], result=result, events=arena.events, trace=trace)
            (out/f'{case["name"]}-{field}.json').write_text(json.dumps(record), encoding='utf-8')
            (out/'rows.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
            print(f'{case["name"]} {field}: {row["cleared"]}/{row["source_count"]}, '
                  f'{row["total_s"]:.2f}s, audit={audit["passed"]}', flush=True)
    stable = hashes() == snapshot
    summary = dict(runs=len(rows), independent_layouts=len(cases),
        passed=sum(r['audit']['passed'] for r in rows),
        all_cleared=sum(r['all_cleared'] for r in rows), source_instances=sum(r['source_count'] for r in rows),
        source_snapshot_stable=stable, local_only=True,
        maximum_total_s=max(r['total_s'] for r in rows), maximum_runtime_s=max(r['runtime_s'] for r in rows),
        audit_counts={k:sum(r['audit'].get(k,0) for r in rows) for k in
                      ('events','polygons','optical_plans','triangle_certificates')})
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)
    raise SystemExit(0 if stable and summary['passed']==len(rows) else 1)


if __name__ == '__main__':
    main()
