"""Predeclared local correctness stress runs for 25-station task sharing.

The ten geometrically difficult layouts are each tested under three fixed
endpoint error fields. This is not an unbiased performance comparison.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from benchmark import public_api
from polar_cover import PolarCover
from stress_check import EndpointArena, stress_cases
from task_sharing_solver import VARIANTS, solve_multi
from validation import validate_run


FIELDS = ('plus_one', 'minus_one', 'checker_one')


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='new output directory; existing paths are rejected')
    parser.add_argument('--variant', choices=VARIANTS, default='resume25')
    parser.add_argument('--fields', default=','.join(FIELDS),
                        help='comma-separated fixed error fields; default is all three')
    parser.add_argument('--cases', default='',
                        help='optional comma-separated stress case names; default is all ten')
    args = parser.parse_args()
    from benchmark_task_sharing import hashes

    fields = args.fields.split(',')
    if not fields or len(set(fields)) != len(fields) or not set(fields) <= set(FIELDS):
        parser.error('--fields must contain distinct names from '+','.join(FIELDS))
    cases = stress_cases()
    if args.cases:
        names = args.cases.split(',')
        known = {case['name'] for case in cases}
        if len(set(names)) != len(names) or not set(names) <= known:
            parser.error('--cases must contain distinct known stress case names')
        cases = [case for case in cases if case['name'] in names]
    snapshot = hashes()
    script_path = Path(__file__).resolve()
    stress_path = script_path.with_name('stress_check.py')
    script_hash, stress_hash = _sha256(script_path), _sha256(stress_path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(arguments=vars(args), variant=args.variant,
                    local_only=True, official_simulator_used=False,
                    purpose='correctness stress; not an unbiased performance benchmark',
                    expected_runs=len(cases)*len(fields), fields=fields, cases=cases,
                    source_sha256=snapshot, test_sha256=script_hash,
                    stress_source_sha256=stress_hash, python=platform.python_version(),
                    numpy=np.__version__,
                    runtime_scope='solver only, excludes audit and serialization',
                    rounding='nearest 0.01 degree, after fixed error in [-1,1]')
    _write(out/'manifest.json', manifest)
    rows, mesh = [], PolarCover()
    for case in cases:
        for field in fields:
            arena = EndpointArena(case['sources'], case['seed'], field)
            trace, result = [], None
            started = time.perf_counter()
            try:
                try:
                    result = solve_multi(public_api(arena), variant=args.variant, trace=trace)
                finally:
                    runtime = time.perf_counter()-started
                audit = validate_run(case['sources'], arena.events, trace, result,
                                     mesh.stations, mesh.indices)
            except Exception as exc:
                audit = dict(passed=False, error_type=type(exc).__name__, message=str(exc))
            row = dict(case=case['name'], seed=case['seed'], field=field, variant=args.variant,
                       runtime_s=runtime, audit=audit, **arena.evaluation())
            rows.append(row)
            record = dict(summary=row, sources=case['sources'], events=arena.events,
                          trace=trace, result=result)
            _write(out/f'{case["name"]}-{field}-{args.variant}.json', record)
            _write(out/'rows.json', rows)
            print(f'{case["name"]} {field} {args.variant}: '
                  f'{row["cleared"]}/{row["source_count"]}, {row["total_s"]:.2f}s, '
                  f'runtime={runtime:.2f}s, audit={audit["passed"]}', flush=True)
    source_stable = hashes() == snapshot
    test_stable = _sha256(script_path) == script_hash and _sha256(stress_path) == stress_hash
    summary = dict(variant=args.variant, runs=len(rows), independent_layouts=len(cases),
                   passed=sum(row['audit']['passed'] for row in rows),
                   all_cleared=sum(row['all_cleared'] for row in rows),
                   source_instances=sum(row['source_count'] for row in rows),
                   source_snapshot_stable=source_stable, stress_snapshot_stable=test_stable,
                   local_only=True, official_simulator_used=False,
                   maximum_total_s=max(row['total_s'] for row in rows),
                   maximum_runtime_s=max(row['runtime_s'] for row in rows),
                   audit_counts={key:sum(row['audit'].get(key, 0) for row in rows) for key in
                       ('events', 'polygons', 'optical_plans', 'triangle_certificates')})
    _write(out/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    successful = (source_stable and test_stable and summary['passed'] == len(rows)
                  and summary['all_cleared'] == len(rows))
    raise SystemExit(0 if successful else 1)


if __name__ == '__main__':
    main()
