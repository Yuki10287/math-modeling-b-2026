"""Isolated local development comparison; never contacts an official client."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
Q4 = next(p for p in Path(__file__).resolve().parents if (p/'polar_cover.py').is_file())
PROJECT = Q4.parent.parent
sys.path.insert(0, str(Q4))
from shared import load_file
from polar_cover import PolarCover
from task_sharing_solver import solve_multi as baseline
from refined_cover_experiment import solve_multi as candidate
from refined_cover_geometry import RefinedCover

VALIDATION = Q4/'00_公共几何与定位/code/validation.py'
ENVIRONMENT = Q4.parent/'q3_model_v2/00_主方案_lean/code/environment.py'
INITIAL_CASES = Q4/'01_初版31站/code/benchmark.py'
POPULATIONS_SOURCE = Q4/'04_22站覆盖实验/code/benchmark_cells.py'
LocalArena = load_file('_refined_test_environment', ENVIRONMENT).LocalArena
validate_run = load_file('_refined_test_validation', VALIDATION).validate_run

# Reuse the exact case-generator and public-interface function ASTs without
# importing historical benchmark mains or their unrelated runtime modules.
for path, names in ((INITIAL_CASES, {'make_case', 'public_api'}),
                    (POPULATIONS_SOURCE, {'sources_for'})):
    import math
    source = ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), globals())

POPULATIONS = ('uniform', 'boundary', 'cluster', 'omni_heavy', 'dir_heavy')


def hashes():
    paths = [Q4/n for n in ('q4_official_client.py', 'q4_share25_client.py', 'solver.py',
        'joint_solver.py', 'task_sharing_solver.py', 'shared.py', 'directional_cover.py',
        'polar_cover.py', 'localization.py', 'route_planning.py', 'guarded_policy.py')]
    paths += [Q4.parent/'q3_model_v2/geometry.py', ENVIRONMENT, VALIDATION, INITIAL_CASES, POPULATIONS_SOURCE]
    paths += [HERE/n for n in ('refined_cover_geometry.py', 'refined_cover_experiment.py',
                             'benchmark_refined_service_cover.py')]
    if (HERE/'refined_cover_audit.py').is_file():
        paths.append(HERE/'refined_cover_audit.py')
    return {p.relative_to(PROJECT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def run_case(seed, population, variant):
    sources = sources_for(seed, population)
    arena, trace = LocalArena(sources, seed, 'smooth'), []
    result, error = None, None
    started = time.perf_counter()
    try:
        result = (baseline if variant == 'baseline_share25' else candidate)(
            public_api(arena), variant='share25', trace=trace)
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    elapsed = time.perf_counter()-started
    mesh = PolarCover() if variant == 'baseline_share25' else RefinedCover()
    try:
        if error:
            raise AssertionError(error['message'])
        audit = validate_run(sources, arena.events, trace, result, mesh.stations, mesh.indices)
        domain = dict(passed=True, original_frozen_cover=True) if variant == 'baseline_share25' else mesh.independent_domain_audit()
    except Exception as exc:
        audit = dict(passed=False, type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        domain = dict(passed=False)
    row = dict(seed=seed, population=population, variant=variant, field='smooth',
        runtime_s=elapsed, audit=audit, domain_audit=domain, error=error,
        experiment={} if result is None else result.get('experiment', {}), **arena.evaluation())
    return dict(summary=row, sources=sources, events=arena.events, trace=trace, result=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--populations', default=','.join(POPULATIONS))
    parser.add_argument('--start-seed', type=int, default=34000)
    args = parser.parse_args()
    populations = args.populations.split(',')
    assert set(populations) <= set(POPULATIONS)
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot = hashes()
    write(args.out/'manifest.json', dict(populations=populations, start_seed=args.start_seed,
        source_sha256=snapshot, numerical_development_prototype=True,
        local_only=True, official_simulator_contacted=False,
        source_counts='Generated locally; official batch6 counts are never supplied to a solver',
        purpose='Development only, fixed before evaluation; no official promotion'))
    rows = []
    for population in populations:
        seed = args.start_seed+100*POPULATIONS.index(population)
        for variant in ('baseline_share25', 'refined_service_cover'):
            record = run_case(seed, population, variant)
            row = record['summary']
            write(args.out/f'{seed}-{population}-{variant}.json', record)
            rows.append(row)
            print(f"{seed} {population} {variant}: {row['total_s']:.3f}s; "
                  f"{row['cleared']}/{row['source_count']}; audit={row['audit']['passed']}; "
                  f"runtime={row['runtime_s']:.2f}; {row['experiment']}", flush=True)
            if not row['audit']['passed']:
                print(row['audit'], flush=True)
    paired = []
    for population in populations:
        a,b = [r for r in rows if r['population'] == population]
        paired.append(dict(population=population, baseline_s=a['total_s'], candidate_s=b['total_s'],
            saved_s=a['total_s']-b['total_s'], reduction_pct=100*(a['total_s']-b['total_s'])/a['total_s']))
    stable = hashes() == snapshot
    report = dict(passed=stable and all(r['audit']['passed'] for r in rows), rows=rows,
        pairs=paired, source_snapshot_stable=stable, local_only=True, official_simulator_contacted=False,
        exact_certificate_audit_integrated=False, numerical_development_prototype=True)
    write(args.out/'summary.json', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}, ensure_ascii=False), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
