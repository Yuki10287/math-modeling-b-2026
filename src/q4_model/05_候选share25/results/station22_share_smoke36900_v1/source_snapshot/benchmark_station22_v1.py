"""Frozen five-layout local pairing: original share25 versus new 14+7+1 cover."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
Q4 = next(p for p in HERE.parents if (p/'polar_cover.py').is_file())
PROJECT = Q4.parent.parent
sys.path.insert(0, str(Q4))
from shared import load_file
from polar_cover import PolarCover
from task_sharing_solver import solve_multi as baseline
from share22_geometry_experiment_v1 import solve_multi as candidate
from station22_geometry_v1 import CERTIFICATE
from validate_station22_v1 import validate as validate_exact_candidate
from refined_cover_audit import exact_certificate_audit as validate_exact_baseline

ENV = Q4.parent/'q3_model_v2/00_主方案_lean/code/environment.py'
VAL = Q4/'00_公共几何与定位/code/validation.py'
CASES = Q4/'01_初版31站/code/benchmark.py'
POP = Q4/'04_22站覆盖实验/code/benchmark_cells.py'
LocalArena = load_file('_station22_local_arena', ENV).LocalArena
validate_run = load_file('_station22_trajectory_validation', VAL).validate_run
for path, names in ((CASES, {'make_case', 'public_api'}), (POP, {'sources_for'})):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), globals())


def hashes():
    paths = [Q4/name for name in ('q4_official_client.py', 'q4_share25_client.py', 'solver.py',
        'joint_solver.py', 'task_sharing_solver.py', 'shared.py', 'directional_cover.py',
        'polar_cover.py', 'localization.py', 'route_planning.py', 'guarded_policy.py')]
    paths += [Q4.parent/'q3_model_v2/geometry.py', ENV, VAL, CASES, POP, CERTIFICATE]
    paths += [HERE/name for name in ('station22_geometry_v1.py', 'share22_geometry_experiment_v1.py',
        'validate_station22_v1.py', 'audit_station_certificates_v1.py', 'refined_cover_audit.py',
        'benchmark_station22_v1.py')]
    return {str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--start-seed', type=int, default=37000)
    parser.add_argument('--populations', default='uniform,boundary,cluster,omni_heavy,dir_heavy')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    def save(name, obj):
        with (args.out/name).open('x', encoding='utf-8') as stream:
            json.dump(obj, stream, ensure_ascii=False, indent=2)
    snapshot = hashes()
    populations = args.populations.split(',')
    save('manifest.json', dict(local_only=True, official_contacted=False, source_hashes=snapshot,
        populations=populations, start_seed=args.start_seed, field='smooth', candidate='station22_14_outer_7_inner_r970',
        fixed_before_run=True, parameter_tuning_during_run=False))
    summaries = []
    for index, population in enumerate(populations):
        seed = args.start_seed+100*index
        sources = sources_for(seed, population)
        for variant, solve in (('baseline_share25', baseline), ('station22_share', candidate)):
            arena, trace = LocalArena(sources, seed, 'smooth'), []
            result, error, audit = None, None, dict(passed=False)
            begin = time.perf_counter()
            try:
                result = solve(public_api(arena), variant='share25', trace=trace)
            except Exception as exc:
                error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
            elapsed = time.perf_counter()-begin
            try:
                assert error is None, error
                if variant == 'baseline_share25':
                    mesh = PolarCover()
                    audit = validate_run(sources, arena.events, trace, result, mesh.stations, mesh.indices)
                    audit['exact_certificate'] = validate_exact_baseline(result, arena.events)
                else:
                    exact, adapted, vertices, triangles = validate_exact_candidate(result, arena.events, CERTIFICATE)
                    audit = validate_run(sources, arena.events, trace, adapted, vertices, triangles)
                    audit['exact_certificate'] = exact
            except Exception as exc:
                audit = dict(passed=False, type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
            row = dict(seed=seed, population=population, variant=variant, field='smooth',
                runtime_s=elapsed, error=error, audit=audit, **arena.evaluation())
            save(f'{seed}-{population}-{variant}.json', dict(summary=row, sources=sources,
                result=result, events=arena.events, trace=trace))
            summaries.append(row)
            print('CASE', seed, population, variant, 'time', row['time_s'], 'audit', audit['passed'],
                'runtime', round(elapsed, 2), 'error', error, flush=True)
    stable = snapshot == hashes()
    passed = stable and all(row['audit']['passed'] and row['error'] is None for row in summaries)
    save('summary.json', dict(passed=passed, source_snapshot_stable=stable, local_only=True,
        official_contacted=False, rows=summaries, source_hashes=snapshot))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
