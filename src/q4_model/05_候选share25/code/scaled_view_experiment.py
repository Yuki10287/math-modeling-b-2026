"""Scale-aware, sign-symmetric view candidates; original score and safety rules.

The original finite pool uses transverse offsets of 40/100 m and only one sign
of an along-axis displacement. We retain every original view and add opposite
along-axis views plus transverse baselines r/2 and r, where r is the current
position polygon's minimum enclosing radius. No source truth or test outcome
sets these geometric scales. This changes action candidates, not probability
weights, feedback updates, optical coverage, source scheduling or stopping.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from types import FunctionType

import numpy as np

HERE = Path(__file__).resolve().parent
Q4 = HERE.parents[1]
PROJECT = Q4.parent.parent
sys.path.insert(0, str(Q4))

import localization
import task_sharing_solver
import q4_share25_client
from polar_cover import PolarCover
from continuous_hypothesis_experiment import (
    LocalArena, public_api, sources_for, validate_run, ENVIRONMENT_PATH,
    VALIDATION_PATH, POPULATIONS,
)


def additional_views(center, radius, u, v):
    result = [center+.35*radius*u+offset*v for offset in (-100., -40., 40., 100.)]
    result.extend(center+along*u+offset*v
                  for along in (0., -.35*radius, .35*radius)
                  for offset in (-radius, -.5*radius, .5*radius, radius))
    return result


def make_solver():
    parsed = ast.parse(Path(localization.__file__).read_text(encoding='utf-8-sig'))
    function = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'choose_measure')
    inserted = 0
    for index, statement in enumerate(function.body):
        if ast.dump(statement) == ast.dump(ast.parse('best = None').body[0]):
            function.body[index:index] = ast.parse('candidates.extend(_additional_views(center, r, u, v))').body
            inserted += 1
            break
    assert inserted == 1
    namespace = localization.__dict__.copy()
    namespace['_additional_views'] = additional_views
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(Path(__file__)), 'exec'), namespace)
    solver_globals = task_sharing_solver.__dict__.copy()
    solver_globals['choose_measure'] = namespace['choose_measure']
    original = task_sharing_solver.solve_multi
    return FunctionType(original.__code__, solver_globals, original.__name__, original.__defaults__, original.__closure__)


def candidate_geometry_check():
    center, radius = np.array([137., -289.]), 317.
    theta = .73
    u = np.array([np.cos(theta), np.sin(theta)])
    v = np.array([-u[1], u[0]])
    def pool(a, b):
        original = [center+along*a+offset*b for along in (0., -.35*radius)
                    for offset in (-100., -40., 40., 100.)]
        return np.asarray(original+additional_views(center, radius, a, b))
    first, reversed_axes = pool(u, v), pool(-u, -v)
    assert np.max(np.min(np.linalg.norm(first[:, None]-reversed_axes, axis=2), axis=1)) < 1e-10
    assert np.max(np.min(np.linalg.norm(reversed_axes[:, None]-first, axis=2), axis=1)) < 1e-10
    return dict(passed=True, property='diameter-endpoint reversal preserves the geometric pool', extra_views=16)


def hashes():
    result = {f'src/{k}': v for k, v in q4_share25_client.source_hashes().items()}
    for path in (Path(__file__), HERE/'continuous_hypothesis_experiment.py', ENVIRONMENT_PATH, VALIDATION_PATH):
        result[path.relative_to(PROJECT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--start-seed', type=int, default=46000)
    args = parser.parse_args()
    check = candidate_geometry_check()
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot = hashes()
    write(args.out/'manifest.json', dict(local_only=True, official_simulator_contacted=False,
        start_seed=args.start_seed, populations=POPULATIONS, fields=['smooth'], layouts_per_population=1,
        source_sha256=snapshot, geometric_check=check,
        fixed_before_evaluation=True, extra_transverse_scales=[.5, 1.],
        unchanged='Original score, beliefs, full optical fallback, 25 station route policy and completion certificate'))
    rows = []
    for pindex, population in enumerate(POPULATIONS):
        seed = args.start_seed+100*pindex
        sources = sources_for(seed, population)
        for variant in ('share25', 'scaled_views25'):
            arena, trace = LocalArena(sources, seed, 'smooth'), []
            result, error = None, None
            started = time.perf_counter()
            try:
                solve = task_sharing_solver.solve_multi if variant == 'share25' else make_solver()
                result = solve(public_api(arena), variant='share25', trace=trace)
            except Exception:
                error = traceback.format_exc()
            elapsed = time.perf_counter()-started
            try:
                if error:
                    raise RuntimeError(error)
                cover = PolarCover()
                audit = validate_run(sources, arena.events, trace, result, cover.stations, cover.indices)
            except Exception:
                audit = dict(passed=False, error=traceback.format_exc())
            row = dict(seed=seed, population=population, variant=variant, field='smooth',
                       audit=audit, runtime_s=elapsed, error=error, **arena.evaluation())
            rows.append(row)
            write(args.out/f'{seed}-{population}-{variant}.json', dict(summary=row, sources=sources,
                  events=arena.events, trace=trace, result=result))
            print(f'{seed} {population} {variant}: {row["total_s"]:.3f}s; '
                  f'{row["cleared"]}/{row["source_count"]}; audit={audit["passed"]}; wall={elapsed:.2f}s', flush=True)
            if not audit['passed']:
                write(args.out/'failed_rows.json', rows)
                raise RuntimeError(audit)
    pairs = [dict(seed=a['seed'], population=a['population'], baseline_s=a['total_s'],
                  candidate_s=b['total_s'], saved_s=a['total_s']-b['total_s'])
             for a, b in zip(rows[::2], rows[1::2])]
    before, after = sum(p['baseline_s'] for p in pairs), sum(p['candidate_s'] for p in pairs)
    stable = hashes() == snapshot
    summary = dict(passed=stable, source_snapshot_stable=stable, local_only=True,
        official_simulator_contacted=False, pairs=pairs, mean_baseline_s=before/len(pairs),
        mean_candidate_s=after/len(pairs), saved_pct=100*(before-after)/before,
        faster=sum(p['saved_s'] > 1e-7 for p in pairs), slower=sum(p['saved_s'] < -1e-7 for p in pairs),
        sources_per_variant=sum(r['source_count'] for r in rows[::2]))
    write(args.out/'rows.json', rows)
    write(args.out/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if stable else 1


if __name__ == '__main__':
    raise SystemExit(main())
