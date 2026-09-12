"""Local ablations: defer ambiguous service to the required scan scaffold.

This changes task eligibility and/or the maximum number of profitable shared
observations, not the source region, local policy, scan mesh, or completion
certificate. The frozen solver is compiled into a private namespace after two
explicit AST edits. It is never rewritten or monkey-patched in process.

defer_two: omit ambiguous source tasks while required scan stations remain.
defer_all: the same eligibility rule, plus all profitable shared observations.
share_all: original task eligibility, plus all profitable shared observations.

An ambiguous source has no near feedback and minimum enclosing radius greater
than the certified 20 m clearance threshold. Once required scans finish, every
source task is eligible again. No true source count, pose, or error is exposed.
"""
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
Q4 = HERE.parents[1]
PROJECT = Q4.parent.parent
sys.path.insert(0, str(Q4))

import task_sharing_solver
from polar_cover import PolarCover
from continuous_hypothesis_experiment import (
    LocalArena, public_api, sources_for, validate_run, ENVIRONMENT_PATH,
    VALIDATION_PATH, POPULATIONS,
)
import q4_share25_client

VARIANTS = ('share25', 'defer_two', 'defer_all', 'share_all')


def make_solver(variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    if variant == 'share25':
        return task_sharing_solver.solve_multi
    source = ast.parse(Path(task_sharing_solver.__file__).read_text(encoding='utf-8-sig'))
    function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'solve_multi')
    defer = variant in ('defer_two', 'defer_all')
    unlimited = variant in ('defer_all', 'share_all')
    eligibility_edits = sharing_edits = 0
    if defer:
        for node in ast.walk(function):
            if (isinstance(node, ast.For) and isinstance(node.iter, ast.Call)
                    and isinstance(node.iter.func, ast.Attribute)
                    and isinstance(node.iter.func.value, ast.Name)
                    and node.iter.func.value.id == 'beliefs'
                    and node.iter.func.attr == 'items'):
                node.body.insert(0, ast.parse(
                    'if pending and b.near is None and core.mec(b.P)[1] > 20-1e-6:\n'
                    '    continue\n').body[0])
                eligibility_edits += 1
        assert eligibility_edits == 1
    if unlimited:
        sharing = next(n for n in function.body if isinstance(n, ast.FunctionDef) and n.name == 'share_at_stop')
        for node in ast.walk(sharing):
            if isinstance(node, ast.Slice) and isinstance(node.upper, ast.Constant) and node.upper.value == 2:
                node.upper = None
                sharing_edits += 1
        assert sharing_edits == 1
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = task_sharing_solver.solve_multi.__globals__.copy()
    exec(compile(module, str(Path(__file__)), 'exec'), namespace)
    return namespace['solve_multi']


def hashes():
    result = {f'src/{k}': v for k, v in q4_share25_client.source_hashes().items()}
    for path in (Path(__file__), HERE/'continuous_hypothesis_experiment.py', ENVIRONMENT_PATH, VALIDATION_PATH):
        result[path.relative_to(PROJECT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def run_case(seed, population, field, variant):
    sources = sources_for(seed, population)
    arena, trace = LocalArena(sources, seed, field), []
    result, error = None, None
    started = time.perf_counter()
    try:
        result = make_solver(variant)(public_api(arena), variant='share25', trace=trace)
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
    row = dict(seed=seed, population=population, field=field, variant=variant,
               runtime_s=elapsed, error=error, audit=audit, **arena.evaluation())
    return dict(summary=row, sources=sources, events=arena.events, trace=trace, result=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--start-seed', type=int, default=36000)
    parser.add_argument('--layouts', type=int, default=1)
    parser.add_argument('--fields', default='smooth')
    parser.add_argument('--variants', default=','.join(VARIANTS))
    args = parser.parse_args()
    variants, fields = args.variants.split(','), args.fields.split(',')
    assert len(set(variants)) == len(variants) and set(variants) <= set(VARIANTS)
    assert variants[0] == 'share25' and args.layouts > 0
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot = hashes()
    write(args.out/'manifest.json', dict(
        start_seed=args.start_seed, layouts_per_population=args.layouts,
        populations=POPULATIONS, fields=fields, variants=variants,
        source_sha256=snapshot, local_only=True, official_simulator_contacted=False,
        purpose='Predeclared task-eligibility and sharing ablations; paired local evaluation.',
        clearance_radius_m=20., unchanged='25 stations, source beliefs, local sensing, optical cover, stopping proof'))
    rows = []
    for pindex, population in enumerate(POPULATIONS):
        for layout in range(args.layouts):
            seed = args.start_seed+100*pindex+layout
            for field in fields:
                for variant in variants:
                    case = run_case(seed, population, field, variant)
                    row = case['summary']
                    rows.append(row)
                    write(args.out/f'{seed}-{population}-{field}-{variant}.json', case)
                    print(f'{seed} {population} {field} {variant}: {row["total_s"]:.3f}s; '
                          f'{row["cleared"]}/{row["source_count"]}; audit={row["audit"]["passed"]}; '
                          f'wall={row["runtime_s"]:.2f}s', flush=True)
                    if not row['audit']['passed']:
                        write(args.out/'failed_rows.json', rows)
                        raise RuntimeError(row['audit'])
    pairs = []
    for row in rows:
        if row['variant'] == 'share25':
            continue
        baseline = next(r for r in rows if r['variant'] == 'share25' and
                        (r['seed'], r['population'], r['field']) == (row['seed'], row['population'], row['field']))
        pairs.append(dict(seed=row['seed'], population=row['population'], field=row['field'],
                          variant=row['variant'], baseline_s=baseline['total_s'], candidate_s=row['total_s'],
                          saved_s=baseline['total_s']-row['total_s']))
    groups = {}
    for variant in variants:
        group = [r for r in rows if r['variant'] == variant]
        paired = [r for r in pairs if r['variant'] == variant]
        groups[variant] = dict(runs=len(group), cleared=sum(r['cleared'] for r in group),
            sources=sum(r['source_count'] for r in group), mean_total_s=float(np.mean([r['total_s'] for r in group])),
            mean_runtime_s=float(np.mean([r['runtime_s'] for r in group])),
            faster=sum(r['saved_s'] > 1e-7 for r in paired), slower=sum(r['saved_s'] < -1e-7 for r in paired))
        if paired:
            groups[variant]['saved_pct'] = 100*sum(r['saved_s'] for r in paired)/sum(r['baseline_s'] for r in paired)
    stable = hashes() == snapshot
    write(args.out/'rows.json', rows)
    write(args.out/'summary.json', dict(passed=stable and all(r['audit']['passed'] for r in rows),
        source_snapshot_stable=stable, local_only=True, official_simulator_contacted=False, pairs=pairs, groups=groups))
    print(json.dumps(groups, ensure_ascii=False, indent=2), flush=True)
    return 0 if stable else 1


if __name__ == '__main__':
    raise SystemExit(main())
