"""Second development stage: isolate route scheduling from coverage geometry."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
from benchmark import LocalArena, public_api
from benchmark_cells import hashes as cell_hashes, sources_for, run_case as cell_case
from polar_cover import PolarCover
from validation import validate_run
from task_sharing_solver import solve_multi
from shared import ROOT


def hashes():
    result = cell_hashes()
    for name in ('task_sharing_solver.py', 'benchmark_task_sharing.py'):
        path = Path(__file__).with_name(name)
        result[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def run_case(seed, population, field, variant):
    if variant in ('baseline25', 'context', 'interleave'):
        return cell_case(seed, population, field, variant)
    sources = sources_for(seed, population)
    arena, trace = LocalArena(sources, seed, field), []
    result, started = None, time.perf_counter()
    try:
        result = solve_multi(public_api(arena), variant, trace)
        runtime = time.perf_counter()-started
        mesh = PolarCover()
        audit = validate_run(sources, arena.events, trace, result, mesh.stations, mesh.indices)
    except Exception as exc:
        runtime = time.perf_counter()-started
        audit = dict(passed=False, error_type=type(exc).__name__, message=str(exc))
    row = dict(seed=seed, population=population, field=field, variant=variant,
               runtime_s=runtime, audit=audit, **arena.evaluation())
    return dict(summary=row, sources=sources, result=result, events=arena.events, trace=trace)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--start-seed', type=int, required=True)
    parser.add_argument('--layouts', type=int, default=1)
    parser.add_argument('--populations', default='uniform,boundary,cluster,omni_heavy,dir_heavy')
    parser.add_argument('--fields', default='smooth')
    parser.add_argument('--variants', default='baseline25,context,resume25,share25')
    args = parser.parse_args()
    variants = args.variants.split(',')
    assert set(variants) <= {'baseline25', 'context', 'interleave', 'resume25', 'share25'}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    snapshot, rows = hashes(), []
    (out/'manifest.json').write_text(json.dumps(dict(arguments=vars(args), code_hashes=snapshot,
        python=platform.python_version(), numpy=np.__version__, local_only=True,
        official_simulator_used=False, paired_layouts=True,
        runtime_scope='solver only, excludes audit and serialization'), indent=2), encoding='utf-8')
    for pindex, population in enumerate(args.populations.split(',')):
        for layout in range(args.layouts):
            seed = args.start_seed+100*pindex+layout
            for field in args.fields.split(','):
                for variant in variants:
                    case = run_case(seed, population, field, variant)
                    row = case['summary']; rows.append(row)
                    (out/f'{seed}-{population}-{field}-{variant}.json').write_text(json.dumps(case), encoding='utf-8')
                    (out/'rows.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
                    print(f'{seed} {population} {field} {variant}: {row["total_s"]:.2f}s; '
                          f'{row["cleared"]}/{row["source_count"]}; runtime {row["runtime_s"]:.2f}s; '
                          f'audit={row["audit"]["passed"]}', flush=True)
                    if not row['audit']['passed']: print(row['audit'], flush=True)
    groups = {}
    for variant in variants:
        rr = [r for r in rows if r['variant'] == variant]
        groups[variant] = dict(runs=len(rr), passed=sum(r['audit']['passed'] for r in rr),
            all_cleared=sum(r['all_cleared'] for r in rr),
            mean_total_s=float(np.mean([r['total_s'] for r in rr])),
            pooled_per_source_s=sum(r['total_s'] for r in rr)/sum(r['source_count'] for r in rr),
            mean_runtime_s=float(np.mean([r['runtime_s'] for r in rr])),
            max_runtime_s=max(r['runtime_s'] for r in rr))
    stable = hashes() == snapshot
    summary = dict(groups=groups, source_snapshot_stable=stable, local_only=True,
                   independent_layouts=args.layouts*len(args.populations.split(',')))
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)
    raise SystemExit(0 if stable and all(r['audit']['passed'] for r in rows) else 1)


if __name__ == '__main__': main()
