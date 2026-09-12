"""Paired local experiments for the 22-station research candidates."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
from benchmark import LocalArena, public_api, make_case
from benchmark_joint import SOURCE_FILES as OLD_FILES
from joint_solver import solve_multi as baseline
from cell_solver import solve_multi as candidate, VARIANTS
from cell_cover import CERTIFICATE_PATH
from cell_validation import validate_cell_run
from polar_cover import PolarCover
from validation import validate_run
from shared import ROOT


def hashes():
    folder = Path(__file__).parent
    names = set(OLD_FILES) | {'cell_cover.py', 'cell_solver.py', 'cell_validation.py',
                             'context_policy.py', 'benchmark_cells.py'}
    paths = [folder/n for n in sorted(names)] + [
        ROOT/'q3_model_v2/geometry.py', ROOT/'q3_model_v2/environment.py',
        CERTIFICATE_PATH, CERTIFICATE_PATH.with_name('independent_exact_audit.py')]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def sources_for(seed, population):
    if population not in ('uniform', 'boundary', 'cluster', 'omni_heavy', 'dir_heavy'):
        raise ValueError('unknown population')
    sources = make_case(seed, population)
    if population in ('omni_heavy', 'dir_heavy'):
        rng = np.random.default_rng(seed+1000000)
        minority = int(rng.integers(len(sources)))
        for k, source in enumerate(sources):
            directional = k == minority if population == 'omni_heavy' else k != minority
            source['orientation'] = float(rng.uniform(0, 2*np.pi)) if directional else None
    return sources


def run_case(seed, population, field, variant):
    sources = sources_for(seed, population)
    arena, trace = LocalArena(sources, seed, field), []
    result = None
    started = time.perf_counter()
    try:
        result = (baseline(public_api(arena), 'shared', trace) if variant == 'baseline25'
                  else candidate(public_api(arena), variant, trace))
        runtime = time.perf_counter()-started
        if variant == 'baseline25':
            mesh = PolarCover()
            audit = validate_run(sources, arena.events, trace, result, mesh.stations, mesh.indices)
        else:
            audit = validate_cell_run(sources, arena.events, trace, result)
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
    parser.add_argument('--variants', default='baseline25,cells,context,interleave,opscan')
    args = parser.parse_args()
    variants = args.variants.split(',')
    if not set(variants) <= {'baseline25', *VARIANTS}:
        raise ValueError('unknown variant')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    snapshot, rows = hashes(), []
    manifest = dict(arguments=vars(args), code_hashes=snapshot,
        python=platform.python_version(), numpy=np.__version__, local_only=True,
        official_simulator_used=False, paired_layouts=True,
        population_counts='random integers 10 through 16; no official source counts used',
        runtime_scope='solver only, excludes independent audit and serialization')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for pindex, population in enumerate(args.populations.split(',')):
        for layout in range(args.layouts):
            seed = args.start_seed + 100*pindex + layout
            for field in args.fields.split(','):
                for variant in variants:
                    case = run_case(seed, population, field, variant)
                    row = case['summary']
                    rows.append(row)
                    (out/f'{seed}-{population}-{field}-{variant}.json').write_text(
                        json.dumps(case, ensure_ascii=False), encoding='utf-8')
                    (out/'rows.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
                    print(f'{seed} {population} {field} {variant}: {row["total_s"]:.2f}s; '
                          f'{row["cleared"]}/{row["source_count"]}; runtime {row["runtime_s"]:.2f}s; '
                          f'audit={row["audit"]["passed"]}', flush=True)
                    if not row['audit']['passed']:
                        print(row['audit'], flush=True)
    stable = hashes() == snapshot
    groups = {}
    for variant in variants:
        rr = [r for r in rows if r['variant'] == variant]
        groups[variant] = dict(runs=len(rr), passed=sum(r['audit']['passed'] for r in rr),
            all_cleared=sum(r['all_cleared'] for r in rr),
            mean_total_s=float(np.mean([r['total_s'] for r in rr])),
            pooled_per_source_s=sum(r['total_s'] for r in rr)/sum(r['source_count'] for r in rr),
            mean_runtime_s=float(np.mean([r['runtime_s'] for r in rr])),
            max_runtime_s=max(r['runtime_s'] for r in rr))
    summary = dict(groups=groups, source_snapshot_stable=stable, local_only=True,
                   independent_layouts=args.layouts*len(args.populations.split(',')))
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)
    raise SystemExit(0 if stable and all(r['audit']['passed'] for r in rows) else 1)


if __name__ == '__main__':
    main()
