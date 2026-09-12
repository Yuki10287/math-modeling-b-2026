"""New-layout validation of the already-frozen 14+7+1 station candidate.

No geometry or solver parameter selection happens here. The development source
manifest must still match byte for byte before and after all requested runs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import benchmark_station22_v1 as frozen

POPULATIONS = ('uniform', 'boundary', 'cluster', 'omni_heavy', 'dir_heavy')
VARIANTS = ('baseline_share25', 'station22_share')
DEVELOPMENT = Path(__file__).resolve().parent.parent/'results/station22_share_dev37000_v1/manifest.json'


def run_case(seed, population, field, variant):
    sources = frozen.sources_for(seed, population)
    arena, trace = frozen.LocalArena(sources, seed, field), []
    result, error, audit = None, None, dict(passed=False)
    begin = time.perf_counter()
    try:
        solve = frozen.baseline if variant == VARIANTS[0] else frozen.candidate
        result = solve(frozen.public_api(arena), variant='share25', trace=trace)
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    elapsed = time.perf_counter()-begin
    try:
        assert error is None, error
        if variant == VARIANTS[0]:
            mesh = frozen.PolarCover()
            audit = frozen.validate_run(sources, arena.events, trace, result, mesh.stations, mesh.indices)
            audit['exact_certificate'] = frozen.validate_exact_baseline(result, arena.events)
        else:
            exact, adapted, vertices, triangles = frozen.validate_exact_candidate(result, arena.events, frozen.CERTIFICATE)
            audit = frozen.validate_run(sources, arena.events, trace, adapted, vertices, triangles)
            audit['exact_certificate'] = exact
    except Exception as exc:
        audit = dict(passed=False, type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    row = dict(seed=seed, population=population, variant=variant, field=field,
        runtime_s=elapsed, error=error, audit=audit, **arena.evaluation())
    return dict(summary=row, sources=sources, result=result, events=arena.events, trace=trace)


def comparison(rows):
    groups = {}
    for variant in VARIANTS:
        group = [r for r in rows if r['variant'] == variant]
        total = [r['total_s'] for r in group]
        groups[variant] = dict(runs=len(group), mean_total_s=float(np.mean(total)),
            p90_total_s=float(np.quantile(total, .90)), p95_total_s=float(np.quantile(total, .95)),
            maximum_total_s=max(total), source_count=sum(r['source_count'] for r in group),
            cleared=sum(r['cleared'] for r in group), passed=sum(r['audit']['passed'] for r in group),
            mean_runtime_s=float(np.mean([r['runtime_s'] for r in group])),
            time_parts_mean_s={key:float(np.mean([r['time_parts_s'][key] for r in group])) for key in group[0]['time_parts_s']})
    keyed = {(r['seed'], r['population'], r['field'], r['variant']):r for r in rows}
    pairs = []
    for key, a in keyed.items():
        if key[-1] != VARIANTS[0]:
            continue
        b = keyed[key[:-1]+(VARIANTS[1],)]
        pairs.append(dict(seed=key[0], population=key[1], field=key[2],
            baseline_s=a['total_s'], candidate_s=b['total_s'], saving_s=a['total_s']-b['total_s'],
            saving_fraction=(a['total_s']-b['total_s'])/a['total_s']))
    by_population = {}
    for name in POPULATIONS:
        group = [r for r in pairs if r['population'] == name]
        before, after = sum(r['baseline_s'] for r in group), sum(r['candidate_s'] for r in group)
        by_population[name] = dict(pairs=len(group), faster=sum(r['saving_s'] > 1e-7 for r in group),
            slower=sum(r['saving_s'] < -1e-7 for r in group), mean_saving_s=(before-after)/len(group),
            total_reduction_fraction=(before-after)/before)
    before, after = sum(r['baseline_s'] for r in pairs), sum(r['candidate_s'] for r in pairs)
    return dict(groups=groups, pairs=pairs, by_population=by_population,
        total_reduction_fraction=(before-after)/before, mean_saving_s=(before-after)/len(pairs),
        faster=sum(r['saving_s'] > 1e-7 for r in pairs), slower=sum(r['saving_s'] < -1e-7 for r in pairs),
        ties=sum(abs(r['saving_s']) <= 1e-7 for r in pairs), worst_pair=min(pairs, key=lambda r:r['saving_fraction']),
        independent_layouts=len({r['seed'] for r in pairs}), condition_pairs=len(pairs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--start-seed', type=int, required=True)
    parser.add_argument('--layouts', type=int, required=True)
    parser.add_argument('--fields', required=True)
    args = parser.parse_args()
    expected = json.loads(DEVELOPMENT.read_text())['source_hashes']
    assert frozen.hashes() == expected, 'development model or dependency hash changed'
    driver_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    fields = args.fields.split(',')
    assert set(fields) <= {'smooth', 'constant', 'extreme'} and len(fields) == len(set(fields))
    args.out.mkdir(parents=True, exist_ok=False)
    def save(name, obj):
        with (args.out/name).open('x', encoding='utf-8') as stream:
            json.dump(obj, stream, ensure_ascii=False, indent=2)
    save('manifest.json', dict(local_only=True, official_contacted=False, source_hashes=expected,
        driver_sha256=driver_hash, development_manifest_sha256=hashlib.sha256(DEVELOPMENT.read_bytes()).hexdigest(),
        start_seed=args.start_seed, layouts_per_population=args.layouts, fields=fields,
        populations=POPULATIONS, variants=VARIANTS, candidate_unchanged_from_development=True,
        quantile_method='numpy linear interpolation; descriptive finite-sample quantiles'))
    rows = []
    for population_index, population in enumerate(POPULATIONS):
        for layout in range(args.layouts):
            seed = args.start_seed+100*population_index+layout
            for field in fields:
                for variant in VARIANTS:
                    case = run_case(seed, population, field, variant)
                    row = case['summary']
                    save(f'{seed}-{population}-{field}-{variant}.json', case)
                    rows.append(row)
                    print('CASE', seed, population, field, variant, 'time', round(row['total_s'], 3),
                        'cleared', row['cleared'], '/', row['source_count'], 'audit', row['audit']['passed'], flush=True)
    stable = frozen.hashes() == expected and hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == driver_hash
    passed = stable and all(row['audit']['passed'] and row['error'] is None for row in rows)
    summary = dict(passed=passed, source_snapshot_stable=stable, local_only=True, official_contacted=False,
        candidate_unchanged_from_development=True, rows=rows, **comparison(rows))
    save('summary.json', summary)
    print('COMPLETE', 'passed', passed, 'reduction', summary['total_reduction_fraction'],
        'faster', summary['faster'], 'slower', summary['slower'], flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
