"""Global-value ablation with fixed local costs and frozen joint scheduling engine."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
from benchmark import multi_case
from solver import solve_multi as lean_solver
from value_solver import solve_multi as joint_solver
from validate_model import (BoundedArena, PublicAPI, independent_cells, independent_time_audit,
    observation_audit, certificate_audit, stress_cases)
from value_validation import trace_truth_audit


def hashes():
    root = Path(__file__).parent
    names = ['solver.py', 'joint_solver.py', 'joint_planning.py', 'flexible_coverage.py', 'geometry.py',
             'local_policy.py', 'coverage_model.py', 'belief_model.py', 'scan_planning.py',
             'v1_solver.py', 'baseline_solver.py', 'recovery.py', 'environment.py',
             'validate_model.py', 'benchmark.py', 'benchmark_joint.py',
             'forecast_solver.py', 'forecast_policy.py', 'benchmark_forecast.py',
             'value_solver.py', 'global_value_policy.py', 'value_validation.py',
             'audit_forecast_results.py', 'benchmark_value.py']
    return {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in names}


def sparse_case(seed, kind):
    rng = np.random.default_rng(seed)
    channels = rng.choice(np.arange(1, 21), 10, replace=False)
    angle = rng.uniform(0, 2*np.pi, 10)
    if kind == 'boundary10':
        radius = rng.uniform(1650, 1800, 10)
    elif kind == 'one_side10':
        radius = 1800*np.sqrt(rng.uniform(.05, 1, 10))
        angle = rng.uniform(-.4, .4, 10)+rng.uniform(0, 2*np.pi)
    else:
        radius = 1800*np.sqrt(rng.uniform(0, 1, 10))
    return [dict(channel=int(c), position=[float(r*np.cos(a)), float(r*np.sin(a))],
                 radius=1000.) for c, r, a in zip(channels, radius, angle)]


def summary(rows):
    groups = {}
    for schedule in sorted({r['schedule'] for r in rows}):
        rs = [r for r in rows if r['schedule'] == schedule]
        groups[schedule] = dict(runs=len(rs), passed=sum(r['passed'] for r in rs),
            sources=sum(r['source_count'] for r in rs),
            mean_total_s=float(np.mean([r['total_s'] for r in rs])),
            mean_move_s=float(np.mean([r['time_parts_s']['move'] for r in rs])),
            mean_measures=float(np.mean([r['counts']['measure'] for r in rs])),
            mean_tail_s=float(np.mean([r['tail_s'] for r in rs])),
            max_total_s=max(r['total_s'] for r in rs),
            mean_runtime_s=float(np.mean([r['runtime_s'] for r in rs])),
            max_runtime_s=max(r['runtime_s'] for r in rs))
    base = {r['name']: r for r in rows if r['schedule'] == 'original'}
    comparisons = {}
    for schedule in groups:
        if schedule == 'original':
            continue
        rs = [r for r in rows if r['schedule'] == schedule]
        pairs = [(base[r['name']], r) for r in rs]
        ds = np.array([a['total_s']-b['total_s'] for a,b in pairs])
        comparisons[schedule] = dict(pairs=len(pairs), faster=int(sum(ds>1e-6)),
            equal=int(sum(abs(ds)<=1e-6)), slower=int(sum(ds < -1e-6)),
            mean_saved_s=float(ds.mean()), reduction_pct=float(100*ds.sum()/sum(a['total_s'] for a,b in pairs)),
            worst_regression_s=max(0., -float(ds.min())),
            worst_regression_case=pairs[int(np.argmin(ds))][0]['name'],
            largest_saving_s=float(ds.max()))
    return dict(groups=groups, comparisons_to_original=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-seed', type=int, default=7000)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--fields', default='smooth,hash,extreme')
    parser.add_argument('--schedules', default='original,route,coverage,waypoints,shared')
    parser.add_argument('--population', choices=['uniform','sparse10','boundary10','one_side10','stress'], default='uniform')
    parser.add_argument('--split', choices=['development','holdout','stress'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'cases').mkdir()
    before = hashes()
    manifest = dict(local_only=True, official_simulator_used=False, truth_hidden=True,
        start_seed=args.start_seed, seeds=args.seeds, fields=args.fields.split(','),
        schedules=args.schedules.split(','), population=args.population, split=args.split,
        layout_distribution_is_self_defined=True, source_sha256=before)
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    cases = stress_cases() if args.population == 'stress' else [
        dict(name=f'{seed}-{args.population}-{field}', seed=seed, field=field,
             sources=multi_case(seed) if args.population=='uniform' else sparse_case(seed,args.population))
        for seed in range(args.start_seed,args.start_seed+args.seeds) for field in args.fields.split(',')]
    rows, centers = [], independent_cells()
    for case in cases:
        for schedule in args.schedules.split(','):
            arena = BoundedArena(case['sources'], case['seed'], case['field'], max_actions=3000)
            trace, outcome, error, checks = [], {}, None, {}
            t = time.perf_counter()
            try:
                solve = lean_solver if schedule == 'lean' else joint_solver
                outcome = solve(PublicAPI(arena), trace=trace, schedule=schedule)
                runtime = time.perf_counter()-t
                assert outcome['complete'] and arena.evaluation()['all_cleared'], outcome
                timing = independent_time_audit(arena)
                observations = observation_audit(arena)
                checks = dict(timing=timing,
                    certificate=certificate_audit(arena,outcome,observations,centers),
                    geometry=trace_truth_audit(arena,trace))
            except Exception:
                runtime = time.perf_counter()-t
                error = traceback.format_exc()
            last = max((e['time_s'] for e in arena.events if e.get('clear_result')=='success'),default=0.)
            row = dict(name=case['name'], seed=case['seed'], field=case['field'], schedule=schedule,
                passed=error is None, error=error, runtime_s=runtime,
                tail_s=arena.time_s-last, **arena.evaluation())
            rows.append(row)
            detail = dict(result=row, sources=case['sources'], outcome=outcome,
                independent_checks=checks, events=arena.events, trace=trace)
            (args.out/'cases'/f"{case['name']}-{schedule}.json").write_text(json.dumps(detail),encoding='utf-8')
            (args.out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
            print(json.dumps({k: row[k] for k in ('name','schedule','passed','total_s','runtime_s')}),flush=True)
    result = dict(**summary(rows), source_snapshot_stable=before==hashes())
    (args.out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
    raise SystemExit(0 if all(r['passed'] for r in rows) and result['source_snapshot_stable'] else 1)


if __name__ == '__main__':
    main()

