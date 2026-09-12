"""New paired local cases with frozen policy rollouts and independent audits."""
import argparse
import hashlib
import json
import platform
import time
import traceback
from pathlib import Path
import numpy as np

from bootstrap import ROOT, Q3, PREVIOUS, baseline
from benchmark import multi_case
from benchmark_joint import sparse_case, summary
from validate_model import (BoundedArena, PublicAPI, independent_cells, independent_time_audit,
    observation_audit, certificate_audit, trace_truth_audit, stress_cases)
from rollout_solver import solve_multi
from closed_preview import ClosedLoopSelector


def hashes():
    paths = [Q3/name for name in ('baseline_solver.py', 'v1_solver.py', 'solver.py', 'geometry.py',
             'belief_model.py', 'coverage_model.py', 'scan_planning.py', 'local_policy.py',
             'recovery.py', 'environment.py', 'benchmark.py', 'benchmark_joint.py', 'validate_model.py')]
    paths += [PREVIOUS/'scan_preview.py']
    paths += [Path(__file__).with_name(name) for name in (
        'bootstrap.py', 'rollout_solver.py', 'scenario_model.py', 'closed_preview.py', 'benchmark_closed.py')]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run_case(case, variant, centers, cache, scenario_count=6):
    arena = BoundedArena(case['sources'], case['seed'], case['field'], max_actions=3000)
    trace, outcome, checks, error = [], {}, {}, None
    started = time.perf_counter()
    try:
        if variant == 'lean':
            outcome = baseline.solve_multi(PublicAPI(arena), trace=trace)
        else:
            selector = ClosedLoopSelector(scenario_count=scenario_count, cache=cache,
                risk=0. if variant == 'closed_mean' else .2) if variant.startswith('closed') else None
            outcome = solve_multi(PublicAPI(arena), variant='closed' if variant.startswith('closed') else variant,
                                  selector=selector, trace=trace)
        runtime = time.perf_counter()-started
        assert outcome['complete'] and arena.evaluation()['all_cleared'], outcome
        observations = observation_audit(arena)
        checks = dict(timing=independent_time_audit(arena),
                      certificate=certificate_audit(arena, outcome, observations, centers),
                      geometry=trace_truth_audit(arena, trace))
    except Exception:
        runtime = time.perf_counter()-started
        error = traceback.format_exc()
    forecasts = [r for r in trace if r.get('phase') == 'closed_scan_forecast']
    last = max((e['time_s'] for e in arena.events if e.get('clear_result') == 'success'), default=0.)
    row = dict(name=case['name'], seed=case['seed'], field=case['field'], schedule=variant,
        passed=error is None, error=error, runtime_s=runtime, tail_s=arena.time_s-last,
        forecasts=len(forecasts), changed=sum(r['changed'] for r in forecasts),
        cache_hits=sum(r['cached'] for r in forecasts), **arena.evaluation())
    return dict(result=row, sources=case['sources'], outcome=outcome,
                independent_checks=checks, events=arena.events, trace=trace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--start-seed', type=int, default=24000)
    parser.add_argument('--layouts', type=int, default=1)
    parser.add_argument('--populations', default='uniform,sparse10,boundary10,one_side10')
    parser.add_argument('--fields', default='smooth')
    parser.add_argument('--variants', default='lean,reference,legacy,closed,closed_mean')
    parser.add_argument('--scenarios', type=int, choices=(6,12), default=6)
    parser.add_argument('--split', choices=('development','holdout','stress'), default='development')
    args = parser.parse_args()
    variants = args.variants.split(',')
    assert variants[0] == 'lean' and set(variants) <= {'lean','reference','legacy','closed','closed_mean'}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'cases').mkdir()
    before = hashes()
    cases = stress_cases() if args.populations == 'stress' else [
        dict(name=f'{seed}-{pop}-{field}', seed=seed, field=field,
             sources=multi_case(seed) if pop == 'uniform' else sparse_case(seed, pop))
        for pi, pop in enumerate(args.populations.split(','))
        for seed in range(args.start_seed+100*pi, args.start_seed+100*pi+args.layouts)
        for field in args.fields.split(',')]
    manifest = dict(arguments={**vars(args),'out':str(args.out)}, cases=cases, source_sha256=before,
                    python=platform.python_version(), numpy=np.__version__, local_only=True,
                    official_simulator_used=False, truth_hidden=True,
                    note='Self-defined layouts; shared forecast cache is keyed only by public model state.')
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    rows, cache, centers = [], {}, independent_cells()
    for case in cases:
        reference = None
        for variant in variants:
            print(json.dumps(dict(starting=case['name'], variant=variant)), flush=True)
            detail = run_case(case, variant, centers, cache, args.scenarios)
            row = detail['result']
            if variant == 'lean':
                reference = detail['events']
            if variant == 'reference':
                assert detail['events'] == reference, 'reference is not event-equivalent to lean'
            rows.append(row)
            (args.out/'cases'/f'{case["name"]}-{variant}.json').write_text(json.dumps(detail), encoding='utf-8')
            (args.out/'rows.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
            print(json.dumps({k: row[k] for k in ('name','schedule','passed','total_s','runtime_s','forecasts','changed','cache_hits')}), flush=True)
    stable = before == hashes()
    result = dict(**summary(rows), source_snapshot_stable=stable, cached_forecast_states=len(cache))
    (args.out/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if stable and all(r['passed'] for r in rows) else 1)


if __name__ == '__main__':
    main()
