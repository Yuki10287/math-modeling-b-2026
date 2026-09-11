"""同布局、同固定地点误差场的配对实验。真值只在环境与事后审计中使用。"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from environment import LocalArena, audit_trace
from solver import solve_multi


def multi_case(seed):
    # 与原实验的生成顺序完全一致，便于对照旧种子；分布不是题目给定条件。
    rng = np.random.default_rng(seed)
    n = int(rng.integers(10, 17))
    channels = rng.choice(np.arange(1, 21), n, replace=False)
    sources = []
    for channel in channels:
        r, a = 1800 * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        sources.append(dict(channel=int(channel), position=[float(r*np.cos(a)), float(r*np.sin(a))],
                            radius=float(rng.uniform(1000, 1500))))
    return sources


def code_hashes():
    root = Path(__file__).parent
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ['baseline_solver.py', 'v1_solver.py', 'solver.py', 'geometry.py',
                         'belief_model.py', 'coverage_model.py', 'scan_planning.py', 'local_policy.py',
                         'recovery.py', 'environment.py', 'benchmark.py']}


def independent_event_check(events):
    position = np.zeros(2)
    channel, total = 1, 0.
    largest = 0.
    for event in events:
        q = np.array(event['position'])
        total += float(np.linalg.norm(q-position))/5
        if event['action'] == 'measure':
            total += 5 + (event['channel'] != channel)
            channel = event['channel']
        else:
            total += 5 if event['clear_result'] == 'success' else 3
        position = q
        largest = max(largest, abs(total-event['time_s']))
    return largest


def run_case(seed, field, schedule, policy='time', max_local_steps=30):
    sources = multi_case(seed)
    arena = LocalArena(sources, seed, field)
    trace, outcome, error = [], {}, None
    start = time.perf_counter()
    try:
        outcome = solve_multi(arena, policy=policy, schedule=schedule, trace=trace,
                              max_local_steps=max_local_steps)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    runtime = time.perf_counter() - start
    audit_error = independent_event_check(arena.events)
    violations = audit_trace(arena, trace)
    metrics = arena.evaluation()
    last_clear = max((e['time_s'] for e in arena.events
                      if e['action'] == 'clear' and e['clear_result'] == 'success'), default=0.)
    result = dict(seed=seed, field=field, schedule=schedule, policy=policy,
                  status=outcome.get('status', 'error'), error=error,
                  certificate_valid=outcome.get('complete', False), runtime_s=runtime,
                  belief_violations=violations, max_event_time_error_s=audit_error,
                  fallback_count=sum(r.get('phase') == 'fallback_start' for r in trace),
                  time_after_last_clear_s=metrics['total_s']-last_clear, **metrics)
    return result, dict(result=result, sources=sources, outcome=outcome, events=arena.events, trace=trace)


def summarize(rows):
    groups = {}
    for schedule in sorted({r['schedule'] for r in rows}):
        selected = [r for r in rows if r['schedule'] == schedule]
        times = np.array([r['total_s'] for r in selected])
        groups[schedule] = dict(
            runs=len(selected), source_instances=sum(r['source_count'] for r in selected),
            all_cleared=sum(r['all_cleared'] for r in selected),
            certified_complete=sum(r['certificate_valid'] for r in selected),
            errors=sum(r['error'] is not None for r in selected),
            belief_violation_runs=sum(bool(r['belief_violations']) for r in selected),
            mean_total_s=float(times.mean()), median_total_s=float(np.median(times)),
            p90_total_s=float(np.quantile(times, .9)), max_total_s=float(times.max()),
            mean_per_source_s=float(np.mean([r['average_s'] for r in selected])),
            pooled_per_source_s=float(sum(times)/sum(r['cleared'] for r in selected)),
            mean_distance_m=float(np.mean([r['distance_m'] for r in selected])),
            mean_measures=float(np.mean([r['counts']['measure'] for r in selected])),
            mean_switches=float(np.mean([r['counts']['switch'] for r in selected])),
            mean_scan_tail_s=float(np.mean([r['time_after_last_clear_s'] for r in selected])),
            max_runtime_s=max(r['runtime_s'] for r in selected),
            fallback_count=sum(r['fallback_count'] for r in selected),
            clear_failures=sum(r['counts']['clear_fail'] for r in selected),
        )
    base = {(r['seed'], r['field']): r for r in rows if r['schedule'] == 'v1'}
    pairs = {}
    for schedule in groups:
        if schedule == 'v1':
            continue
        paired = [(base[(r['seed'], r['field'])], r) for r in rows
                  if r['schedule'] == schedule and (r['seed'], r['field']) in base]
        if not paired:
            continue
        deltas = np.array([b['total_s']-r['total_s'] for b, r in paired])
        pairs[schedule] = dict(
            conditions=len(paired), faster=int(np.sum(deltas > 1e-6)),
            equal=int(np.sum(np.abs(deltas) <= 1e-6)), slower=int(np.sum(deltas < -1e-6)),
            mean_saved_s=float(deltas.mean()),
            aggregate_reduction_pct=float(100*sum(deltas)/sum(b['total_s'] for b, r in paired)),
            mean_paired_reduction_pct=float(np.mean([100*(b['total_s']-r['total_s'])/b['total_s'] for b,r in paired])),
            worst_regression_s=float(max(0, -deltas.min())),
            paired_successes=sum(b['all_cleared'] and r['all_cleared'] for b,r in paired),
        )
    return dict(groups=groups, comparisons_to_v1=pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-seed', type=int, required=True)
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--fields', default='smooth,hash,extreme')
    parser.add_argument('--schedules', default='v1,coverage,shared,full')
    parser.add_argument('--policy', default='time', choices=['time', 'geometry', 'midpoint'])
    parser.add_argument('--split', required=True, choices=['development', 'holdout'])
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out/'cases').mkdir()
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), split=args.split,
                    first_seed=args.start_seed, last_seed=args.start_seed+args.seeds-1,
                    independent_layouts=args.seeds, fields=args.fields.split(','),
                    schedules=args.schedules.split(','), policy=args.policy, code_hashes=code_hashes(),
                    numpy=np.__version__, python=platform.python_version(),
                    official_simulator=False,
                    layout_model='area-uniform disk, uniform reception radii, fixed spatial error fields; self-defined')
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    rows, equality = [], []
    for seed in range(args.start_seed, args.start_seed+args.seeds):
        for field in args.fields.split(','):
            reference = None
            for schedule in args.schedules.split(','):
                result, detail = run_case(seed, field, schedule, args.policy)
                rows.append(result)
                if schedule == 'baseline':
                    reference = detail['events']
                elif schedule == 'safe' and reference is not None:
                    equality.append(dict(seed=seed, field=field, equal=reference == detail['events']))
                (out/'cases'/f'{seed}-{field}-{schedule}.json').write_text(
                    json.dumps(detail, ensure_ascii=False), encoding='utf-8')
                with (out/'progress.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False)+'\n')
        print(json.dumps(dict(seed=seed, completed_runs=len(rows),
                              failures=sum(not r['all_cleared'] or not r['certificate_valid'] for r in rows)),
                         ensure_ascii=False), flush=True)
    summary = {**summarize(rows), 'safe_events_equal_baseline': equality,
               'code_hashes_unchanged': manifest['code_hashes'] == code_hashes()}
    (out/'rows.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
