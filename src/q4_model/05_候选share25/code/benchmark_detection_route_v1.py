"""Fixed 49000-series local pairing of discovery-fee-aware routing."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import scipy
import benchmark_station22_v1 as support
import detection_latency_route_v1 as candidate


def math_checks():
    M = candidate.CLOUD_SIZE
    all_bits = (1 << M)-1
    h, weights = candidate.expected_remaining(0, M)
    assert abs(h-13) < 1e-12 and len(weights) == 7
    for count in (1, M//2, M):
        h15, _ = candidate.expected_remaining(15, count)
        w0 = count/M
        assert abs(h15-16*w0/(1+16*w0)) < 1e-12 and 0 <= h15 <= 1
    masks = [all_bits, 0]
    early = candidate.prefix_state([0,1], masks, all_bits)[1][-1]
    late = candidate.prefix_state([1,0], masks, all_bits)[1][-1]
    assert 6*(20-13)*2+6*13/M*early == 162
    assert 6*(20-13)*2+6*13/M*late == 240
    assert candidate.surviving_cloud(()) == all_bits
    return dict(passed=True, uniform_total_prior_mean=13,
        discovered_15_expected_remaining_bounded_by_one=True,
        two_stop_scan_fee_before_discovery_checks_s=[162,240],
        cloud_is_never_absence_evidence=True)


def hashes():
    result = support.hashes()
    for path in (Path(__file__), Path(candidate.__file__)):
        result[str(path.relative_to(support.PROJECT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--start-seed', type=int, default=49000)
    args = parser.parse_args()
    checks = math_checks()
    args.out.mkdir(parents=True, exist_ok=False)
    def save(name, obj):
        with (args.out/name).open('x', encoding='utf-8') as stream:
            json.dump(obj, stream, ensure_ascii=False, indent=2)
    snapshot = hashes()
    save('manifest.json', dict(local_only=True, official_contacted=False, source_hashes=snapshot,
        start_seed=args.start_seed, populations=['uniform','boundary','cluster','omni_heavy','dir_heavy'],
        field='smooth', candidate='discovery_latency_25', math_checks=checks,
        count_prior='uniform integers 10..16; uniform channel subsets conditional on count',
        source_prior='uniform disk position; uniform radius 1000..1500; types half each; uniform directional heading',
        cloud_size=candidate.CLOUD_SIZE, cloud_seed=candidate.CLOUD_SEED, cloud_sha256=candidate.cloud()[-1],
        start_tours=candidate.START_TOURS, two_opt_sweeps=candidate.TWO_OPT_SWEEPS,
        numpy_version=np.__version__, scipy_version=scipy.__version__, fixed_before_run=True))
    rows = []
    for index, population in enumerate(('uniform','boundary','cluster','omni_heavy','dir_heavy')):
        seed = args.start_seed+100*index
        sources = support.sources_for(seed, population)
        for variant, solve in (('baseline_share25', support.baseline), ('discovery_latency_25', candidate.solve_multi)):
            arena, trace = support.LocalArena(sources, seed, 'smooth'), []
            result, error, audit = None, None, dict(passed=False)
            started = time.perf_counter()
            try:
                result = solve(support.public_api(arena), variant='share25', trace=trace)
            except Exception as exc:
                error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
            elapsed = time.perf_counter()-started
            try:
                assert error is None, error
                mesh = support.PolarCover()
                audit = support.validate_run(sources, arena.events, trace, result, mesh.stations, mesh.indices)
                audit['exact_certificate'] = support.validate_exact_baseline(result, arena.events)
            except Exception as exc:
                audit = dict(passed=False, type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
            routes = [r for r in trace if r['phase'] == 'discovery_fee_route']
            row = dict(seed=seed, population=population, field='smooth', variant=variant,
                runtime_s=elapsed, error=error, audit=audit, **arena.evaluation(),
                route_diagnostics=dict(scored_states=len(routes),
                    order_changed=sum(r.get('applied', False) for r in routes),
                    first_task_changed=sum(r.get('first_task_changed', False) for r in routes),
                    cloud_empty_fallbacks=sum(r.get('reason') == 'scoring_cloud_empty_original_route_preserved' for r in routes),
                    two_opt_evaluations=sum(r.get('two_opt_evaluations',0) for r in routes)))
            save(f'{seed}-{population}-{variant}.json', dict(summary=row, sources=sources,
                result=result, events=arena.events, trace=trace))
            rows.append(row)
            print('CASE', seed, population, variant, 'time', round(row['total_s'],3),
                'cleared',row['cleared'],'/',row['source_count'], 'audit', audit['passed'],
                'runtime', round(elapsed,2), 'routes', row['route_diagnostics'], flush=True)
    stable = hashes() == snapshot
    passed = stable and all(r['audit']['passed'] and r['error'] is None for r in rows)
    a, b = rows[::2], rows[1::2]
    before, after = sum(r['total_s'] for r in a), sum(r['total_s'] for r in b)
    save('summary.json', dict(passed=passed, source_snapshot_stable=stable, rows=rows,
        local_only=True, official_contacted=False, source_hashes=snapshot,
        mean_baseline_s=before/5, mean_candidate_s=after/5, total_reduction_fraction=(before-after)/before,
        faster=sum(v['total_s']<u['total_s'] for u,v in zip(a,b))))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
