"""Ten new layouts for the frozen 49000 discovery-delay route model."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import benchmark_detection_route_v1 as frozen

BASE = frozen.support
CANDIDATE = frozen.candidate
POPULATIONS = ('uniform','boundary','cluster','omni_heavy','dir_heavy')
VARIANTS = ('baseline_share25','discovery_latency_25')
DEVELOPMENT = Path(__file__).resolve().parent.parent/'results/discovery_latency_dev49000_v1/manifest.json'


def posterior_audit(trace):
    checked, empty = 0, 0
    for row in trace:
        if row['phase'] != 'discovery_fee_route':
            continue
        if row.get('reason') == 'scoring_cloud_empty_original_route_preserved':
            empty += 1
            assert not row['applied']
            continue
        D, U = row['discovered'], row['unknown']
        assert U == 20-D
        w0 = row['surviving_hypotheses']/CANDIDATE.CLOUD_SIZE
        weights = {n:math.comb(n,D)*w0**(n-D) for n in range(max(10,D),17)}
        norm = sum(weights.values())
        expected = sum((n-D)*w for n,w in weights.items())/norm
        assert abs(expected-row['expected_remaining_sources']) < 1e-10
        assert expected <= 16-D+1e-10
        assert set(row['total_count_posterior']) == {str(n) for n in weights}
        assert all(abs(row['total_count_posterior'][str(n)]-w/norm) < 1e-10 for n,w in weights.items())
        assert row['selected_proxy_s'] <= row['original_proxy_s']+1e-7
        checked += 1
    return dict(passed=True, posterior_states=checked, cloud_empty_fallbacks=empty)


def run(seed, population, variant):
    sources = BASE.sources_for(seed,population)
    arena, trace = BASE.LocalArena(sources,seed,'smooth'), []
    result, error, audit = None, None, dict(passed=False)
    start = time.perf_counter()
    try:
        solver = BASE.baseline if variant == VARIANTS[0] else CANDIDATE.solve_multi
        result = solver(BASE.public_api(arena),variant='share25',trace=trace)
    except Exception as exc:
        error = dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    runtime = time.perf_counter()-start
    try:
        assert error is None,error
        mesh = BASE.PolarCover()
        audit = BASE.validate_run(sources,arena.events,trace,result,mesh.stations,mesh.indices)
        audit['exact_certificate'] = BASE.validate_exact_baseline(result,arena.events)
        audit['prior_and_route_proxy'] = posterior_audit(trace)
    except Exception as exc:
        audit = dict(passed=False,type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    routes = [r for r in trace if r['phase']=='discovery_fee_route']
    row = dict(seed=seed,population=population,variant=variant,field='smooth',runtime_s=runtime,
        error=error,audit=audit,**arena.evaluation(),route_diagnostics=dict(
            scored_states=len(routes),order_changed=sum(r.get('applied',False) for r in routes),
            first_task_changed=sum(r.get('first_task_changed',False) for r in routes),
            cloud_empty_fallbacks=sum(r.get('reason')=='scoring_cloud_empty_original_route_preserved' for r in routes),
            two_opt_evaluations=sum(r.get('two_opt_evaluations',0) for r in routes)))
    return dict(summary=row,sources=sources,result=result,events=arena.events,trace=trace)


def decomposition(case):
    groups = defaultdict(lambda:defaultdict(float))
    previous, channel = np.zeros(2),1
    actions = [r for r in case['trace'] if r['phase']=='actual_action']
    for event, action in zip(case['events'],actions):
        q,c = np.asarray(event['position']),event['channel']
        bucket = groups[action['reason']]
        bucket['move_s'] += float(np.linalg.norm(q-previous))/5
        if event['action']=='measure':
            bucket['measure_s'] += 5
            bucket['switch_s'] += int(channel!=c)
            channel = c
        else:
            key = 'successful_clear_s' if event['clear_result']=='success' else 'failed_clear_s'
            bucket[key] += 5 if event['clear_result']=='success' else 3
        previous = q
    assert abs(sum(sum(v.values()) for v in groups.values())-case['summary']['total_s']) < 1e-6
    return {key:dict(values,total_s=sum(values.values())) for key,values in groups.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--start-seed',type=int,default=52000)
    args = parser.parse_args()
    manifest = json.loads(DEVELOPMENT.read_text())
    snapshot = manifest['source_hashes']
    assert frozen.hashes()==snapshot
    assert CANDIDATE.cloud()[-1]==manifest['cloud_sha256']
    own_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.mkdir(parents=True,exist_ok=False)
    def save(name,obj):
        with (args.out/name).open('x',encoding='utf-8') as stream:
            json.dump(obj,stream,ensure_ascii=False,indent=2)
    save('manifest.json',dict(local_only=True,official_contacted=False,source_hashes=snapshot,
        driver_sha256=own_hash,development_manifest_sha256=hashlib.sha256(DEVELOPMENT.read_bytes()).hexdigest(),
        start_seed=args.start_seed,layouts_per_population=2,populations=POPULATIONS,field='smooth',
        model_and_parameters_unchanged=True,cloud_sha256=CANDIDATE.cloud()[-1]))
    rows,files = [],{}
    for index,population in enumerate(POPULATIONS):
        for layout in range(2):
            seed = args.start_seed+100*index+layout
            for variant in VARIANTS:
                case = run(seed,population,variant)
                row = case['summary']
                name = f'{seed}-{population}-{variant}.json'
                save(name,case)
                rows.append(row);files[(seed,variant)] = name
                print('CASE',seed,population,variant,'time',round(row['total_s'],3),'cleared',row['cleared'],
                    '/',row['source_count'],'audit',row['audit']['passed'],'routes',row['route_diagnostics'],flush=True)
    a,b = rows[::2],rows[1::2]
    before,after = sum(r['total_s'] for r in a),sum(r['total_s'] for r in b)
    pairs = [dict(seed=u['seed'],population=u['population'],baseline_s=u['total_s'],candidate_s=v['total_s'],
        saving_s=u['total_s']-v['total_s'],saving_fraction=(u['total_s']-v['total_s'])/u['total_s']) for u,v in zip(a,b)]
    groups = {}
    for variant in VARIANTS:
        group = [r for r in rows if r['variant']==variant]
        groups[variant] = dict(runs=len(group),source_count=sum(r['source_count'] for r in group),
            cleared=sum(r['cleared'] for r in group),passed=sum(r['audit']['passed'] for r in group),
            mean_total_s=float(np.mean([r['total_s'] for r in group])),
            p90_total_s=float(np.quantile([r['total_s'] for r in group],.9)),
            p95_total_s=float(np.quantile([r['total_s'] for r in group],.95)),
            mean_runtime_s=float(np.mean([r['runtime_s'] for r in group])),
            mean_time_parts_s={k:float(np.mean([r['time_parts_s'][k] for r in group])) for k in group[0]['time_parts_s']},
            total_first_task_changes=sum(r['route_diagnostics']['first_task_changed'] for r in group),
            total_cloud_empty_fallbacks=sum(r['route_diagnostics']['cloud_empty_fallbacks'] for r in group))
    by_population = {}
    for p in POPULATIONS:
        rr = [r for r in pairs if r['population']==p]
        s,t = sum(r['baseline_s'] for r in rr),sum(r['candidate_s'] for r in rr)
        by_population[p] = dict(pairs=2,faster=sum(r['saving_s']>1e-7 for r in rr),slower=sum(r['saving_s'] < -1e-7 for r in rr),
            reduction_fraction=(s-t)/s)
    worst = min(pairs,key=lambda r:r['saving_fraction'])
    cases = {variant:json.loads((args.out/files[(worst['seed'],variant)]).read_text()) for variant in VARIANTS}
    da,db = (decomposition(cases[v]) for v in VARIANTS)
    differences = {reason:db.get(reason,{}).get('total_s',0)-da.get(reason,{}).get('total_s',0) for reason in set(da)|set(db)}
    save('worst_pair_decomposition.json',dict(pair=worst,baseline_by_reason=da,candidate_by_reason=db,
        candidate_minus_baseline_by_reason_s=differences,
        note='Movement belongs to arrival action; decomposition is descriptive, not an isolated causal effect.',
        input_case_hashes={v:hashlib.sha256((args.out/files[(worst['seed'],v)]).read_bytes()).hexdigest() for v in VARIANTS}))
    stable = frozen.hashes()==snapshot and hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==own_hash
    passed = stable and all(r['audit']['passed'] and r['error'] is None for r in rows)
    save('summary.json',dict(passed=passed,source_snapshot_stable=stable,model_and_parameters_unchanged=True,
        local_only=True,official_contacted=False,rows=rows,pairs=pairs,groups=groups,by_population=by_population,
        reduction_fraction=(before-after)/before,mean_saving_s=(before-after)/10,
        faster=sum(r['saving_s']>1e-7 for r in pairs),slower=sum(r['saving_s'] < -1e-7 for r in pairs),worst_pair=worst))
    print('COMPLETE',passed,'reduction',(before-after)/before,'faster',sum(r['saving_s']>1e-7 for r in pairs),flush=True)
    if not passed: raise SystemExit(1)


if __name__=='__main__':main()
