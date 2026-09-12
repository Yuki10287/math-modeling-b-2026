"""Frozen ten-new-layout three-way evaluation: share25, station22, and pruning.

The two independently selected changes are combined without tuning. Original
station22 geometry and wrapper code are reused; its private raw solve binding
adds only the frozen ordered optical deletion. No official calls.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import FunctionType

import numpy as np
import benchmark_station22_v1 as geometry
import share22_geometry_experiment_v1 as station22
import ordered_optical_prune_experiment as prune
from audit_ordered_optical_prune_v1 import verify_case as exact_prune_audit
from audit_station22_scan_invariant_v1 import audit_case as scan_audit

HERE = Path(__file__).resolve().parent
VARIANTS = ('share25', 'station22', 'station22_prune')
POPULATIONS = ('uniform','boundary','cluster','omni_heavy','dir_heavy')
GEOMETRY_FREEZE = HERE.parent/'results/station22_share_dev37000_v1/manifest.json'
PRUNE_FREEZE = HERE.parent/'results/ordered_optical_prune_dev48000/manifest.json'


def make_combination():
    raw_prune = prune.make_solver()
    private = raw_prune.__globals__.copy()
    private['PolarCover'] = station22.Station22Cover
    raw = FunctionType(raw_prune.__code__,private,'solve_station22_ordered_prune',
                       raw_prune.__defaults__,raw_prune.__closure__)
    wrapper_globals = station22.solve_multi.__globals__.copy()
    wrapper_globals['_solve'] = raw
    combined = FunctionType(station22.solve_multi.__code__,wrapper_globals,'solve_multi',
                            station22.solve_multi.__defaults__,station22.solve_multi.__closure__)
    assert combined.__code__ is station22.solve_multi.__code__
    assert raw.__code__ is raw_prune.__code__
    assert raw_prune.__globals__['PolarCover'] is geometry.PolarCover
    assert station22.solve_multi.__globals__['_solve'] is station22._solve
    return combined


def hashes():
    out = {Path(k).as_posix():v for k,v in geometry.hashes().items()}
    out.update(prune.hashes())
    for name in ('benchmark_station22_ordered_prune_v1.py','audit_ordered_optical_prune_v1.py',
                 'audit_station22_scan_invariant_v1.py'):
        path = HERE/name
        out[path.relative_to(geometry.PROJECT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def check_frozen():
    assert geometry.hashes() == json.loads(GEOMETRY_FREEZE.read_text(encoding='utf-8'))['source_hashes']
    assert prune.hashes() == json.loads(PRUNE_FREEZE.read_text(encoding='utf-8'))['code_hashes']


def run_case(seed,population,variant,combined):
    sources = geometry.sources_for(seed,population)
    arena,trace = geometry.LocalArena(sources,seed,'smooth'),[]
    solver = {'share25':geometry.baseline,'station22':geometry.candidate,'station22_prune':combined}[variant]
    before = time.perf_counter()
    result = solver(geometry.public_api(arena),variant='share25',trace=trace)
    runtime = time.perf_counter()-before
    case = dict(sources=sources,result=result,events=arena.events,trace=trace)
    if variant == 'share25':
        mesh = geometry.PolarCover()
        audit = geometry.validate_run(sources,arena.events,trace,result,mesh.stations,mesh.indices)
        audit['exact_station_cover'] = geometry.validate_exact_baseline(result,arena.events)
    else:
        exact,adapted,vertices,triangles = geometry.validate_exact_candidate(result,arena.events,geometry.CERTIFICATE)
        audit = geometry.validate_run(sources,arena.events,trace,adapted,vertices,triangles)
        audit['exact_station_cover'] = exact
    if variant == 'station22_prune':
        audit['source_retention_and_prune_execution'] = prune.audit_prune_case(case)
        audit['independent_exact_prune'] = exact_prune_audit(case)
        assert (sum(r['phase']=='optical_plan' for r in trace)
                == sum(r['phase']=='ordered_optical_prune' for r in trace)
                == audit['independent_exact_prune']['plans']), 'missing pruning proof for an optical service'
    case['summary'] = dict(seed=seed,population=population,field='smooth',variant=variant,
                           runtime_s=runtime,audit=audit,**arena.evaluation())
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    check_frozen()
    snapshot,combined = hashes(),make_combination()
    out = args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    prune.write(out/'manifest.json',dict(code_hashes=snapshot,local_only=True,official_contacted=False,
        variants=VARIANTS,populations=POPULATIONS,field='smooth',seeds=[50000+100*i+j for i in range(5) for j in range(2)],
        layouts_per_population=2,frozen_before_runs=True,parameter_tuning=False,
        geometry_development_manifest_sha256=hashlib.sha256(GEOMETRY_FREEZE.read_bytes()).hexdigest(),
        prune_development_manifest_sha256=hashlib.sha256(PRUNE_FREEZE.read_bytes()).hexdigest(),
        purpose='Independent new-layout combination evaluation; do not add previous percentage gains.'))
    rows,pairs,invariant_rows = [],[],[]
    for population_index,population in enumerate(POPULATIONS):
        for layout in range(2):
            seed,cases = 50000+100*population_index+layout,{}
            for variant in VARIANTS:
                case = run_case(seed,population,variant,combined)
                path = out/f'{seed}-{population}-{variant}.json'
                prune.write(path,case)
                invariant_rows.append(scan_audit(path))
                cases[variant] = case
                rows.append(case['summary'])
                print(f'{seed} {population} {variant}: {case["summary"]["total_s"]:.3f}s; '
                      f'{case["summary"]["cleared"]}/{case["summary"]["source_count"]}; audit=True',flush=True)
            original,new = cases['station22'],cases['station22_prune']
            replay = prune.constrained_replay(original)
            replay['independent_exact_prune'] = exact_prune_audit(original,replay)
            same = [prune.event_without_time(e) for e in replay['events']] == [prune.event_without_time(e) for e in new['events']]
            assert same, 'combined actual actions differ from original22 certified deletion'
            error = max(abs(a['time_s']-b['time_s']) for a,b in zip(replay['events'],new['events']))
            assert error < 1e-6 and original['result'] == new['result']
            prune.write(out/f'{seed}-{population}-homomorphism.json',replay)
            a,b,c = [cases[v]['summary']['total_s'] for v in VARIANTS]
            pairs.append(dict(seed=seed,population=population,share25_s=a,station22_s=b,station22_prune_s=c,
                station22_saved_s=a-b,combined_saved_s=a-c,combined_reduction_percent=100*(a-c)/a,
                prune_increment_s=b-c,prune_reduction_percent=100*(b-c)/b,
                identical_retained_actions=same,identical_completion_certificate=True,time_error_s=error,
                removed_actual=replay['removed_actual'],removed_original_only=replay['removed_original_only'],
                additional_from_contraction=replay['additional_from_contraction']))
    groups = {}
    for variant in VARIANTS:
        group = [r for r in rows if r['variant']==variant]
        values = [r['total_s'] for r in group]
        groups[variant] = dict(mean_total_s=float(np.mean(values)),maximum_total_s=max(values),
            p90_total_s=float(np.quantile(values,.9)),source_count=sum(r['source_count'] for r in group),
            cleared=sum(r['cleared'] for r in group),passes=sum(r['audit']['passed'] for r in group),
            mean_wall_s=float(np.mean([r['runtime_s'] for r in group])))
    a,b,c = [groups[v]['mean_total_s'] for v in VARIANTS]
    check_frozen()
    stable = snapshot == hashes()
    summary = dict(passed=stable and all(r['audit']['passed'] for r in rows),source_snapshot_stable=stable,
        groups=groups,pairs=pairs,combined_reduction_percent=100*(a-c)/a,
        station22_reduction_percent=100*(a-b)/a,prune_increment_percent=100*(b-c)/b,
        faster_vs25=sum(r['combined_saved_s']>1e-7 for r in pairs),slower_vs25=sum(r['combined_saved_s']< -1e-7 for r in pairs),
        ties_vs25=sum(abs(r['combined_saved_s'])<=1e-7 for r in pairs),
        faster_vs22=sum(r['prune_increment_s']>1e-7 for r in pairs),slower_vs22=sum(r['prune_increment_s']< -1e-7 for r in pairs),
        worst_pair_vs25=min(pairs,key=lambda r:r['combined_reduction_percent']),
        removed_actual=sum(r['removed_actual'] for r in pairs),official_contacted=False)
    prune.write(out/'rows.json',rows)
    prune.write(out/'scan_invariant_audit.json',dict(passed=True,cases=invariant_rows))
    prune.write(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
