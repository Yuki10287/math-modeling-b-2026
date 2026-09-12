"""Fixed 45000 local paired ablation of real-convex-region optical execution."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
Q4 = HERE.parents[1]
PROJECT = Q4.parent.parent
sys.path.insert(0, str(Q4))
import localization
import q4_share25_client
import task_sharing_solver
from polar_cover import PolarCover
from continuous_hypothesis_experiment import (
    LocalArena, public_api, sources_for, validate_run, ENVIRONMENT_PATH,
    VALIDATION_PATH, POPULATIONS,
)
from polygon_optical_experiment import optical_plan, make_solver
from audit_polygon_optical_v1 import audit_certificate, audit_run


def write(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def hashes():
    result = {f'src/{key}': value for key, value in q4_share25_client.source_hashes().items()}
    for p in (Path(__file__), HERE/'polygon_optical_experiment.py', HERE/'audit_polygon_optical_v1.py',
              HERE/'continuous_hypothesis_experiment.py', ENVIRONMENT_PATH, VALIDATION_PATH):
        result[p.relative_to(PROJECT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def diagnostics(archive_path):
    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        for case, channel in ((1, 20), (3, 14)):
            trace = json.loads(archive.read(f'测试结果/q4-share25-practice-{case:02}.jsonl.beliefs.json'))
            start = [0., 0.]
            for row in trace:
                if row['phase'] == 'actual_action':
                    start = row['position']
                if row['phase'] == 'optical_plan' and row['channel'] == channel:
                    break
            else:
                raise RuntimeError('Expected recorded optical plan')
            P, q = np.asarray(row['polygon']), np.asarray(start)
            old = localization.optical_plan(P, q)
            began = time.perf_counter()
            new = optical_plan(P, q)
            runtime_s = time.perf_counter()-began
            audit = audit_certificate(new['optical_certificate'], row['polygon'], new['path'].tolist())
            assert np.allclose(old['path'], row['path'], rtol=0, atol=1e-8), 'saved fallback differs from frozen source'
            rows.append(dict(case=case, channel=channel, old_points=len(old['path']),
                new_points=len(new['path']), skipped_grid_cells=new['omitted_cells'],
                old_worst_s=old['worst_s'], new_worst_s=new['worst_s'],
                old_score=old['score'], new_score=new['score'], runtime_s=runtime_s,
                proof_audit=audit, certificate=new['optical_certificate'],
                saved_polygon=row['polygon'], saved_start=start,
                scores_are_existing_quadrature_not_actual_source_time=True))
    return dict(local_only=True, official_simulator_contacted=False,
        archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(), rows=rows,
        warning='Saved-state geometry only; no official counterfactual source location or time known.')


def run_case(seed, population, field, variant):
    sources = sources_for(seed, population)
    arena, trace = LocalArena(sources, seed, field), []
    result, error = None, None
    begun = time.perf_counter()
    try:
        solver = task_sharing_solver.solve_multi if variant == 'share25' else make_solver()
        result = solver(public_api(arena), variant='share25', trace=trace)
    except Exception:
        error = traceback.format_exc()
    runtime_s = time.perf_counter()-begun
    try:
        if error:
            raise RuntimeError(error)
        mesh = PolarCover()
        # Original audit checks actions/truth/timing/absence and any unchanged
        # rectangle plans. New plans use a separate phase and exact auditor.
        audit = validate_run(sources, arena.events, trace, result, mesh.stations, mesh.indices)
        audit['convex_optical'] = audit_run(arena.events, trace)
    except Exception:
        audit = dict(passed=False, error=traceback.format_exc())
    plans = [r for r in trace if r['phase'] == 'polygon_optical_plan']
    row = dict(seed=seed, population=population, field=field, variant=variant,
        runtime_s=runtime_s, audit=audit, error=error,
        polygon_optical_plans=len(plans), skipped_grid_cells=sum(r['omitted_cells'] for r in plans),
        **arena.evaluation())
    return dict(summary=row, sources=sources, events=arena.events, trace=trace, result=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--diagnostic-zip', type=Path)
    parser.add_argument('--diagnostics-only', action='store_true')
    parser.add_argument('--start-seed', type=int, default=45000)
    parser.add_argument('--populations', default=','.join(POPULATIONS))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot = hashes()
    write(args.out/'manifest.json', dict(source_sha256=snapshot, start_seed=args.start_seed,
        populations=args.populations.split(','), field='smooth', phase='development',
        official_simulator_contacted=False, local_only=True,
        changed='Only execution of an already selected optical fallback covers exact convex P instead of full rectangle.',
        unchanged='Frozen 25 stations, task routing, P, measure/optical comparison, local sensing and sharing score.',
        proof='Independent exact vertex/edge enumeration of every grid cell, external P and actual path binding.'))
    if args.diagnostic_zip:
        result = diagnostics(args.diagnostic_zip)
        write(args.out/'official_saved_geometry.json', result)
        print(json.dumps([{k: v for k, v in row.items() if k not in ('certificate', 'saved_polygon', 'saved_start')}
                          for row in result['rows']], ensure_ascii=False, indent=2), flush=True)
    if args.diagnostics_only:
        assert hashes() == snapshot
        return
    rows = []
    for population in args.populations.split(','):
        seed = args.start_seed+100*POPULATIONS.index(population)
        for variant in ('share25', 'polygon_optical25'):
            record = run_case(seed, population, 'smooth', variant)
            row = record['summary']
            rows.append(row)
            write(args.out/f'{seed}-{population}-{variant}.json', record)
            print(f'{seed} {population} {variant}: {row["total_s"]:.3f}s '
                  f'{row["cleared"]}/{row["source_count"]}, audit={row["audit"]["passed"]}, '
                  f'plans={row["polygon_optical_plans"]}, empty_cells={row["skipped_grid_cells"]}, '
                  f'wall={row["runtime_s"]:.2f}s', flush=True)
            if not row['audit']['passed']:
                write(args.out/'failed_rows.json', rows)
                raise RuntimeError(row['audit'])
    pairs = [dict(seed=a['seed'], population=a['population'], baseline_s=a['total_s'],
                  candidate_s=b['total_s'], saved_s=a['total_s']-b['total_s'])
             for a, b in zip(rows[::2], rows[1::2])]
    before, after = sum(p['baseline_s'] for p in pairs), sum(p['candidate_s'] for p in pairs)
    stable = hashes() == snapshot
    summary = dict(passed=stable and all(r['audit']['passed'] for r in rows), source_snapshot_stable=stable,
        local_only=True, official_simulator_contacted=False, phase='development', pairs=pairs,
        independent_layouts=len(pairs), mean_baseline_s=before/len(pairs), mean_candidate_s=after/len(pairs),
        saved_pct=100*(before-after)/before, faster=sum(p['saved_s']>1e-7 for p in pairs),
        slower=sum(p['saved_s']<-1e-7 for p in pairs), sources_per_variant=sum(r['source_count'] for r in rows[::2]),
        polygon_optical_plans=sum(r['polygon_optical_plans'] for r in rows),
        skipped_grid_cells=sum(r['skipped_grid_cells'] for r in rows))
    write(args.out/'rows.json', rows)
    write(args.out/'summary.json', summary)
    snapdir = args.out/'source_snapshot'
    snapdir.mkdir()
    for name in ('polygon_optical_experiment.py', 'audit_polygon_optical_v1.py', 'benchmark_polygon_optical.py'):
        (snapdir/name).write_bytes((HERE/name).read_bytes())
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    assert stable


if __name__ == '__main__':
    main()
