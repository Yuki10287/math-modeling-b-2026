"""Paired local ablation; use tools/run_study.py q4_share --script ... ."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from aspect_optical_experiment import solve_multi as candidate
from benchmark import LocalArena, public_api
from benchmark_cells import sources_for
from polar_cover import PolarCover
from shared import ROOT
from task_sharing_solver import solve_multi as baseline
from validation import validate_run


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def hashes():
    names = ('aspect_optical_experiment.py', 'benchmark_aspect_optical.py',
             'shared.py', 'localization.py', 'guarded_policy.py', 'route_planning.py',
             'polar_cover.py', 'directional_cover.py', 'solver.py', 'task_sharing_solver.py',
             'benchmark.py', 'benchmark_cells.py', 'validation.py')
    paths = [Path(__file__).parent/name for name in names]
    paths += [ROOT/'q3_model_v2/geometry.py', ROOT/'q3_model_v2/environment.py']
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--start-seed', type=int, required=True)
    parser.add_argument('--layouts', type=int, default=1)
    parser.add_argument('--fields', default='smooth')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    snapshot = hashes()
    write(out/'manifest.json', dict(arguments=vars(args), source_sha256=snapshot,
          local_only=True, official_simulator_contacted=False,
          purpose='development ablation, not an independent final performance estimate',
          changed_model='cell aspect ratio in certified rectangular optical fallback',
          unchanged='25 stations, source selection, feedback region, stopping certificate',
          populations=['uniform', 'boundary', 'cluster', 'omni_heavy', 'dir_heavy']))
    rows = []
    for pindex, population in enumerate(('uniform', 'boundary', 'cluster', 'omni_heavy', 'dir_heavy')):
        for layout in range(args.layouts):
            seed = args.start_seed+100*pindex+layout
            sources = sources_for(seed, population)
            for field in args.fields.split(','):
                for variant in ('share25', 'aspect_optical'):
                    arena, trace = LocalArena(sources, seed, field), []
                    started = time.perf_counter()
                    result = None
                    try:
                        result = (baseline(public_api(arena), variant='share25', trace=trace)
                                  if variant == 'share25' else candidate(public_api(arena), trace))
                        runtime_s = time.perf_counter()-started
                        mesh = PolarCover()
                        audit = validate_run(sources, arena.events, trace, result, mesh.stations, mesh.indices)
                    except Exception as exc:
                        runtime_s = time.perf_counter()-started
                        audit = dict(passed=False, error_type=type(exc).__name__, message=str(exc))
                    row = dict(seed=seed, population=population, field=field, variant=variant,
                               runtime_s=runtime_s, audit=audit, **arena.evaluation())
                    rows.append(row)
                    write(out/f'{seed}-{population}-{field}-{variant}.json',
                          dict(summary=row, sources=sources, result=result, events=arena.events, trace=trace))
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    if not audit['passed']:
                        write(out/'failed_rows.json', rows)
                        raise RuntimeError('local validation failed')
    differences = []
    for i in range(0, len(rows), 2):
        old, new = rows[i:i+2]
        differences.append(dict(seed=old['seed'], population=old['population'], field=old['field'],
                                old_s=old['total_s'], new_s=new['total_s'],
                                saved_s=old['total_s']-new['total_s']))
    stable = hashes() == snapshot
    write(out/'rows.json', rows)
    write(out/'summary.json', dict(passed=stable and all(r['audit']['passed'] for r in rows),
          source_snapshot_stable=stable, pairs=differences,
          independent_layouts=args.layouts*5,
          mean_baseline_s=float(np.mean([r['old_s'] for r in differences])),
          mean_candidate_s=float(np.mean([r['new_s'] for r in differences])),
          mean_saved_s=float(np.mean([r['saved_s'] for r in differences])),
          faster=sum(r['saved_s']>1e-7 for r in differences),
          slower=sum(r['saved_s']<-1e-7 for r in differences),
          local_only=True, official_simulator_contacted=False))
    if not stable:
        raise RuntimeError('source snapshot changed during the experiment')


if __name__ == '__main__':
    main()
