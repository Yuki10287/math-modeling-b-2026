"""Paired local tests of certified continuous negative-feedback contraction.

The only model edit is a contraction immediately before the frozen solver logs
each updated known-source belief. All accepted feedback, task choice, local
sensing, optical fallback, fixed 25 stations and final absence proof remain.
Proofs are retained in the trace for a separate exact auditor.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
Q4 = HERE.parents[1]
PROJECT = Q4.parent.parent
sys.path.insert(0, str(Q4))

import task_sharing_solver
import q4_share25_client
from polar_cover import PolarCover
from continuous_hypothesis_experiment import (
    LocalArena, public_api, sources_for, validate_run, ENVIRONMENT_PATH,
    VALIDATION_PATH, POPULATIONS,
)
from negative_region_boxes_v1 import contract_position
from shared import load_file

_validation = load_file('_negative_contraction_validation', VALIDATION_PATH)


def make_solver():
    parsed = ast.parse(Path(task_sharing_solver.__file__).read_text(encoding='utf-8-sig'))
    function = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'solve_multi')
    measure = next(n for n in function.body if isinstance(n, ast.FunctionDef) and n.name == 'measure')
    inserted = 0
    for node in ast.walk(measure):
        if not isinstance(node, ast.If):
            continue
        for index, statement in enumerate(node.body):
            if ast.dump(statement) == ast.dump(ast.parse('b = beliefs[c]').body[0]):
                patch = ast.parse('''
before_contraction = b.P.copy()
contracted_P, contraction_info = _contract_position(b.P, b.positives, b.negatives)
if contraction_info['applied']:
    b.P = np.asarray(contracted_P, float)
    trace.append(dict(phase='negative_contraction', channel=c, event=calls,
        original_polygon=before_contraction.tolist(), polygon=b.P.tolist(),
        positives=[p.tolist() for p in b.positives],
        negatives=[p.tolist() for p in b.negatives], info=contraction_info))
elif 'error' in contraction_info:
    trace.append(dict(phase='contraction_fallback', channel=c, event=calls,
        info=contraction_info))
''').body
                node.body[index+1:index+1] = patch
                inserted += 1
                break
    assert inserted == 1
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = task_sharing_solver.solve_multi.__globals__.copy()
    namespace['_contract_position'] = contract_position
    exec(compile(module, str(Path(__file__)), 'exec'), namespace)
    return namespace['solve_multi']


def hashes():
    result = {f'src/{k}': v for k, v in q4_share25_client.source_hashes().items()}
    paths = [Path(__file__), HERE/'negative_region_boxes_v1.py',
             HERE/'continuous_hypothesis_experiment.py', ENVIRONMENT_PATH, VALIDATION_PATH]
    for path in paths:
        result[path.relative_to(PROJECT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def contraction_trace_audit(sources, events, trace):
    """Check data provenance and hidden-source retention, not proof predicates.

    A separate auditor subsequently verifies each rational exclusion proof.
    This function does not participate in policy choice.
    """
    truth = {int(s['channel']): np.asarray(s['position']) for s in sources}
    count = 0
    for row in trace:
        if row['phase'] != 'negative_contraction':
            continue
        count += 1
        c, last = row['channel'], row['event']
        prefix = [e for e in events[:last] if e['action'] == 'measure' and e['channel'] == c]
        positive = {tuple(e['position']) for e in prefix if e['measure_result'] in ('direction', 'near')}
        negative = {tuple(e['position']) for e in prefix if e['measure_result'] == 'no_signal'}
        assert all(tuple(p) in positive for p in row['positives']), 'unobserved positive evidence'
        assert all(tuple(p) in negative for p in row['negatives']), 'unobserved negative evidence'
        assert _validation.polygon_contains(row['original_polygon'], truth[c])
        assert _validation.polygon_contains(row['polygon'], truth[c]), 'contraction excluded hidden source'
        certificate = row['info']['certificate']
        assert certificate is not None and row['info']['applied']
        assert certificate['output_polygon'] == row['polygon']
        assert certificate['positive_points'] == row['positives']
        assert certificate['negative_points'] == row['negatives']
    return dict(passed=True, contractions=count, all_used_feedback_observed=True,
                hidden_sources_retained=True, exact_certificate_check='separate independent audit required')


def run_case(seed, population, field, variant):
    sources = sources_for(seed, population)
    arena, trace = LocalArena(sources, seed, field), []
    result, error = None, None
    started = time.perf_counter()
    try:
        solver = task_sharing_solver.solve_multi if variant == 'share25' else make_solver()
        result = solver(public_api(arena), variant='share25', trace=trace)
    except Exception:
        error = traceback.format_exc()
    elapsed = time.perf_counter()-started
    try:
        if error:
            raise RuntimeError(error)
        mesh = PolarCover()
        audit = validate_run(sources, arena.events, trace, result, mesh.stations, mesh.indices)
        audit['contraction_trace'] = contraction_trace_audit(sources, arena.events, trace)
    except Exception:
        audit = dict(passed=False, error=traceback.format_exc())
    records = [r for r in trace if r['phase'] == 'negative_contraction']
    row = dict(seed=seed, population=population, field=field, variant=variant,
        runtime_s=elapsed, error=error, audit=audit,
        contractions=len(records), contraction_fallbacks=sum(r['phase'] == 'contraction_fallback' for r in trace),
        **arena.evaluation())
    return dict(summary=row, sources=sources, events=arena.events, trace=trace, result=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--start-seed', type=int, default=38000)
    parser.add_argument('--layouts', type=int, default=1)
    parser.add_argument('--fields', default='smooth')
    parser.add_argument('--populations', default=','.join(POPULATIONS))
    parser.add_argument('--phase', choices=('development', 'holdout', 'stress'), default='development')
    args = parser.parse_args()
    populations, fields = args.populations.split(','), args.fields.split(',')
    assert len(set(populations)) == len(populations) and set(populations) <= set(POPULATIONS)
    assert args.layouts > 0
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot = hashes()
    write(args.out/'manifest.json', dict(start_seed=args.start_seed, layouts_per_population=args.layouts,
        populations=populations, fields=fields, phase=args.phase, variants=['share25', 'negative_contraction25'],
        source_sha256=snapshot, local_only=True, official_simulator_contacted=False,
        changed='Certified position contraction after real positive/negative feedback only.',
        unchanged='25 stations, source scheduling, local sensing score, full optical cover, stopping proof',
        exact_certificate_audit='Separate independent auditor; not this driver.'))
    rows = []
    for population in populations:
        pindex = POPULATIONS.index(population)
        for layout in range(args.layouts):
            seed = args.start_seed+100*pindex+layout
            for field in fields:
                for variant in ('share25', 'negative_contraction25'):
                    record = run_case(seed, population, field, variant)
                    row = record['summary']
                    rows.append(row)
                    write(args.out/f'{seed}-{population}-{field}-{variant}.json', record)
                    print(f'{seed} {population} {field} {variant}: {row["total_s"]:.3f}s; '
                          f'{row["cleared"]}/{row["source_count"]}; audit={row["audit"]["passed"]}; '
                          f'contractions={row["contractions"]}; wall={row["runtime_s"]:.2f}s', flush=True)
                    if not row['audit']['passed']:
                        write(args.out/'failed_rows.json', rows)
                        raise RuntimeError(row['audit'])
    pairs = []
    for old, new in zip(rows[::2], rows[1::2]):
        pairs.append(dict(seed=old['seed'], population=old['population'], field=old['field'],
                          baseline_s=old['total_s'], candidate_s=new['total_s'], saved_s=old['total_s']-new['total_s']))
    total_old, total_new = sum(r['baseline_s'] for r in pairs), sum(r['candidate_s'] for r in pairs)
    stable = hashes() == snapshot
    summary = dict(passed=stable and all(r['audit']['passed'] for r in rows),
        source_snapshot_stable=stable, phase=args.phase, local_only=True,
        official_simulator_contacted=False, independent_layouts=args.layouts*len(populations), pairs=pairs,
        mean_baseline_s=total_old/len(pairs), mean_candidate_s=total_new/len(pairs),
        saved_pct=100*(total_old-total_new)/total_old,
        faster=sum(r['saved_s'] > 1e-7 for r in pairs), slower=sum(r['saved_s'] < -1e-7 for r in pairs),
        sources_per_variant=sum(r['source_count'] for r in rows[::2]),
        contractions=sum(r['contractions'] for r in rows[1::2]),
        exact_certificate_audit='pending independent audit')
    write(args.out/'rows.json', rows)
    write(args.out/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if stable else 1


if __name__ == '__main__':
    raise SystemExit(main())
