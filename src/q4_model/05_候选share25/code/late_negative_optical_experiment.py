"""Use certified negative-feedback contraction only after optical fallback wins.

The original belief, sensing score, task route and active-view budget stay
unchanged. A strictly smaller certified polygon is used only to construct the
already-selected optical service path. It is NEVER assigned to belief.P.
Five fixed smooth-field development pairs use seeds 41000 through 41400.
"""
import argparse
import ast
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
Q4 = HERE.parents[1]
PROJECT = Q4.parent.parent
sys.path.insert(0, str(Q4))

import task_sharing_solver
import q4_share25_client
from polar_cover import PolarCover
from shared import load_file
from continuous_hypothesis_experiment import (
    LocalArena, public_api, sources_for, validate_run, ENVIRONMENT_PATH,
    VALIDATION_PATH, POPULATIONS,
)
from negative_region_boxes_v1 import contract_position
from audit_negative_region_boxes_v1 import verify_certificate, demand

validation = load_file('_late_negative_optical_truth_validation', VALIDATION_PATH)


def make_solver():
    tree = ast.parse(Path(task_sharing_solver.__file__).read_text(encoding='utf-8-sig'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'solve_multi')
    unchanged = {n.name: ast.dump(n) for n in function.body if isinstance(n, ast.FunctionDef) and n.name != 'service'}
    outer_loop = ast.dump(next(n for n in function.body if isinstance(n, ast.For)))
    service = next(n for n in function.body if isinstance(n, ast.FunctionDef) and n.name == 'service')
    matches = 0
    for index, statement in enumerate(service.body):
        if not (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                and ast.unparse(statement.value.func) == 'trace.append'):
            continue
        payload = statement.value.args[0]
        if not (isinstance(payload, ast.Call) and any(k.arg == 'phase' and
                isinstance(k.value, ast.Constant) and k.value.value == 'optical_plan' for k in payload.keywords)):
            continue
        for keyword in payload.keywords:
            if keyword.arg == 'polygon':
                assert ast.unparse(keyword.value) == 'belief.P.tolist()'
                keyword.value = ast.parse('execution_polygon.tolist()', mode='eval').body
        patch = ast.parse('''
execution_polygon = belief.P
contracted_polygon, contraction_info = _contract_position(belief.P, belief.positives, belief.negatives)
if contraction_info['applied']:
    execution_polygon = np.asarray(contracted_polygon, float)
    plan = optical_plan(execution_polygon, np.asarray(api.position))
    trace.append(dict(phase='late_negative_optical_contraction', channel=c, event=calls,
        original_polygon=belief.P.tolist(), polygon=execution_polygon.tolist(),
        positives=[p.tolist() for p in belief.positives],
        negatives=[p.tolist() for p in belief.negatives], info=contraction_info))
elif contraction_info.get('reason') == 'proof_failed_original_region_preserved':
    trace.append(dict(phase='late_negative_optical_fallback', channel=c, event=calls,
        info=contraction_info))
''').body
        service.body[index:index] = patch
        matches += 1
        break
    assert matches == 1
    assert unchanged == {n.name: ast.dump(n) for n in function.body if isinstance(n, ast.FunctionDef) and n.name != 'service'}
    assert outer_loop == ast.dump(next(n for n in function.body if isinstance(n, ast.For)))
    # The added statements may assign the execution polygon, never belief.P.
    assert not any(isinstance(n, ast.Attribute) and n.attr == 'P' and isinstance(n.ctx, ast.Store)
                   for statement in patch for n in ast.walk(statement))
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = task_sharing_solver.solve_multi.__globals__.copy()
    namespace['_contract_position'] = contract_position
    exec(compile(module, str(Path(__file__)), 'exec'), namespace)
    return namespace['solve_multi']


def audit_late_contractions(case):
    events, trace = case['events'], case['trace']
    truths = {s['channel']: s['position'] for s in case['sources']}
    previous_beliefs, latest_event, rows = {}, None, []
    for index, row in enumerate(trace):
        if row['phase'] == 'actual_action':
            latest_event = row['event']
        if row['phase'] == 'belief':
            previous_beliefs[row['channel']] = row
        if row['phase'] != 'late_negative_optical_contraction':
            continue
        c, event = row['channel'], row['event']
        demand(type(event) is int and event == latest_event and 1 <= event <= len(events), 'late contraction event provenance')
        earlier = previous_beliefs[c]
        demand(row['original_polygon'] == earlier['polygon'], 'late contraction differs from unchanged belief')
        demand(row['positives'] == earlier['positives'] and row['negatives'] == earlier['negatives'], 'late premises differ from latest belief')
        prior = [e for e in events[:event] if e['action'] == 'measure' and e['channel'] == c]
        positive = {tuple(e['position']) for e in prior if e['measure_result'] in ('direction', 'near')}
        negative = {tuple(e['position']) for e in prior if e['measure_result'] == 'no_signal'}
        demand(all(tuple(p) in positive for p in row['positives']), 'unobserved late positive evidence')
        demand(all(tuple(p) in negative for p in row['negatives']), 'unobserved late negative evidence')
        demand(row['info']['applied'] is True, 'late contraction not applied')
        demand(index+1 < len(trace) and trace[index+1]['phase'] == 'optical_plan' and
               trace[index+1]['channel'] == c and trace[index+1]['polygon'] == row['polygon'],
               'optical execution plan not bound to contracted polygon')
        demand(validation.polygon_contains(row['original_polygon'], truths[c]), 'source outside old region')
        demand(validation.polygon_contains(row['polygon'], truths[c]), 'source excluded by late contraction')
        proof = verify_certificate(row['info']['certificate'], row['original_polygon'], row['positives'],
                                   row['negatives'], row['polygon'], require_reduction=True)
        proof.update(channel=c, event=event, true_source_retained=True,
                     original_belief_unchanged=True, optical_plan_uses_certified_region=True)
        rows.append(proof)
    return dict(passed=True, contractions=len(rows), rows=rows)


def hashes():
    values = {f'src/{k}': v for k, v in q4_share25_client.source_hashes().items()}
    paths = [Path(__file__), HERE/'negative_region_boxes_v1.py',
             HERE/'audit_negative_region_boxes_v1.py', HERE/'continuous_hypothesis_experiment.py',
             ENVIRONMENT_PATH, VALIDATION_PATH]
    for path in paths:
        values[path.relative_to(PROJECT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return values


def write(path, data, exclusive=False):
    with path.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def run_case(seed, population, variant, candidate):
    sources = sources_for(seed, population)
    arena, trace, result, error = LocalArena(sources, seed, 'smooth'), [], None, None
    started = time.perf_counter()
    try:
        result = (task_sharing_solver.solve_multi if variant == 'share25' else candidate)(
            public_api(arena), variant='share25', trace=trace)
    except Exception:
        error = traceback.format_exc()
    runtime = time.perf_counter()-started
    case = dict(sources=sources, result=result, events=arena.events, trace=trace)
    try:
        if error:
            raise AssertionError(error)
        mesh = PolarCover()
        audit = validate_run(sources, arena.events, trace, result, mesh.stations, mesh.indices)
        audit['late_contraction_proofs'] = audit_late_contractions(case)
    except Exception:
        audit = dict(passed=False, error=traceback.format_exc())
    row = dict(seed=seed, population=population, field='smooth', variant=variant,
               runtime_s=runtime, audit=audit,
               late_contractions=sum(r['phase'] == 'late_negative_optical_contraction' for r in trace),
               proof_error_fallbacks=sum(r['phase'] == 'late_negative_optical_fallback' for r in trace),
               **arena.evaluation())
    case['summary'] = row
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    candidate, snapshot = make_solver(), hashes()
    write(out/'manifest.json', dict(code_hashes=snapshot, local_only=True,
        official_simulator_contacted=False, populations=POPULATIONS,
        seeds=[41000+100*k for k in range(5)], field='smooth',
        variants=['share25', 'late_negative_optical'], python=platform.python_version(),
        numpy=np.__version__, scipy=scipy.__version__, unchanged_original_belief=True,
        contraction_depth=6, contraction_node_budget=255,
        changed='Execution-only region contraction AFTER the original optical fallback decision.',
        independent_proof_auditor='audit_negative_region_boxes_v1.verify_certificate',
        purpose='Fixed five-layout development comparison, no parameter tuning.'), exclusive=True)
    rows, pairs = [], []
    for index, population in enumerate(POPULATIONS):
        seed, cases = 41000+100*index, []
        for variant in ('share25', 'late_negative_optical'):
            case = run_case(seed, population, variant, candidate)
            cases.append(case)
            row = case['summary']
            rows.append(row)
            write(out/f'{seed}-{population}-{variant}.json', case, exclusive=True)
            write(out/'rows.json', rows)
            print(f'{seed} {population} {variant}: {row["total_s"]:.2f}s; '
                  f'{row["cleared"]}/{row["source_count"]}; audit={row["audit"]["passed"]}; '
                  f'late={row["late_contractions"]}; wall={row["runtime_s"]:.2f}s', flush=True)
        base, new = cases
        markers = [r for r in new['trace'] if r['phase'] == 'late_negative_optical_contraction']
        first_event = markers[0]['event'] if markers else len(new['events'])
        prefix_equal = base['events'][:first_event] == new['events'][:first_event]
        valid = base['summary']['audit']['passed'] and new['summary']['audit']['passed'] and prefix_equal
        a, b = base['summary']['total_s'], new['summary']['total_s']
        pairs.append(dict(seed=seed, population=population, passed=valid,
                          baseline_s=a, candidate_s=b, saved_s=a-b,
                          reduction_percent=100*(a-b)/a,
                          real_events_equal_before_first_applied_contraction=prefix_equal))
    groups = {}
    for variant in ('share25', 'late_negative_optical'):
        group = [r for r in rows if r['variant'] == variant]
        groups[variant] = dict(runs=len(group), passes=sum(r['audit']['passed'] for r in group),
            source_instances=sum(r['source_count'] for r in group), cleared=sum(r['cleared'] for r in group),
            mean_total_s=float(np.mean([r['total_s'] for r in group])),
            mean_wall_s=float(np.mean([r['runtime_s'] for r in group])),
            late_contractions=sum(r['late_contractions'] for r in group),
            proof_error_fallbacks=sum(r['proof_error_fallbacks'] for r in group))
    stable = snapshot == hashes()
    a, b = groups['share25']['mean_total_s'], groups['late_negative_optical']['mean_total_s']
    report = dict(passed=stable and all(r['passed'] for r in pairs), local_only=True,
        source_snapshot_stable=stable, groups=groups, pairs=pairs,
        reduction_percent=100*(a-b)/a, faster=sum(r['saved_s'] > 1e-7 for r in pairs),
        slower=sum(r['saved_s'] < -1e-7 for r in pairs),
        unchanged_local_decision_before_optical_fallback=True, belief_polygon_never_assigned=True)
    write(out/'summary.json', report, exclusive=True)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
