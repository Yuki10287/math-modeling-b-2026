"""Predeclared small paired development test of survey-first structure."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from survey_solver import MAIN, JOINT, ROOT, core, baseline, solve_multi, additional_scan_bearings
from benchmark import multi_case
from benchmark_joint import sparse_case, summary
from validate_model import (BoundedArena, PublicAPI, independent_cells, independent_time_audit,
                            observation_audit, certificate_audit, trace_truth_audit)


def hashes():
    names = ('baseline_solver.py', 'v1_solver.py', 'solver.py', 'geometry.py', 'belief_model.py',
             'coverage_model.py', 'scan_planning.py', 'local_policy.py', 'recovery.py',
             'environment.py', 'benchmark.py', 'validate_model.py')
    paths = [MAIN / name for name in names]
    paths += [JOINT / name for name in ('benchmark_joint.py', 'joint_solver.py', 'joint_planning.py', 'flexible_coverage.py')]
    paths += [Path(__file__).with_name(name) for name in ('survey_solver.py', 'benchmark_survey.py')]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def behavior_audit(trace):
    selections = [r for r in trace if r.get('phase') == 'survey_selection']
    assert all(not r['unknown_channels'] or r['chosen']['kind'] == 'scan' for r in selections)
    extras = [r for r in trace if r.get('phase') == 'survey_additional_bearing']
    keys = [(r['channel'], *r['position']) for r in extras]
    assert len(keys) == len(set(keys))
    for r in extras:
        assert core.reception_certified(np.asarray(r['polygon']), np.asarray(r['position']), np.asarray(r['witness']))
        assert core.nearest_certified_clear(np.asarray(r['polygon']), np.asarray(r['position'])) is None
    return dict(scan_priority_selections_checked=len(selections),
                supplemental_reception_certificates_checked=len(extras),
                supplemental_repeat_measurements=0)


def run_case(case, variant, centers):
    arena = BoundedArena(case['sources'], case['seed'], case['field'], max_actions=3000)
    trace, outcome, checks, error = [], {}, {}, None
    started = time.perf_counter()
    try:
        outcome = baseline.solve_multi(PublicAPI(arena), trace=trace) if variant == 'lean' else solve_multi(PublicAPI(arena), variant=variant, trace=trace)
        runtime = time.perf_counter() - started
        assert outcome['complete'] and arena.evaluation()['all_cleared'], outcome
        observations = observation_audit(arena)
        checks = dict(timing=independent_time_audit(arena),
                      certificate=certificate_audit(arena, outcome, observations, centers),
                      geometry=trace_truth_audit(arena, trace), behavior=behavior_audit(trace))
    except Exception:
        runtime = time.perf_counter() - started
        error = traceback.format_exc()
    last = max((e['time_s'] for e in arena.events if e.get('clear_result') == 'success'), default=0.0)
    row = dict(name=case['name'], seed=case['seed'], field=case['field'], schedule=variant,
               passed=error is None, error=error, runtime_s=runtime, tail_s=arena.time_s - last,
               supplemental_bearings=sum(r.get('phase') == 'survey_additional_bearing' for r in trace),
               **arena.evaluation())
    return dict(result=row, sources=case['sources'], outcome=outcome,
                independent_checks=checks, events=arena.events, trace=trace)


def preflight(centers):
    references = []
    for seed in (27990, 27991):
        case = dict(name=f'preflight-{seed}', seed=seed, field='smooth', sources=multi_case(seed))
        a, b = [run_case(case, variant, centers) for variant in ('lean', 'reference')]
        assert a['result']['passed'] and b['result']['passed'], (a['result']['error'], b['result']['error'])
        assert a['events'] == b['events'], 'Reference events differ from frozen lean'
        references.append(dict(seed=seed, events=len(a['events']), total_s=a['result']['total_s'], exact_event_match=True))
    # An eligibility fixture isolates every reason for skipping an extra read.
    polygon = np.array([[-100., 0.], [100., 0.]])
    model = SimpleNamespace(position=np.array([0., 200.]), channel=1, trace=[], beliefs={})
    def item(P=polygon, near=None, conflict=None):
        return dict(P=P.copy(), witness=np.array([0., 0.]), near=near, conflict=conflict)
    model.beliefs = {1: item(), 2: item(np.array([[0., 0.]])), 3: item(near=np.zeros(2)),
                     4: item(conflict='fixture'), 5: item(), 6: item(np.array([[-1600., -1000.], [1600., -1000.]]))}
    recorded, reads, clears = {(5, 0., 200.)}, [], []
    model.recorded = lambda c, q: (c, *q) in recorded
    def measure(q, c):
        reads.append(c)
        recorded.add((c, *q))
        return {'measure_result': 'direction', 'svd_deg': 0.0}
    model.measure = measure
    model.clear = lambda q, c: clears.append(c) or {'clear_result': 'success'}
    additional_scan_bearings(model)
    assert reads == [1], reads
    additional_scan_bearings(model)
    assert reads == [1], 'A supplemental stop read was repeated'
    return dict(passed=True, reference_runs=references, full_runs=4,
                supplemental_skip_conditions_checked=['already_certified_clear', 'near', 'conflict', 'already_recorded', 'no_reception_certificate'],
                repeated_supplemental_call_adds_no_measurements=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preflight-result', type=Path,
                        help='Reuse completed checks only when all model dependency hashes match')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'cases').mkdir()
    before, centers = hashes(), independent_cells()
    cases = [dict(name=f'{seed}-{kind}-{field}', seed=seed, field=field,
                  sources=multi_case(seed) if kind == 'uniform' else sparse_case(seed, kind))
             for seed, kind in ((27000, 'uniform'), (27100, 'sparse10'), (27200, 'boundary10'), (27300, 'one_side10'))
             for field in ('smooth', 'extreme')]
    manifest = dict(split='development', local_only=True, official_simulator_used=False,
                    truth_hidden_from_solver=True, source_sha256=before, cases=cases,
                    variants=['lean', 'survey', 'survey_all'], python=platform.python_version(), numpy=np.__version__,
                    independent_layouts=4, paired_conditions=8,
                    gate=dict(minimum_mean_reduction_pct=2.0, faster_must_exceed_slower=True, all_audits_must_pass=True),
                    design='Complete unknown-channel search before travel for source service; retain original free-at-stop clears and shared observations. survey_all adds all eligible certified fresh bearings at scan stops. Fixed parameters; no tuning after results.')
    (args.out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    model_hashes = {name: value for name, value in before.items() if Path(name).name != 'benchmark_survey.py'}
    if args.preflight_result:
        checks = json.loads(args.preflight_result.read_text(encoding='utf-8'))
        assert checks['passed'] and checks['model_source_sha256'] == model_hashes
    else:
        checks = preflight(centers)
        checks['model_source_sha256'] = model_hashes
    (args.out / 'preflight.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(json.dumps(dict(preflight=checks)), flush=True)
    rows = []
    for case in cases:
        for variant in manifest['variants']:
            print(json.dumps(dict(starting=case['name'], variant=variant)), flush=True)
            detail = run_case(case, variant, centers)
            row = detail['result']
            rows.append(row)
            (args.out / 'cases' / f'{case["name"]}-{variant}.json').write_text(json.dumps(detail), encoding='utf-8')
            (args.out / 'rows.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
            print(json.dumps({k: row[k] for k in ('name', 'schedule', 'passed', 'total_s', 'runtime_s', 'supplemental_bearings', 'error')}), flush=True)
    stable = before == hashes()
    result = dict(**summary(rows), source_snapshot_stable=stable, preflight_passed=checks['passed'])
    result['holdout_eligible'] = [variant for variant, pair in result['comparisons_to_lean'].items()
                                  if stable and all(r['passed'] for r in rows)
                                  and pair['reduction_pct'] >= 2.0 and pair['faster'] > pair['slower']]
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if stable and all(r['passed'] for r in rows) else 1)


if __name__ == '__main__':
    main()
