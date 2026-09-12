"""A separate frozen batch for the known-source extension; no official API."""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

from bootstrap import ROOT, baseline
from benchmark_closed import hashes as previous_hashes
from benchmark import multi_case
from benchmark_joint import sparse_case, summary
from validate_model import (BoundedArena, PublicAPI, independent_cells, independent_time_audit,
    observation_audit, certificate_audit, trace_truth_audit)
from conditional_preview import solve_conditional, ConditionalSelector


def hashes():
    result = previous_hashes()
    for name in ('conditional_worlds.py', 'conditional_preview.py', 'benchmark_conditional.py'):
        path = Path(__file__).with_name(name)
        result[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--start-seed', default=26000, type=int)
    parser.add_argument('--layouts', default=1, type=int)
    parser.add_argument('--populations', default='uniform,sparse10')
    parser.add_argument('--fields', default='smooth')
    parser.add_argument('--variants', default='lean,ledger_reference,conditional,conditional_mean')
    parser.add_argument('--split', choices=('development','holdout'), default='development')
    args = parser.parse_args()
    variants = args.variants.split(',')
    assert variants[0] == 'lean' and set(variants) <= {'lean','ledger_reference','conditional','conditional_mean'}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'cases').mkdir()
    cases = [dict(name=f'{seed}-{pop}-{field}', seed=seed, field=field,
                  sources=multi_case(seed) if pop == 'uniform' else sparse_case(seed,pop))
        for pi,pop in enumerate(args.populations.split(','))
        for seed in range(args.start_seed+100*pi,args.start_seed+100*pi+args.layouts)
        for field in args.fields.split(',')]
    before = hashes()
    manifest = dict(arguments={**vars(args),'out':str(args.out)}, cases=cases, source_sha256=before,
                    local_only=True, official_simulator_used=False, truth_hidden=True,
                    decisions=1, scenarios=6, max_candidates=3, risk_weights=[.2,0.], margin_s=5.)
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    rows, cache, centers = [], {}, independent_cells()
    for case in cases:
        reference_events = None
        for variant in variants:
            print(json.dumps(dict(starting=case['name'],variant=variant)),flush=True)
            arena = BoundedArena(case['sources'],case['seed'],case['field'],max_actions=3000)
            trace, checks, outcome, error = [], {}, {}, None
            started = time.perf_counter()
            try:
                if variant == 'lean':
                    outcome = baseline.solve_multi(PublicAPI(arena),trace=trace)
                else:
                    selector = ConditionalSelector(cache=cache,risk=0. if variant=='conditional_mean' else .2)
                    outcome = solve_conditional(PublicAPI(arena),trace=trace,
                        reference=variant=='ledger_reference',selector=selector)
                runtime = time.perf_counter()-started
                assert outcome['complete'] and arena.evaluation()['all_cleared'],outcome
                observations = observation_audit(arena)
                checks = dict(timing=independent_time_audit(arena),
                    certificate=certificate_audit(arena,outcome,observations,centers),
                    geometry=trace_truth_audit(arena,trace))
            except Exception:
                runtime = time.perf_counter()-started
                error = traceback.format_exc()
            if variant=='lean': reference_events=arena.events
            if variant=='ledger_reference': assert arena.events==reference_events,'ledger changed real actions'
            forecasts = [r for r in trace if r.get('phase')=='conditional_forecast']
            last=max((e['time_s'] for e in arena.events if e.get('clear_result')=='success'),default=0.)
            row=dict(name=case['name'],seed=case['seed'],field=case['field'],schedule=variant,
                passed=error is None,error=error,runtime_s=runtime,tail_s=arena.time_s-last,
                forecasts=len(forecasts),changed=sum(r['changed'] for r in forecasts),
                unavailable=sum(r['unavailable'] for r in forecasts), **arena.evaluation())
            rows.append(row)
            detail=dict(result=row,sources=case['sources'],outcome=outcome,independent_checks=checks,
                        events=arena.events,trace=trace)
            (args.out/'cases'/f'{case["name"]}-{variant}.json').write_text(json.dumps(detail),encoding='utf-8')
            (args.out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
            print(json.dumps({k:row[k] for k in ('name','schedule','passed','total_s','runtime_s','forecasts','changed','unavailable')}),flush=True)
    stable=before==hashes()
    result=dict(**summary(rows),source_snapshot_stable=stable)
    (args.out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
    raise SystemExit(0 if stable and all(r['passed'] for r in rows) else 1)


if __name__=='__main__': main()
